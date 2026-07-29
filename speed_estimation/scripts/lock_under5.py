"""Lock within-vehicle LOO MAE < 5 using physics/classical features + Ridge."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\Antigravity\DopplerLab")
OUT = ROOT / "speed_estimation" / "outputs" / "physics_vs13_v2"
OUT.mkdir(parents=True, exist_ok=True)

feat = pd.read_csv(ROOT / "speed_estimation/outputs/physics_vs13/probe_rich_features.csv").rename(
    columns={"speed": "speed_kmh"}
)
v2 = pd.read_csv(OUT / "features.csv")
feat["key"] = feat.vehicle + "_" + feat.groupby("vehicle").cumcount().astype(str)
v2["key"] = v2.vehicle + "_" + v2.groupby("vehicle").cumcount().astype(str)
extra = [
    c
    for c in v2.columns
    if c not in feat.columns
    and c not in ("clip_id", "vehicle", "speed_kmh", "cpa_time_s", "wav_path", "length_m", "wheelbase_m", "key")
    and np.issubdtype(v2[c].dtype, np.number)
]
feat = feat.merge(v2[["key", "clip_id"] + extra], on="key", how="left")

POOL = [
    c
    for c in feat.columns
    if c
    not in (
        "speed_kmh",
        "vehicle",
        "key",
        "clip_id",
    )
    and np.isfinite(feat[c]).mean() > 0.9
]
# prefer physics/level/tyre + mel stats
PREFERRED = [
    c
    for c in POOL
    if c.startswith(("e_", "er_", "tyre", "ms", "mfcc", "m_hi", "env", "rms", "cent", "bw", "roll", "flat", "inv_w"))
    or c in ("env_peak", "env_rms_db")
]
if len(PREFERRED) < 20:
    PREFERRED = POOL

SET_A = ["e_mid", "e_hi", "e_tyre_db", "rms_pre_db", "env_peak"]
SET_A = [c for c in SET_A if c in feat.columns]


def fill(tr, te, cols):
    cols = [c for c in cols if c in tr.columns]
    Xtr = tr[cols].to_numpy(float)
    Xte = te[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    for j in range(Xtr.shape[1]):
        Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
        Xte[~np.isfinite(Xte[:, j]), j] = med[j]
    return Xtr, Xte, tr["speed_kmh"].to_numpy(float), te["speed_kmh"].to_numpy(float)


def top_corr_cols(tr: pd.DataFrame, pool: list[str], k: int = 12) -> list[str]:
    scores = []
    y = tr["speed_kmh"].to_numpy(float)
    for c in pool:
        x = tr[c].to_numpy(float)
        m = np.isfinite(x)
        if m.sum() < 8:
            continue
        cv = abs(np.corrcoef(x[m], y[m])[0, 1])
        if np.isfinite(cv):
            scores.append((cv, c))
    scores.sort(reverse=True)
    cols = [c for _, c in scores[:k]]
    # always include SET_A if present
    for c in SET_A:
        if c in tr.columns and c not in cols:
            cols.append(c)
    return cols


def predict_ridge(tr, te, cols, poly: int) -> float:
    Xtr, Xte, ytr, _ = fill(tr, te, cols)
    if poly > 1:
        pf = PolynomialFeatures(poly, include_bias=False)
        Xtr = pf.fit_transform(Xtr)
        Xte = pf.transform(Xte)
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(Xtr), ytr)
    p = float(m.predict(sc.transform(Xte))[0])
    lo = float(ytr.min()) - 5.0
    hi = float(ytr.max()) + 5.0
    return float(np.clip(p, lo, hi))


rows = []
for v in sorted(feat.vehicle.unique()):
    d = feat[feat.vehicle == v].reset_index(drop=True)
    for i in range(len(d)):
        tr = d.drop(index=i)
        te = d.iloc[[i]]
        # Candidate A: fixed energy poly2
        pA = predict_ridge(tr, te, SET_A, poly=2)
        # Candidate B: top-corr classical features poly1
        colsB = top_corr_cols(tr, PREFERRED, k=12)
        pB = predict_ridge(tr, te, colsB, poly=1)
        # Candidate C: top-corr poly2 on first 6 only (avoid explosion)
        colsC = top_corr_cols(tr, PREFERRED, k=6)
        pC = predict_ridge(tr, te, colsC, poly=2)

        # pick using train self-consistency: which candidate has lower MAE on a quick holdout of tr
        # score each by fitting on tr without last row
        def score(cols, poly):
            if len(tr) < 10:
                return 0.0
            te2 = tr.iloc[[-1]]
            tr2 = tr.iloc[:-1]
            p = predict_ridge(tr2, te2, cols if poly != 2 or len(cols) <= 6 else cols[:6], poly)
            return abs(p - float(te2["speed_kmh"].iloc[0]))

        sA = score(SET_A, 2)
        sB = score(colsB, 1)
        sC = score(colsC, 2)
        # blend weights inverse to holdout error
        eps = 1e-3
        wA, wB, wC = 1 / (sA + eps), 1 / (sB + eps), 1 / (sC + eps)
        wsum = wA + wB + wC
        pred = (wA * pA + wB * pB + wC * pC) / wsum
        # soft clip again
        lo = float(tr["speed_kmh"].min()) - 5
        hi = float(tr["speed_kmh"].max()) + 5
        pred = float(np.clip(pred, lo, hi))

        rows.append(
            {
                "clip_id": te["clip_id"].iloc[0] if "clip_id" in te.columns else f"{v}_{i}",
                "vehicle": v,
                "speed_kmh": float(te["speed_kmh"].iloc[0]),
                "pred_kmh": pred,
                "abs_err": abs(pred - float(te["speed_kmh"].iloc[0])),
                "pA": pA,
                "pB": pB,
                "pC": pC,
            }
        )
    print(f"done {v}", flush=True)

pred_df = pd.DataFrame(rows)
y = pred_df["speed_kmh"].to_numpy()
p = pred_df["pred_kmh"].to_numpy()
overall = {
    "mae_kmh": float(np.mean(np.abs(p - y))),
    "rmse_kmh": float(np.sqrt(np.mean((p - y) ** 2))),
    "r2": float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)),
    "n": int(len(y)),
}
per = (
    pred_df.groupby("vehicle")
    .apply(
        lambda g: pd.Series(
            {
                "mae_kmh": float(np.mean(np.abs(g.pred_kmh - g.speed_kmh))),
                "rmse_kmh": float(np.sqrt(np.mean((g.pred_kmh - g.speed_kmh) ** 2))),
                "n": int(len(g)),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)

# Also compute LOVO and normal with SET_A poly2 for reporting
def ridge_matrix(tr, te, cols, poly=2):
    Xtr, Xte, ytr, yte = fill(tr, te, cols)
    pf = PolynomialFeatures(poly, include_bias=False)
    Xtr = pf.fit_transform(Xtr)
    Xte = pf.transform(Xte)
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), yte


lovo_rows = []
for v in sorted(feat.vehicle.unique()):
    tr = feat[feat.vehicle != v]
    te = feat[feat.vehicle == v]
    pred, yte = ridge_matrix(tr, te, SET_A, 2)
    for clip, yt, yp in zip(te.get("clip_id", te.index), yte, pred):
        lovo_rows.append({"vehicle": v, "speed_kmh": float(yt), "pred_kmh": float(yp)})
lovo_df = pd.DataFrame(lovo_rows)
lovo = {
    "mae_kmh": float(np.mean(np.abs(lovo_df.pred_kmh - lovo_df.speed_kmh))),
    "rmse_kmh": float(np.sqrt(np.mean((lovo_df.pred_kmh - lovo_df.speed_kmh) ** 2))),
}

rng = np.random.RandomState(0)
bins = pd.qcut(feat["speed_kmh"], 5, duplicates="drop")
tr_i, te_i = [], []
for b in bins.cat.categories:
    idx = np.where(bins == b)[0]
    rng.shuffle(idx)
    ntr = int(0.7 * len(idx))
    tr_i.extend(idx[:ntr])
    te_i.extend(idx[ntr:])
tr = feat.iloc[tr_i]
te = feat.iloc[te_i]
pred, yte = ridge_matrix(tr, te, SET_A, 2)
normal = {
    "mae_kmh": float(np.mean(np.abs(pred - yte))),
    "rmse_kmh": float(np.sqrt(np.mean((pred - yte) ** 2))),
    "n_train": int(len(tr)),
    "n_test": int(len(te)),
}

summary = {
    "protocol_primary": "within-vehicle leave-one-clip-out",
    "model": "Physics/classical level+tyre+mel-stat features; polynomial Ridge blend (no neural nets)",
    "within_loo": {
        "overall": overall,
        "mean_vehicle_mae": float(per["mae_kmh"].mean()),
        "per_vehicle": per.to_dict(orient="records"),
        "hit_under_5": bool(overall["mae_kmh"] < 5.0),
    },
    "lovo_setA_poly2": lovo,
    "normal_70_30_setA_poly2": normal,
}

pred_df.to_csv(OUT / "predictions_within_loo_under5.csv", index=False)
(OUT / "summary_under5.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary["within_loo"]["overall"], indent=2))
print("hit_under_5", summary["within_loo"]["hit_under_5"])
print("LOVO", lovo)
print("normal", normal)
for _, r in per.sort_values("mae_kmh", ascending=False).iterrows():
    print(f"  {r.vehicle}: {r.mae_kmh:.2f}")
