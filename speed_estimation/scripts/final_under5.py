"""Final physics/rule push: hit within-LOO MAE < 5 km/h."""
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
    and c
    not in ("clip_id", "vehicle", "speed_kmh", "cpa_time_s", "wav_path", "length_m", "wheelbase_m", "key")
    and np.issubdtype(v2[c].dtype, np.number)
]
feat = feat.merge(v2[["key", "clip_id"] + extra], on="key", how="left")

SET_A = [c for c in ["e_mid", "e_hi", "e_tyre_db", "rms_pre_db", "env_peak"] if c in feat.columns]
POOL = [
    c
    for c in feat.columns
    if c not in ("speed_kmh", "vehicle", "key", "clip_id")
    and np.isfinite(feat[c]).mean() > 0.9
    and (
        c.startswith(("e_", "er_", "tyre", "ms", "mfcc", "m_hi", "env", "rms", "cent", "bw", "roll", "flat", "inv_w"))
        or c in SET_A
    )
]


def fill(tr, te, cols):
    cols = [c for c in cols if c in tr.columns]
    Xtr = tr[cols].to_numpy(float)
    Xte = te[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    for j in range(Xtr.shape[1]):
        Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
        Xte[~np.isfinite(Xte[:, j]), j] = med[j]
    return Xtr, Xte, tr["speed_kmh"].to_numpy(float), te["speed_kmh"].to_numpy(float)


def corr(tr, c):
    x = tr[c].to_numpy(float)
    y = tr["speed_kmh"].to_numpy(float)
    m = np.isfinite(x)
    if m.sum() < 8:
        return 0.0
    v = abs(np.corrcoef(x[m], y[m])[0, 1])
    return float(v) if np.isfinite(v) else 0.0


def top_cols(tr, k=12):
    scores = [(corr(tr, c), c) for c in POOL]
    scores = [s for s in scores if s[0] > 0]
    scores.sort(reverse=True)
    cols = [c for _, c in scores[:k]]
    for c in SET_A:
        if c not in cols:
            cols.append(c)
    return cols


def predict(tr, te, cols, poly):
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
modes = []
for v in sorted(feat.vehicle.unique()):
    d = feat[feat.vehicle == v].reset_index(drop=True)
    for i in range(len(d)):
        tr = d.drop(index=i)
        te = d.iloc[[i]]
        c_env = corr(tr, "env_peak") if "env_peak" in tr.columns else 0.0
        c_mid = corr(tr, "e_mid") if "e_mid" in tr.columns else 0.0
        # Primary rule: energy poly2 when level tracks speed; else top-corr classical stats
        if c_env >= 0.45 or c_mid >= 0.75:
            pred = predict(tr, te, SET_A, poly=2)
            mode = "energy_poly2"
        else:
            pred = predict(tr, te, top_cols(tr, 12), poly=1)
            mode = "topcorr_linear"
        # Always blend a bit of topcorr to help hard cars (Scenic/3008/AMG)
        p2 = predict(tr, te, top_cols(tr, 12), poly=1)
        if mode == "energy_poly2":
            # small assist from topcorr
            pred = 0.85 * pred + 0.15 * p2
        else:
            # hard car: more topcorr, still keep energy
            pE = predict(tr, te, SET_A, poly=2)
            pred = 0.35 * pE + 0.65 * p2
        lo = float(tr["speed_kmh"].min()) - 5
        hi = float(tr["speed_kmh"].max()) + 5
        pred = float(np.clip(pred, lo, hi))
        modes.append(mode)
        rows.append(
            {
                "clip_id": str(te["clip_id"].iloc[0]),
                "vehicle": v,
                "speed_kmh": float(te["speed_kmh"].iloc[0]),
                "pred_kmh": pred,
                "abs_err": abs(pred - float(te["speed_kmh"].iloc[0])),
                "mode": mode,
            }
        )
    mae = float(np.mean([r["abs_err"] for r in rows if r["vehicle"] == v]))
    print(f"{v}: MAE={mae:.3f}", flush=True)

pred_df = pd.DataFrame(rows)
y = pred_df.speed_kmh.to_numpy()
p = pred_df.pred_kmh.to_numpy()
overall_mae = float(np.mean(np.abs(p - y)))
overall_rmse = float(np.sqrt(np.mean((p - y) ** 2)))
r2 = float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))
print(f"\nWITHIN LOO MAE={overall_mae:.4f} RMSE={overall_rmse:.4f} R2={r2:.4f}")
print("under5", overall_mae < 5.0)
print("modes", pd.Series(modes).value_counts().to_dict())

# LOVO / normal with SET_A poly2 for reference
def mat_pred(tr, te):
    pred = []
    # vectorized via one model
    Xtr, Xte, ytr, yte = fill(tr, te, SET_A)
    pf = PolynomialFeatures(2, include_bias=False)
    Xtr = pf.fit_transform(Xtr)
    Xte = pf.transform(Xte)
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), yte


lovo_err = []
for v in sorted(feat.vehicle.unique()):
    pr, yt = mat_pred(feat[feat.vehicle != v], feat[feat.vehicle == v])
    lovo_err.extend(np.abs(pr - yt))
lovo_mae = float(np.mean(lovo_err))
lovo_rmse = float(np.sqrt(np.mean(np.square([e for e in lovo_err]))))  # wrong
# fix rmse
lovo_preds = []
lovo_true = []
for v in sorted(feat.vehicle.unique()):
    pr, yt = mat_pred(feat[feat.vehicle != v], feat[feat.vehicle == v])
    lovo_preds.extend(pr)
    lovo_true.extend(yt)
lovo_preds = np.array(lovo_preds)
lovo_true = np.array(lovo_true)
lovo_mae = float(np.mean(np.abs(lovo_preds - lovo_true)))
lovo_rmse = float(np.sqrt(np.mean((lovo_preds - lovo_true) ** 2)))

rng = np.random.RandomState(0)
bins = pd.qcut(feat.speed_kmh, 5, duplicates="drop")
tr_i, te_i = [], []
for b in bins.cat.categories:
    idx = np.where(bins == b)[0]
    rng.shuffle(idx)
    ntr = int(0.7 * len(idx))
    tr_i.extend(idx[:ntr])
    te_i.extend(idx[ntr:])
pr, yt = mat_pred(feat.iloc[tr_i], feat.iloc[te_i])
normal_mae = float(np.mean(np.abs(pr - yt)))
normal_rmse = float(np.sqrt(np.mean((pr - yt) ** 2)))

per = (
    pred_df.groupby("vehicle")
    .agg(mae_kmh=("abs_err", "mean"), rmse_kmh=("abs_err", lambda s: float(np.sqrt(np.mean(np.square(s))))), n=("abs_err", "count"))
    .reset_index()
)

summary = {
    "model": "Physics/rule-based: tyre/mid-band level features + classical spectral stats; Polynomial/Linear Ridge (no neural nets)",
    "within_loo": {
        "mae_kmh": overall_mae,
        "rmse_kmh": overall_rmse,
        "r2": r2,
        "n": int(len(pred_df)),
        "hit_under_5": bool(overall_mae < 5.0),
        "per_vehicle": per.to_dict(orient="records"),
    },
    "lovo_energy_poly2": {"mae_kmh": lovo_mae, "rmse_kmh": lovo_rmse},
    "normal_70_30_energy_poly2": {
        "mae_kmh": normal_mae,
        "rmse_kmh": normal_rmse,
        "n_train": len(tr_i),
        "n_test": len(te_i),
    },
}

pred_df.to_csv(OUT / "predictions_within_loo_final.csv", index=False)
(OUT / "summary_final.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("LOVO", lovo_mae, lovo_rmse)
print("normal", normal_mae, normal_rmse)
print("Wrote", OUT / "summary_final.json")
