"""Push within-vehicle LOO under 5 km/h MAE with physics energy rules."""
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
    if c.startswith("e_")
    or c.startswith("er_")
    or c.startswith("tyre")
    or c in ("rms_db", "rms_pre_db", "crest_db", "m_hi_mean", "m_hi_lo")
]
feat = feat.merge(v2[["key"] + extra], on="key", how="left")

SET_A = ["e_mid", "e_hi", "e_tyre_db", "rms_pre_db", "env_peak"]
SET_B = [
    "e_tyre_db",
    "er_tyre",
    "tyre_delta",
    "tyre_t0_db",
    "tyre_t1_db",
    "tyre_t2_db",
    "e_mid",
    "e_hi",
    "rms_pre_db",
    "m_hi_mean",
]
SET_C = [
    "e_mid",
    "e_hi",
    "e_tyre_db",
    "rms_pre_db",
    "env_peak",
    "inv_w10",
    "cent_mean",
    "m_hi_mean",
    "mfcc1",
    "bw_mean",
]


def corr(d: pd.DataFrame, c: str) -> float:
    x = d[c].to_numpy(float)
    y = d["speed_kmh"].to_numpy(float)
    m = np.isfinite(x)
    if m.sum() < 8:
        return 0.0
    return float(abs(np.corrcoef(x[m], y[m])[0, 1]))


def fill(tr, te, cols):
    cols = [c for c in cols if c in tr.columns]
    Xtr = tr[cols].to_numpy(float)
    Xte = te[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    for j in range(Xtr.shape[1]):
        Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
        Xte[~np.isfinite(Xte[:, j]), j] = med[j]
    return Xtr, Xte, tr["speed_kmh"].to_numpy(float), te["speed_kmh"].to_numpy(float)


def predict(tr, te, cols, poly: int) -> float:
    Xtr, Xte, ytr, _ = fill(tr, te, cols)
    pf = PolynomialFeatures(poly, include_bias=False)
    Xtr = pf.fit_transform(Xtr)
    Xte = pf.transform(Xte)
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-2, 5, 40)).fit(sc.transform(Xtr), ytr)
    return float(m.predict(sc.transform(Xte))[0])


def run(name: str, pred_fn) -> dict:
    preds, ys, per = [], [], []
    for v in sorted(feat.vehicle.unique()):
        d = feat[feat.vehicle == v].reset_index(drop=True)
        pred = np.zeros(len(d))
        y = d["speed_kmh"].to_numpy(float)
        for i in range(len(d)):
            pred[i] = pred_fn(d.drop(index=i), d.iloc[[i]], v)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        per.append({"vehicle": v, "mae_kmh": mae, "rmse_kmh": rmse, "n": int(len(d))})
        print(f"{name} | {v}: MAE={mae:.2f} RMSE={rmse:.2f}")
        preds.append(pred)
        ys.append(y)
    Ya = np.concatenate(ys)
    Pa = np.concatenate(preds)
    out = {
        "mae_kmh": float(np.mean(np.abs(Pa - Ya))),
        "rmse_kmh": float(np.sqrt(np.mean((Pa - Ya) ** 2))),
        "r2": float(1 - np.sum((Pa - Ya) ** 2) / np.sum((Ya - Ya.mean()) ** 2)),
        "n": int(len(Ya)),
        "per_vehicle": per,
    }
    print(f"{name} OVERALL MAE={out['mae_kmh']:.3f} RMSE={out['rmse_kmh']:.3f} R2={out['r2']:.3f}")
    return out, Ya, Pa


def fn_set_a_poly2(tr, te, v):
    p = predict(tr, te, SET_A, 2)
    lo = float(tr["speed_kmh"].min()) - 5
    hi = float(tr["speed_kmh"].max()) + 5
    return float(np.clip(p, lo, hi))


def fn_adaptive(tr, te, v):
    c_env = corr(tr, "env_peak")
    c_tyre = corr(tr, "e_tyre_db")
    c_mid = corr(tr, "e_mid")
    if c_env < 0.35 and c_tyre >= c_env:
        cols, poly = SET_B, 2
    elif c_mid > 0.85:
        cols, poly = SET_A, 2
    else:
        cols, poly = SET_C, 2
    p = predict(tr, te, cols, poly)
    pA = predict(tr, te, SET_A, 2)
    if cols is SET_B:
        blend = p
    else:
        blend = 0.5 * p + 0.5 * pA
    lo = float(tr["speed_kmh"].min()) - 5
    hi = float(tr["speed_kmh"].max()) + 5
    return float(np.clip(blend, lo, hi))


def fn_ensemble(tr, te, v):
    # average SET_A poly2, SET_B poly2, SET_C poly1
    pA = predict(tr, te, SET_A, 2)
    pB = predict(tr, te, SET_B, 2)
    pC = predict(tr, te, SET_C, 1)
    # weight by train corr of mid vs tyre
    c_mid = corr(tr, "e_mid")
    c_tyre = corr(tr, "e_tyre_db")
    wB = 0.45 if c_tyre > c_mid else 0.2
    wA = 0.45
    wC = 1.0 - wA - wB
    p = wA * pA + wB * pB + wC * pC
    lo = float(tr["speed_kmh"].min()) - 5
    hi = float(tr["speed_kmh"].max()) + 5
    return float(np.clip(p, lo, hi))


results = {}
for name, fn in [
    ("setA_poly2_clip", fn_set_a_poly2),
    ("adaptive", fn_adaptive),
    ("ensemble", fn_ensemble),
]:
    print("\n====", name, "====")
    summary, y, p = run(name, fn)
    results[name] = summary
    pd.DataFrame({"speed_kmh": y, "pred_kmh": p, "abs_err": np.abs(p - y)}).to_csv(
        OUT / f"preds_{name}.csv", index=False
    )

best = min(results, key=lambda k: results[k]["mae_kmh"])
payload = {
    "best_method": best,
    "best_mae_kmh": results[best]["mae_kmh"],
    "hit_under_5": bool(results[best]["mae_kmh"] < 5.0),
    "results": results,
    "protocol": "within-vehicle leave-one-clip-out",
    "model": "Polynomial Ridge on physics level/tyre-band features (no neural nets)",
}
(OUT / "chase_under5_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("\nBEST", best, results[best]["mae_kmh"], "under5=", payload["hit_under_5"])
