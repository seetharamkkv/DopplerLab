"""Probe LOVO MAE of physics feature groups on VS13 (venv-only)."""
from __future__ import annotations

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"D:\Antigravity\DopplerLab")
man = pd.read_csv(ROOT / "speed_estimation/outputs/physics_vs13/manifest.csv")


def feats(wav: str, cpa: float) -> dict[str, float]:
    y, sr = librosa.load(wav, sr=22050, mono=True)
    i0 = max(0, int((cpa - 4) * sr))
    i1 = min(len(y), int((cpa + 4) * sr))
    y = y[i0:i1]
    hop = 512
    n = 1 + max(0, (len(y) - hop) // hop)
    env = np.array([np.sqrt(np.mean(y[i * hop : i * hop + hop] ** 2) + 1e-12) for i in range(n)])
    t = np.arange(n) * hop / sr
    env_db = 20 * np.log10(env + 1e-12)
    peak = int(np.argmax(env_db))
    pdb = env_db[peak]
    out: dict[str, float] = {}
    for thr in (3, 6, 10, 15):
        above = env_db >= (pdb - thr)
        if np.any(above):
            idx = np.where(above)[0]
            w = float(t[idx[-1]] - t[idx[0]])
        else:
            w = float("nan")
        out[f"w{thr}"] = w
        out[f"inv_w{thr}"] = (1.0 / w) if (w == w and w > 0.05) else float("nan")

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    cen = (S * freqs[:, None]).sum(0) / (S.sum(0) + 1e-12)
    q = max(1, len(cen) // 4)
    f_app = float(np.median(cen[:q]))
    f_rec = float(np.median(cen[-q:]))
    out["cen_app"] = f_app
    out["cen_rec"] = f_rec
    out["cen_ratio"] = f_app / max(f_rec, 1.0)
    out["cen_delta"] = f_app - f_rec

    hi = (freqs >= 1000) & (freqs <= 4000)
    e_hi = S[hi].sum(0)
    out["hi_early"] = float(np.mean(e_hi[:q]))
    out["hi_late"] = float(np.mean(e_hi[-q:]))
    out["hi_ratio"] = out["hi_early"] / (out["hi_late"] + 1e-12)
    out["log_hi_late"] = float(np.log10(out["hi_late"] + 1e-12))
    out["env_peak"] = float(pdb)
    out["env_rms_db"] = float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12))

    band = (freqs >= 200) & (freqs <= 2000)
    bp = S[band]
    bf = freqs[band]
    ridge = bf[np.argmax(bp, axis=0)]
    r_app = float(np.median(ridge[:q]))
    r_rec = float(np.median(ridge[-q:]))
    out["ridge_app"] = r_app
    out["ridge_rec"] = r_rec
    out["ridge_ratio"] = r_app / max(r_rec, 1.0)
    if out["ridge_ratio"] > 1.0:
        v = 343.0 * (out["ridge_ratio"] - 1.0) / (out["ridge_ratio"] + 1.0)
        out["v_doppler_kmh"] = float(v * 3.6)
    else:
        out["v_doppler_kmh"] = float("nan")
    return out


rows = []
for i, r in man.iterrows():
    f = feats(r.wav_path, float(r.cpa_time_s))
    f["speed"] = float(r.speed_kmh)
    f["vehicle"] = r.vehicle
    rows.append(f)
    if (i + 1) % 100 == 0:
        print(f"feat {i+1}/{len(man)}", flush=True)

feat = pd.DataFrame(rows)
cols = [c for c in feat.columns if c not in ("speed", "vehicle")]
Xcols = [c for c in cols if np.isfinite(feat[c]).mean() > 0.85]
print("features", Xcols)


def lovo(xcols: list[str]) -> tuple[float, float]:
    maes, rmses = [], []
    for v in sorted(feat.vehicle.unique()):
        tr = feat.vehicle != v
        te = feat.vehicle == v
        Xtr = feat.loc[tr, xcols].to_numpy(dtype=float)
        ytr = feat.loc[tr, "speed"].to_numpy(dtype=float)
        Xte = feat.loc[te, xcols].to_numpy(dtype=float)
        yte = feat.loc[te, "speed"].to_numpy(dtype=float)
        med = np.nanmedian(Xtr, axis=0)

        def fill(X: np.ndarray) -> np.ndarray:
            X = np.array(X, dtype=float)
            for j in range(X.shape[1]):
                bad = ~np.isfinite(X[:, j])
                X[bad, j] = med[j]
            return X

        Xtr, Xte = fill(Xtr), fill(Xte)
        sc = StandardScaler().fit(Xtr)
        model = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(sc.transform(Xtr), ytr)
        pred = model.predict(sc.transform(Xte))
        maes.append(float(np.mean(np.abs(pred - yte))))
        rmses.append(float(np.sqrt(np.mean((pred - yte) ** 2))))
    return float(np.mean(maes)), float(np.mean(rmses))


groups = {
    "level": ["env_peak", "env_rms_db", "log_hi_late"],
    "timing": ["w3", "w6", "w10", "w15", "inv_w3", "inv_w6", "inv_w10", "inv_w15"],
    "doppler": ["cen_ratio", "cen_delta", "ridge_ratio", "v_doppler_kmh"],
    "all": Xcols,
}
for name, xc in groups.items():
    xc = [c for c in xc if c in Xcols]
    mae, rmse = lovo(xc)
    print(f"{name}: LOVO MAE={mae:.2f} RMSE={rmse:.2f} n_feat={len(xc)}")

# env_peak affine
maes = []
for v in sorted(feat.vehicle.unique()):
    tr = feat.vehicle != v
    te = feat.vehicle == v
    a, b = np.polyfit(feat.loc[tr, "env_peak"], feat.loc[tr, "speed"], 1)
    pred = a * feat.loc[te, "env_peak"].to_numpy() + b
    yte = feat.loc[te, "speed"].to_numpy()
    maes.append(float(np.mean(np.abs(pred - yte))))
print(f"env_peak affine LOVO MAE={float(np.mean(maes)):.2f}")

mean_maes = []
for v in sorted(feat.vehicle.unique()):
    mu = float(feat.loc[feat.vehicle != v, "speed"].mean())
    yte = feat.loc[feat.vehicle == v, "speed"].to_numpy()
    mean_maes.append(float(np.mean(np.abs(yte - mu))))
print(f"mean baseline LOVO MAE={float(np.mean(mean_maes)):.2f}")

out = ROOT / "speed_estimation/outputs/physics_vs13/probe_features.csv"
feat.to_csv(out, index=False)
print(f"wrote {out}")
