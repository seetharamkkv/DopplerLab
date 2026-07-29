"""Richer classical features (mel bands + timing) LOVO probe — still linear/Ridge only."""
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
    out: dict[str, float] = {}

    # envelope timing
    hop = 512
    n = 1 + max(0, (len(y) - hop) // hop)
    env = np.array([np.sqrt(np.mean(y[i * hop : i * hop + hop] ** 2) + 1e-12) for i in range(n)])
    t = np.arange(n) * hop / sr
    env_db = 20 * np.log10(env + 1e-12)
    peak = int(np.argmax(env_db))
    pdb = float(env_db[peak])
    out["env_peak"] = pdb
    out["env_rms_db"] = float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12))
    for thr in (3, 6, 10, 15):
        above = env_db >= (pdb - thr)
        if np.any(above):
            idx = np.where(above)[0]
            w = float(t[idx[-1]] - t[idx[0]])
        else:
            w = float("nan")
        out[f"w{thr}"] = w
        out[f"inv_w{thr}"] = 1.0 / w if (w == w and w > 0.05) else float("nan")
        # rise/fall asymmetry
        if np.any(above):
            out[f"rise{thr}"] = float(t[peak] - t[idx[0]]) if peak >= idx[0] else float("nan")
            out[f"fall{thr}"] = float(t[idx[-1]] - t[peak]) if peak <= idx[-1] else float("nan")

    # mel band mean/std (classical spectral shape — tyre noise ~ speed)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=32, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    for b in range(32):
        out[f"mel_mean_{b}"] = float(np.mean(mel_db[b]))
        out[f"mel_std_{b}"] = float(np.std(mel_db[b]))
    # early/late halves
    mid = mel_db.shape[1] // 2
    out["mel_early_mean"] = float(np.mean(mel_db[:, :mid]))
    out["mel_late_mean"] = float(np.mean(mel_db[:, mid:]))
    out["mel_delta"] = out["mel_early_mean"] - out["mel_late_mean"]

    # spectral contrast / flatness
    flat = librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=256)
    out["flat_mean"] = float(np.mean(flat))
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)
    out["cent_mean"] = float(np.mean(cent))
    out["cent_std"] = float(np.std(cent))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=1024, hop_length=256)
    out["roll_mean"] = float(np.mean(rolloff))

    # zero-crossing / band energy ratios
    out["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    for name, lo, hi in [("lo", 50, 300), ("mid", 300, 1500), ("hi", 1500, 5000)]:
        m = (freqs >= lo) & (freqs < hi)
        out[f"e_{name}"] = float(np.log10(S[m].sum() + 1e-12))
    out["e_hi_lo"] = out["e_hi"] - out["e_lo"]
    return out


rows = []
for i, r in man.iterrows():
    f = feats(r.wav_path, float(r.cpa_time_s))
    f["speed"] = float(r.speed_kmh)
    f["vehicle"] = r.vehicle
    rows.append(f)
    if (i + 1) % 100 == 0:
        print(f"feat {i+1}", flush=True)

feat = pd.DataFrame(rows)
Xcols = [c for c in feat.columns if c not in ("speed", "vehicle") and np.isfinite(feat[c]).mean() > 0.9]
print("n_feat", len(Xcols))


def eval_split(tr, te, xcols):
    Xtr = tr[xcols].to_numpy(float)
    ytr = tr["speed"].to_numpy(float)
    Xte = te[xcols].to_numpy(float)
    yte = te["speed"].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)

    def fill(X):
        X = np.array(X, float)
        for j in range(X.shape[1]):
            bad = ~np.isfinite(X[:, j])
            X[bad, j] = med[j]
        return X

    Xtr, Xte = fill(Xtr), fill(Xte)
    sc = StandardScaler().fit(Xtr)
    model = RidgeCV(alphas=np.logspace(-2, 5, 25)).fit(sc.transform(Xtr), ytr)
    pred = model.predict(sc.transform(Xte))
    mae = float(np.mean(np.abs(pred - yte)))
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    return mae, rmse


# LOVO
maes, rmses = [], []
for v in sorted(feat.vehicle.unique()):
    mae, rmse = eval_split(feat[feat.vehicle != v], feat[feat.vehicle == v], Xcols)
    maes.append(mae)
    rmses.append(rmse)
    print(f"  {v}: MAE={mae:.2f} RMSE={rmse:.2f}")
print(f"LOVO mean MAE={np.mean(maes):.2f} RMSE={np.mean(rmses):.2f}")

# within-vehicle
maes, rmses = [], []
rng = np.random.RandomState(0)
for v in sorted(feat.vehicle.unique()):
    d = feat[feat.vehicle == v]
    idx = np.arange(len(d))
    rng.shuffle(idx)
    ntr = max(8, int(0.7 * len(d)))
    tr, te = d.iloc[idx[:ntr]], d.iloc[idx[ntr:]]
    if len(te) < 3:
        continue
    mae, rmse = eval_split(tr, te, Xcols)
    maes.append(mae)
    rmses.append(rmse)
print(f"within-vehicle MAE={np.mean(maes):.2f} RMSE={np.mean(rmses):.2f}")

out = ROOT / "speed_estimation/outputs/physics_vs13/probe_mel_features.csv"
feat.to_csv(out, index=False)
print("wrote", out)
