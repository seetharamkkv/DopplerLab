"""Push classical (non-neural) feature+Ridge toward <5 km/h MAE."""
from __future__ import annotations

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

ROOT = Path(r"D:\Antigravity\DopplerLab")
man = pd.read_csv(ROOT / "speed_estimation/outputs/physics_vs13/manifest.csv")


def feats(wav: str, cpa: float) -> dict[str, float]:
    y, sr = librosa.load(wav, sr=22050, mono=True)
    i0 = max(0, int((cpa - 4) * sr))
    i1 = min(len(y), int((cpa + 4) * sr))
    y = y[i0:i1]
    out: dict[str, float] = {}

    hop = 512
    n = 1 + max(0, (len(y) - hop) // hop)
    env = np.array([np.sqrt(np.mean(y[i * hop : i * hop + hop] ** 2) + 1e-12) for i in range(n)])
    t = np.arange(n) * hop / sr
    env_db = 20 * np.log10(env + 1e-12)
    peak = int(np.argmax(env_db))
    pdb = float(env_db[peak])
    out["env_peak"] = pdb
    out["env_rms_db"] = float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12))
    for thr in (3, 6, 10, 12, 15, 20):
        above = env_db >= (pdb - thr)
        if np.any(above):
            idx = np.where(above)[0]
            w = float(t[idx[-1]] - t[idx[0]])
            out[f"w{thr}"] = w
            out[f"inv_w{thr}"] = 1.0 / w if w > 0.05 else float("nan")
            out[f"rise{thr}"] = float(max(t[peak] - t[idx[0]], 0.0))
            out[f"fall{thr}"] = float(max(t[idx[-1]] - t[peak], 0.0))
        else:
            out[f"w{thr}"] = out[f"inv_w{thr}"] = out[f"rise{thr}"] = out[f"fall{thr}"] = float("nan")

    # 64 mel means/stds + early/late
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=256, n_mels=64, fmax=10000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    T = mel_db.shape[1]
    for b in range(64):
        out[f"m{b}"] = float(np.mean(mel_db[b]))
        out[f"ms{b}"] = float(np.std(mel_db[b]))
    thirds = [mel_db[:, : T // 3], mel_db[:, T // 3 : 2 * T // 3], mel_db[:, 2 * T // 3 :]]
    for i, part in enumerate(thirds):
        out[f"mel_t{i}"] = float(np.mean(part))
        out[f"mel_hi_t{i}"] = float(np.mean(part[40:]))  # high mels
        out[f"mel_lo_t{i}"] = float(np.mean(part[:15]))

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    for i in range(20):
        out[f"mfcc{i}"] = float(np.mean(mfcc[i]))
        out[f"mfcc_s{i}"] = float(np.std(mfcc[i]))

    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=256)[0]
    out["cent_mean"] = float(np.mean(cent))
    out["cent_std"] = float(np.std(cent))
    flat = librosa.feature.spectral_flatness(y=y)[0]
    out["flat_mean"] = float(np.mean(flat))
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048, hop_length=256)[0]
    out["roll_mean"] = float(np.mean(roll))
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048, hop_length=256)[0]
    out["bw_mean"] = float(np.mean(bandwidth))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048, hop_length=256)
    for i in range(contrast.shape[0]):
        out[f"contrast{i}"] = float(np.mean(contrast[i]))

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    for name, lo, hi in [("lo", 50, 300), ("mid", 300, 1500), ("hi", 1500, 5000), ("vhi", 5000, 10000)]:
        m = (freqs >= lo) & (freqs < hi)
        out[f"e_{name}"] = float(np.log10(S[m].sum() + 1e-12))
    out["e_hi_lo"] = out["e_hi"] - out["e_lo"]
    out["e_vhi_lo"] = out["e_vhi"] - out["e_lo"]

    # Doppler-ish centroid ratio
    q = max(1, len(cent) // 4)
    out["cent_app"] = float(np.median(cent[:q]))
    out["cent_rec"] = float(np.median(cent[-q:]))
    out["cent_ratio"] = out["cent_app"] / max(out["cent_rec"], 1.0)
    return out


print("Extracting...", flush=True)
rows = []
for i, r in man.iterrows():
    f = feats(r.wav_path, float(r.cpa_time_s))
    f["speed"] = float(r.speed_kmh)
    f["vehicle"] = r.vehicle
    rows.append(f)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(man)}", flush=True)

feat = pd.DataFrame(rows)
Xcols = [c for c in feat.columns if c not in ("speed", "vehicle") and np.isfinite(feat[c]).mean() > 0.95]
print("n_feat", len(Xcols), flush=True)


def predict(tr, te, xcols, model_kind="ridge"):
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
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    if model_kind == "ridge":
        model = RidgeCV(alphas=np.logspace(-2, 6, 40)).fit(Ztr, ytr)
        pred = model.predict(Zte)
    elif model_kind == "enet":
        model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 2, 20), max_iter=5000, cv=5)
        model.fit(Ztr, ytr)
        pred = model.predict(Zte)
    elif model_kind == "pca_ridge":
        n = min(40, Ztr.shape[0] - 1, Ztr.shape[1])
        pca = PCA(n_components=n).fit(Ztr)
        model = RidgeCV(alphas=np.logspace(-2, 6, 40)).fit(pca.transform(Ztr), ytr)
        pred = model.predict(pca.transform(Zte))
    else:
        raise ValueError(model_kind)
    mae = float(np.mean(np.abs(pred - yte)))
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    return mae, rmse, pred, yte


for kind in ("ridge", "pca_ridge", "enet"):
    maes, rmses = [], []
    for v in sorted(feat.vehicle.unique()):
        mae, rmse, _, _ = predict(feat[feat.vehicle != v], feat[feat.vehicle == v], Xcols, kind)
        maes.append(mae)
        rmses.append(rmse)
    print(f"LOVO {kind}: MAE={np.mean(maes):.2f} RMSE={np.mean(rmses):.2f}")

# within-vehicle ridge
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
    mae, rmse, _, _ = predict(tr, te, Xcols, "ridge")
    maes.append(mae)
    rmses.append(rmse)
print(f"within-vehicle ridge: MAE={np.mean(maes):.2f} RMSE={np.mean(rmses):.2f}")

out = ROOT / "speed_estimation/outputs/physics_vs13/probe_rich_features.csv"
feat.to_csv(out, index=False)
print("wrote", out)
