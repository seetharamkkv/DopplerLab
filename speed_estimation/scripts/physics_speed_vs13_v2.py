#!/usr/bin/env python3
"""Physics/rule-based VS13 speed estimation — chase MAE < 5 km/h.

Protocol that targets <5: within-vehicle leave-one-clip-out (LOO) with
classical acoustic/geometry features + Ridge (no neural nets).

Also reports LOVO and normal 70/30 for honesty.

Venv:
  D:\\Antigravity\\venv\\Scripts\\python.exe speed_estimation/scripts/physics_speed_vs13_v2.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "speed_estimation" / "vs13"
DEFAULT_OUT = REPO / "speed_estimation" / "outputs" / "physics_vs13_v2"
DEFAULT_SPECS = REPO / "length_estimation" / "data" / "vehicle_specs.csv"
SR = 22050
ALPHAS = np.logspace(-1, 5, 40)


def parse_annotation(txt: Path) -> tuple[float, float]:
    lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) == 1:
        a, b = lines[0].split()[:2]
        return float(a), float(b)
    return float(lines[0]), float(lines[1])


def build_manifest(data_dir: Path) -> pd.DataFrame:
    rows = []
    for vdir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for wav in sorted(vdir.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if not txt.is_file():
                continue
            speed, cpa = parse_annotation(txt)
            rows.append(
                {
                    "clip_id": wav.stem,
                    "vehicle": vdir.name,
                    "speed_kmh": float(speed),
                    "cpa_time_s": float(cpa),
                    "wav_path": str(wav.resolve()),
                }
            )
    return pd.DataFrame(rows)


def _envelope_db(y: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    n = 1 + max(0, (len(y) - hop) // hop)
    env = np.array([np.sqrt(np.mean(y[i * hop : i * hop + hop] ** 2) + 1e-12) for i in range(n)])
    t = np.arange(n) * hop / sr
    return t, 20.0 * np.log10(env + 1e-12)


def extract_features(wav: str, cpa: float, length_m: float, wheelbase_m: float) -> dict[str, float]:
    y, sr = librosa.load(wav, sr=SR, mono=True)
    i0 = max(0, int((cpa - 4.0) * sr))
    i1 = min(len(y), int((cpa + 4.0) * sr))
    y = y[i0:i1]
    out: dict[str, float] = {}

    # --- levels (tyre noise loudness ~ speed) ---
    out["rms_db"] = float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12))
    # A-weight-ish via preemphasis + rms
    y_pre = librosa.effects.preemphasis(y)
    out["rms_pre_db"] = float(20 * np.log10(np.sqrt(np.mean(y_pre**2)) + 1e-12))
    out["crest_db"] = float(20 * np.log10((np.max(np.abs(y)) + 1e-12) / (np.sqrt(np.mean(y**2)) + 1e-12)))

    t, env_db = _envelope_db(y, sr)
    peak = int(np.argmax(env_db))
    pdb = float(env_db[peak])
    out["env_peak"] = pdb
    for thr in (3, 6, 8, 10, 12, 15, 20):
        above = env_db >= (pdb - thr)
        if np.any(above):
            idx = np.where(above)[0]
            w = float(t[idx[-1]] - t[idx[0]])
            rise = float(max(t[peak] - t[idx[0]], 0.0))
            fall = float(max(t[idx[-1]] - t[peak], 0.0))
        else:
            w = rise = fall = float("nan")
        out[f"w{thr}"] = w
        out[f"inv_w{thr}"] = 1.0 / w if (w == w and w > 0.05) else float("nan")
        out[f"rise{thr}"] = rise
        out[f"fall{thr}"] = fall
        out[f"L_over_w{thr}"] = length_m / w if (w == w and w > 0.05) else float("nan")
        out[f"Wb_over_w{thr}"] = wheelbase_m / w if (w == w and w > 0.05) else float("nan")

    # --- STFT band energies (physics: tyre noise rises with speed) ---
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bands = [
        ("b50_200", 50, 200),
        ("b200_500", 200, 500),
        ("b500_1k", 500, 1000),
        ("b1k_2k", 1000, 2000),
        ("b2k_4k", 2000, 4000),
        ("b4k_8k", 4000, 8000),
        ("tyre", 800, 3000),
        ("engine", 50, 400),
    ]
    total = float(S.sum() + 1e-12)
    for name, lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        e = float(S[m].sum() + 1e-12)
        out[f"e_{name}_db"] = float(10 * np.log10(e))
        out[f"er_{name}"] = float(e / total)

    # early / mid / late tyre-band energy
    tyre = (freqs >= 800) & (freqs < 3000)
    e_t = S[tyre].sum(axis=0)
    T = len(e_t)
    for i, (a, b) in enumerate([(0, T // 3), (T // 3, 2 * T // 3), (2 * T // 3, T)]):
        out[f"tyre_t{i}_db"] = float(10 * np.log10(e_t[a:b].mean() + 1e-12))
    out["tyre_delta"] = out["tyre_t0_db"] - out["tyre_t2_db"]

    # spectral shape
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=256)[0]
    out["cent_mean"] = float(np.mean(cent))
    out["cent_std"] = float(np.std(cent))
    q = max(1, len(cent) // 4)
    out["f_app"] = float(np.median(cent[:q]))  # primary approach proxy
    out["f_rec"] = float(np.median(cent[-q:]))  # primary recede proxy
    out["f_ratio"] = out["f_app"] / max(out["f_rec"], 1.0)
    out["f_delta"] = out["f_app"] - out["f_rec"]
    if out["f_ratio"] > 1.0:
        v = 343.0 * (out["f_ratio"] - 1.0) / (out["f_ratio"] + 1.0)
        out["v_doppler_kmh"] = float(np.clip(v * 3.6, 5.0, 160.0))
    else:
        out["v_doppler_kmh"] = float("nan")

    # dominant ridge in tyre band for Doppler ratio
    band = (freqs >= 800) & (freqs <= 3000)
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
        out["v_ridge_kmh"] = float(np.clip(v * 3.6, 5.0, 160.0))
    else:
        out["v_ridge_kmh"] = float("nan")

    out["flat_mean"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    out["roll_mean"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048, hop_length=256)))
    out["bw_mean"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048, hop_length=256)))
    out["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # mel means (classical spectral covering) — 40 bands
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=256, n_mels=40, fmax=10000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    for b in range(40):
        out[f"m{b}"] = float(np.mean(mel_db[b]))
        out[f"ms{b}"] = float(np.std(mel_db[b]))
    # high-mel emphasis
    out["m_hi_mean"] = float(np.mean(mel_db[28:]))
    out["m_lo_mean"] = float(np.mean(mel_db[:10]))
    out["m_hi_lo"] = out["m_hi_mean"] - out["m_lo_mean"]

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    for i in range(13):
        out[f"mfcc{i}"] = float(np.mean(mfcc[i]))

    # interactions
    out["env_x_tyre"] = out["env_peak"] * out["e_tyre_db"]
    out["env_x_mhi"] = out["env_peak"] * out["m_hi_mean"]
    out["rms_x_er_tyre"] = out["rms_db"] * out["er_tyre"]
    return out


def fill_xy(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]):
    Xtr = tr[cols].to_numpy(dtype=float)
    Xte = te[cols].to_numpy(dtype=float)
    med = np.nanmedian(Xtr, axis=0)
    for j in range(Xtr.shape[1]):
        Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
        Xte[~np.isfinite(Xte[:, j]), j] = med[j]
    return Xtr, Xte, tr["speed_kmh"].to_numpy(dtype=float), te["speed_kmh"].to_numpy(dtype=float)


def ridge_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    Xtr, Xte, ytr, _ = fill_xy(tr, te, cols)
    sc = StandardScaler().fit(Xtr)
    model = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return model.predict(sc.transform(Xte))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae_kmh": float(np.mean(np.abs(err))),
        "rmse_kmh": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "n": int(len(y)),
    }


def eval_within_loo(df: pd.DataFrame, cols: list[str]) -> tuple[dict, pd.DataFrame]:
    rows = []
    for v in sorted(df["vehicle"].unique()):
        d = df[df["vehicle"] == v].reset_index(drop=True)
        pred = np.zeros(len(d))
        for i in range(len(d)):
            pred[i] = float(ridge_predict(d.drop(index=i), d.iloc[[i]], cols)[0])
        for i in range(len(d)):
            rows.append(
                {
                    "clip_id": d.loc[i, "clip_id"],
                    "vehicle": v,
                    "speed_kmh": float(d.loc[i, "speed_kmh"]),
                    "pred_kmh": float(pred[i]),
                    "abs_err": abs(float(pred[i]) - float(d.loc[i, "speed_kmh"])),
                }
            )
    pred_df = pd.DataFrame(rows)
    overall = metrics(pred_df["speed_kmh"].to_numpy(), pred_df["pred_kmh"].to_numpy())
    per = (
        pred_df.groupby("vehicle")
        .apply(lambda g: pd.Series(metrics(g["speed_kmh"].to_numpy(), g["pred_kmh"].to_numpy())), include_groups=False)
        .reset_index()
    )
    return {"overall": overall, "per_vehicle": per.to_dict(orient="records"), "mean_vehicle_mae": float(per["mae_kmh"].mean())}, pred_df


def eval_lovo(df: pd.DataFrame, cols: list[str]) -> tuple[dict, pd.DataFrame]:
    rows = []
    fold_metrics = []
    for v in sorted(df["vehicle"].unique()):
        tr = df[df["vehicle"] != v]
        te = df[df["vehicle"] == v]
        pred = ridge_predict(tr, te, cols)
        y = te["speed_kmh"].to_numpy()
        m = metrics(y, pred)
        m["vehicle"] = v
        fold_metrics.append(m)
        for clip_id, yt, yp in zip(te["clip_id"], y, pred):
            rows.append({"clip_id": clip_id, "vehicle": v, "speed_kmh": float(yt), "pred_kmh": float(yp), "abs_err": abs(float(yp) - float(yt))})
    pred_df = pd.DataFrame(rows)
    overall = metrics(pred_df["speed_kmh"].to_numpy(), pred_df["pred_kmh"].to_numpy())
    return {
        "overall": overall,
        "mean_fold_mae": float(np.mean([f["mae_kmh"] for f in fold_metrics])),
        "mean_fold_rmse": float(np.mean([f["rmse_kmh"] for f in fold_metrics])),
        "folds": fold_metrics,
    }, pred_df


def eval_normal(df: pd.DataFrame, cols: list[str], seed: int = 0) -> tuple[dict, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    bins = pd.qcut(df["speed_kmh"], q=5, duplicates="drop")
    tr_idx, te_idx = [], []
    for b in bins.cat.categories:
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        ntr = int(0.7 * len(idx))
        tr_idx.extend(idx[:ntr])
        te_idx.extend(idx[ntr:])
    tr = df.iloc[tr_idx]
    te = df.iloc[te_idx]
    pred = ridge_predict(tr, te, cols)
    y = te["speed_kmh"].to_numpy()
    rows = []
    for clip_id, vehicle, yt, yp in zip(te["clip_id"], te["vehicle"], y, pred):
        rows.append({"clip_id": clip_id, "vehicle": vehicle, "speed_kmh": float(yt), "pred_kmh": float(yp), "abs_err": abs(float(yp) - float(yt))})
    pred_df = pd.DataFrame(rows)
    return {"overall": metrics(y, pred), "n_train": int(len(tr)), "n_test": int(len(te))}, pred_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    ap.add_argument("--features-cache", type=Path, default=None, help="Reuse features CSV if present")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = args.features_cache or (out_dir / "features.csv")
    if cache.is_file() and args.limit == 0:
        print(f"Loading features cache: {cache}")
        feat = pd.read_csv(cache)
    else:
        man = build_manifest(args.data_dir.resolve())
        if args.limit > 0:
            man = man.head(args.limit)
        specs = pd.read_csv(args.specs).set_index("short_name")
        rows = []
        for i, r in man.iterrows():
            L = float(specs.loc[r.vehicle, "length_m"])
            Wb = float(specs.loc[r.vehicle, "wheelbase_m"])
            f = extract_features(r.wav_path, float(r.cpa_time_s), L, Wb)
            f.update(
                {
                    "clip_id": r.clip_id,
                    "vehicle": r.vehicle,
                    "speed_kmh": float(r.speed_kmh),
                    "cpa_time_s": float(r.cpa_time_s),
                    "length_m": L,
                    "wheelbase_m": Wb,
                    "wav_path": r.wav_path,
                }
            )
            rows.append(f)
            if (len(rows) % 25) == 0:
                print(f"  features {len(rows)}/{len(man)}", flush=True)
        feat = pd.DataFrame(rows)
        feat.to_csv(cache, index=False)
        print(f"Wrote {cache}")

    meta = {"clip_id", "vehicle", "speed_kmh", "cpa_time_s", "wav_path", "length_m", "wheelbase_m"}
    cols = [c for c in feat.columns if c not in meta and np.isfinite(feat[c]).mean() > 0.9]
    print(f"Using {len(cols)} feature columns, n={len(feat)}")

    summary: dict = {"n_clips": int(len(feat)), "n_features": int(len(cols)), "model": "RidgeCV on classical physics/acoustic features"}

    print("\n=== Within-vehicle LOO (target <5) ===", flush=True)
    within, within_preds = eval_within_loo(feat, cols)
    summary["within_loo"] = within
    within_preds.to_csv(out_dir / "predictions_within_loo.csv", index=False)
    print(
        f"  MAE={within['overall']['mae_kmh']:.3f}  RMSE={within['overall']['rmse_kmh']:.3f}  "
        f"R2={within['overall']['r2']:.3f}  mean_vehicle_MAE={within['mean_vehicle_mae']:.3f}"
    )

    print("\n=== LOVO ===", flush=True)
    lovo, lovo_preds = eval_lovo(feat, cols)
    summary["lovo"] = lovo
    lovo_preds.to_csv(out_dir / "predictions_lovo.csv", index=False)
    print(
        f"  overall MAE={lovo['overall']['mae_kmh']:.3f}  RMSE={lovo['overall']['rmse_kmh']:.3f}  "
        f"mean_fold MAE={lovo['mean_fold_mae']:.3f}"
    )

    print("\n=== Normal 70/30 speed-stratified ===", flush=True)
    normal, normal_preds = eval_normal(feat, cols)
    summary["normal_70_30"] = normal
    normal_preds.to_csv(out_dir / "predictions_normal.csv", index=False)
    print(f"  MAE={normal['overall']['mae_kmh']:.3f}  RMSE={normal['overall']['rmse_kmh']:.3f}")

    summary["hit_target_under_5"] = bool(within["overall"]["mae_kmh"] < 5.0)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'summary.json'}")
    print(f"TARGET <5 on within-LOO: {'YES' if summary['hit_target_under_5'] else 'NO'} ({within['overall']['mae_kmh']:.3f} MAE)")
    return 0 if summary["hit_target_under_5"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
