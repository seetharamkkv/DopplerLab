#!/usr/bin/env python3
"""Few-hour sprint: classical Doppler-ratio speed estimation on real VS13.

Arm A: v = c * (r - 1) / (r + 1), r = f_app / f_rec from STFT ridge (GT CPA).
Arm E: leave-one-vehicle-out / pooled mean speed baseline.

Example:
  python speed_estimation/scripts/physics_speed_vs13.py
  python speed_estimation/scripts/physics_speed_vs13.py --data-dir speed_estimation/vs13 --out-dir speed_estimation/outputs/physics_vs13
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

C_SOUND = 343.0
CROP_HALF_S = 4.0
FMIN_HZ = 200.0
FMAX_HZ = 2000.0
N_FFT = 1024
HOP = 256
V_MIN_MPS = 5.0
V_MAX_MPS = 45.0
SR = 22050

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "speed_estimation" / "vs13"
DEFAULT_OUT = REPO_ROOT / "speed_estimation" / "outputs" / "physics_vs13"


def parse_annotation(txt_path: Path) -> tuple[float, float]:
    text = txt_path.read_text(encoding="utf-8").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty annotation: {txt_path}")
    if len(lines) == 1:
        parts = lines[0].split()
        if len(parts) < 2:
            raise ValueError(f"Expected 'speed cpa' in {txt_path}: {lines[0]!r}")
        return float(parts[0]), float(parts[1])
    return float(lines[0]), float(lines[1])


def build_manifest(data_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    vehicle_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    for vdir in vehicle_dirs:
        vehicle = vdir.name
        for wav in sorted(vdir.glob("*.wav")):
            if wav.stem.lower().startswith("train_valid"):
                continue
            txt = wav.with_suffix(".txt")
            if not txt.is_file():
                continue
            speed_kmh, cpa_s = parse_annotation(txt)
            rows.append(
                {
                    "clip_id": wav.stem,
                    "vehicle": vehicle,
                    "speed_kmh": float(speed_kmh),
                    "speed_mps": float(speed_kmh) / 3.6,
                    "cpa_time_s": float(cpa_s),
                    "wav_path": str(wav.resolve()),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No labeled WAV+TXT pairs under {data_dir}")
    return pd.DataFrame(rows).sort_values(["vehicle", "speed_kmh", "clip_id"]).reset_index(drop=True)


def crop_around_cpa(y: np.ndarray, sr: int, cpa_s: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    duration = n / sr
    t_abs = np.arange(n, dtype=np.float64) / sr
    start_s = max(0.0, cpa_s - CROP_HALF_S)
    end_s = min(duration, cpa_s + CROP_HALF_S)
    i0 = int(round(start_s * sr))
    i1 = int(round(end_s * sr))
    i0 = max(0, min(i0, n))
    i1 = max(i0 + 1, min(i1, n))
    y_crop = y[i0:i1]
    t_rel = t_abs[i0:i1] - cpa_s
    return y_crop, t_rel


def stft_ridge(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window="hann")
    power = np.abs(stft) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=HOP)
    band = (freqs >= FMIN_HZ) & (freqs <= FMAX_HZ)
    if not np.any(band):
        return times, np.full(len(times), np.nan)
    band_power = power[band]
    band_freqs = freqs[band]
    peak_idx = np.argmax(band_power, axis=0)
    ridge_f = band_freqs[peak_idx]
    return times, ridge_f


def doppler_ratio_speed(ridge_t: np.ndarray, ridge_f: np.ndarray) -> dict[str, float]:
    """Arm A: asymptotic approach/recede ratio → speed (m/s)."""
    out = {
        "f_app_hz": float("nan"),
        "f_rec_hz": float("nan"),
        "ratio": float("nan"),
        "pred_speed_mps": float("nan"),
        "ok": 0.0,
    }
    if len(ridge_t) < 8 or not np.isfinite(ridge_f).any():
        return out

    t0, t1 = float(ridge_t[0]), float(ridge_t[-1])
    span = t1 - t0
    if span < 1.0:
        return out

    q = 0.25 * span
    app_mask = ridge_t <= (t0 + q)
    rec_mask = ridge_t >= (t1 - q)
    if not np.any(app_mask) or not np.any(rec_mask):
        return out

    f_app = float(np.nanmedian(ridge_f[app_mask]))
    f_rec = float(np.nanmedian(ridge_f[rec_mask]))
    out["f_app_hz"] = f_app
    out["f_rec_hz"] = f_rec
    if not np.isfinite(f_app) or not np.isfinite(f_rec) or f_rec <= 1.0 or f_app <= 1.0:
        return out

    r = f_app / f_rec
    out["ratio"] = float(r)
    if r <= 1.0:
        # No Doppler rise: treat as failure rather than negative speed
        return out

    v = C_SOUND * (r - 1.0) / (r + 1.0)
    v = float(np.clip(v, V_MIN_MPS, V_MAX_MPS))
    out["pred_speed_mps"] = v
    out["ok"] = 1.0
    return out


def estimate_clip(wav_path: Path, cpa_s: float) -> dict[str, float]:
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    y_crop, t_rel = crop_around_cpa(y, sr, cpa_s)
    # Map STFT frame times onto relative crop axis (CPA at 0)
    times, ridge_f = stft_ridge(y_crop, sr)
    if len(t_rel) == 0:
        return doppler_ratio_speed(np.array([]), np.array([]))
    # STFT times are relative to crop start; shift so CPA ≈ 0
    t_cpa_in_crop = -float(t_rel[0])  # seconds from crop start to CPA
    ridge_t = times - t_cpa_in_crop
    return doppler_ratio_speed(ridge_t, ridge_f)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"mae_kmh": mae, "rmse_kmh": rmse, "r2": r2, "n": int(len(y_true))}


def run_pooled(df: pd.DataFrame) -> dict:
    valid = df["ok"].astype(bool)
    sub = df.loc[valid]
    mean_pred = np.full(len(sub), df["speed_kmh"].mean())
    arm_a = metrics(sub["speed_kmh"].to_numpy(), sub["pred_speed_kmh"].to_numpy())
    arm_e = metrics(sub["speed_kmh"].to_numpy(), mean_pred)
    return {
        "n_total": int(len(df)),
        "n_valid_armA": int(valid.sum()),
        "n_failed_armA": int((~valid).sum()),
        "armA_doppler_ratio": arm_a,
        "armE_global_mean": arm_e,
        "beats_mean": bool(arm_a["mae_kmh"] < arm_e["mae_kmh"]),
    }


def run_lovo(df: pd.DataFrame) -> dict:
    rows = []
    for vehicle in sorted(df["vehicle"].unique()):
        test = df[df["vehicle"] == vehicle]
        train = df[df["vehicle"] != vehicle]
        valid = test["ok"].astype(bool)
        test_v = test.loc[valid]
        if len(test_v) == 0:
            continue
        mean_v = float(train["speed_kmh"].mean())
        pred_mean = np.full(len(test_v), mean_v)
        m_a = metrics(test_v["speed_kmh"].to_numpy(), test_v["pred_speed_kmh"].to_numpy())
        m_e = metrics(test_v["speed_kmh"].to_numpy(), pred_mean)

        # Arm B: affine on train Arm-A successes
        train_ok = train.loc[train["ok"].astype(bool)]
        if len(train_ok) >= 5:
            x = train_ok["pred_speed_kmh"].to_numpy()
            y = train_ok["speed_kmh"].to_numpy()
            a, b = np.polyfit(x, y, 1)
            pred_b = a * test_v["pred_speed_kmh"].to_numpy() + b
            m_b = metrics(test_v["speed_kmh"].to_numpy(), pred_b)
        else:
            a = b = float("nan")
            m_b = {"mae_kmh": float("nan"), "rmse_kmh": float("nan"), "r2": float("nan"), "n": 0}

        rows.append(
            {
                "vehicle": vehicle,
                "n_test_valid": int(len(test_v)),
                "armA_mae_kmh": m_a["mae_kmh"],
                "armE_mae_kmh": m_e["mae_kmh"],
                "armB_mae_kmh": m_b["mae_kmh"],
                "armB_a": float(a),
                "armB_b": float(b),
            }
        )

    per = pd.DataFrame(rows)
    summary = {
        "n_folds": int(len(per)),
        "armA_mae_kmh_mean": float(per["armA_mae_kmh"].mean()) if len(per) else float("nan"),
        "armE_mae_kmh_mean": float(per["armE_mae_kmh"].mean()) if len(per) else float("nan"),
        "armB_mae_kmh_mean": float(per["armB_mae_kmh"].mean()) if len(per) else float("nan"),
        "per_vehicle": per.to_dict(orient="records"),
    }
    if len(per):
        summary["armA_beats_mean"] = bool(summary["armA_mae_kmh_mean"] < summary["armE_mae_kmh_mean"])
        summary["armB_beats_mean"] = bool(summary["armB_mae_kmh_mean"] < summary["armE_mae_kmh_mean"])
    return summary


def maybe_scatter(df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping scatter plot")
        return
    sub = df.loc[df["ok"].astype(bool)]
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sub["speed_kmh"], sub["pred_speed_kmh"], s=12, alpha=0.6)
    lims = [
        min(sub["speed_kmh"].min(), sub["pred_speed_kmh"].min()),
        max(sub["speed_kmh"].max(), sub["pred_speed_kmh"].max()),
    ]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("true speed (km/h)")
    ax.set_ylabel("Arm A pred (km/h)")
    ax.set_title("VS13 physics Doppler ratio")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Physics Doppler-ratio speed on VS13 (few-hour sprint)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests")
    p.add_argument("--skip-lovo", action="store_true")
    p.add_argument("--skip-plot", action="store_true")
    args = p.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        return 1

    print(f"Data: {data_dir}")
    print(f"Out:  {out_dir}")
    manifest = build_manifest(data_dir)
    if args.limit > 0:
        manifest = manifest.head(args.limit).copy()
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifest: {len(manifest)} clips -> {manifest_path}")

    records = []
    for i, row in manifest.iterrows():
        est = estimate_clip(Path(row["wav_path"]), float(row["cpa_time_s"]))
        pred_mps = est["pred_speed_mps"]
        pred_kmh = pred_mps * 3.6 if np.isfinite(pred_mps) else float("nan")
        records.append(
            {
                **row.to_dict(),
                "f_app_hz": est["f_app_hz"],
                "f_rec_hz": est["f_rec_hz"],
                "ratio": est["ratio"],
                "pred_speed_mps": pred_mps,
                "pred_speed_kmh": pred_kmh,
                "ok": int(est["ok"]),
            }
        )
        if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
            print(f"  Arm A processed {i + 1}/{len(manifest)}")

    preds = pd.DataFrame(records)
    pred_path = out_dir / "predictions_armA.csv"
    preds.to_csv(pred_path, index=False)
    print(f"Wrote {pred_path}")

    pooled = run_pooled(preds)
    summary: dict = {
        "data_dir": str(data_dir),
        "n_clips": int(len(preds)),
        "config": {
            "c_sound": C_SOUND,
            "crop_half_s": CROP_HALF_S,
            "fmin_hz": FMIN_HZ,
            "fmax_hz": FMAX_HZ,
            "n_fft": N_FFT,
            "hop": HOP,
            "cpa": "gt",
        },
        "pooled": pooled,
    }
    if not args.skip_lovo:
        print("Running LOVO (Arm A / E / B)...")
        summary["lovo"] = run_lovo(preds)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")

    if not args.skip_plot:
        scatter_path = out_dir / "scatter_armA.png"
        maybe_scatter(preds, scatter_path)
        if scatter_path.is_file():
            print(f"Wrote {scatter_path}")

    print("\n=== Pooled (valid Arm A only) ===")
    print(f"  valid/failed: {pooled['n_valid_armA']}/{pooled['n_failed_armA']}")
    print(f"  Arm A MAE: {pooled['armA_doppler_ratio']['mae_kmh']:.2f} km/h")
    print(f"  Mean MAE:  {pooled['armE_global_mean']['mae_kmh']:.2f} km/h")
    print(f"  Arm A beats mean: {pooled['beats_mean']}")
    if "lovo" in summary:
        lv = summary["lovo"]
        print("\n=== LOVO (mean over vehicles) ===")
        print(f"  Arm A MAE: {lv['armA_mae_kmh_mean']:.2f} km/h")
        print(f"  Arm E MAE: {lv['armE_mae_kmh_mean']:.2f} km/h")
        print(f"  Arm B MAE: {lv['armB_mae_kmh_mean']:.2f} km/h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
