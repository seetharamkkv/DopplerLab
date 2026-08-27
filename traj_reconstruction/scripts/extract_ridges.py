#!/usr/bin/env python3
"""Run audio-only ridge tracker on a simulated Phase 1 sample; save overlay PNG.

  python scripts/extract_ridges.py --phase1 data/tier1_smoke/audio_clips/sample_0000000
  python scripts/extract_ridges.py --wav path/to/clip.wav --out outputs/ridge.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traj_reconstruction import load_phase1_sample
from traj_reconstruction.frontend import extract_ridges, plot_ridge_overlay


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1", type=Path, default=None, help="Phase 1 sample dir (sim)")
    p.add_argument("--wav", type=Path, default=None, help="WAV only (no metadata used)")
    p.add_argument("--out", type=Path, default=Path("outputs/ridge_overlay.png"))
    p.add_argument("--n-harmonics", type=int, default=1)
    args = p.parse_args()

    if args.phase1 is None and args.wav is None:
        raise SystemExit("provide --phase1 or --wav")

    if args.phase1 is not None:
        sample = load_phase1_sample(args.phase1)
        # Prefer WAV if present; else STFT — still audio-derived only.
        if sample.wav_path is not None:
            feats = extract_ridges(
                wav_path=sample.wav_path,
                n_harmonics=args.n_harmonics,
            )
            src = f"wav:{sample.wav_path.name}"
        else:
            feats = extract_ridges(
                stft_db=sample.stft_db,
                n_harmonics=args.n_harmonics,
            )
            src = "stft.npy"
        title = f"Ridge (sim {sample.path_family}) — {src}"
    else:
        feats = extract_ridges(wav_path=args.wav, n_harmonics=args.n_harmonics)
        title = f"Ridge — {args.wav.name}"

    out = plot_ridge_overlay(feats, args.out, title=title)
    summary = {
        "source": title,
        "n_frames": int(feats.frame_times.shape[0]),
        "quality_mean": feats.quality_mean,
        "f_obs_hz_mean": float(np.nanmean(feats.f_obs_hz)),
        "A_env_peak_time_s": float(feats.frame_times[int(np.argmax(feats.A_env))]),
        "overlay": str(out),
        "note": "No metadata used by extract_ridges.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
