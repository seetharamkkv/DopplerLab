#!/usr/bin/env python3
"""Fit a parametric straight/arc orbit from audio only (Phase 3a baseline).

Examples:
  python scripts/fit_orbit.py --phase1 data/tier1_smoke/audio_clips/sample_0000000
  python scripts/fit_orbit.py --wav clip.wav --family straight --out outputs/fit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traj_reconstruction import load_phase1_sample, xy_from_state
from traj_reconstruction.parametric import fit_orbit_from_audio, plot_fit_overlay


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1", type=Path, default=None)
    p.add_argument("--wav", type=Path, default=None)
    p.add_argument("--family", choices=("straight", "arc"), default="straight")
    p.add_argument("--use-amplitude", action="store_true")
    p.add_argument("--out", type=Path, default=Path("outputs/fit_orbit.json"))
    p.add_argument("--plot", type=Path, default=Path("outputs/fit_orbit.png"))
    args = p.parse_args()

    gt_xy = None
    gt_speed = None
    gt_cpa = None
    family = args.family

    if args.phase1 is not None:
        sample = load_phase1_sample(args.phase1)
        if sample.path_family in ("straight", "arc"):
            family = sample.path_family  # type: ignore[assignment]
        src = sample.canonical_state_frames
        if src is None:
            src = sample.state_frames
        gt_xy = xy_from_state(src)
        if sample.schema and "kinematics" in sample.schema:
            gt_speed = float(sample.schema["kinematics"].get("speed_mps", 0) or 0) or None
        cpa_path = sample.root / "metadata" / "cpa_distance_m.npy"
        if cpa_path.is_file():
            gt_cpa = float(np.load(cpa_path)[0])
        if sample.wav_path is not None:
            fit = fit_orbit_from_audio(
                wav_path=sample.wav_path,
                family=family,
                use_amplitude=args.use_amplitude,
                gt_xy=gt_xy,
                gt_speed_mps=gt_speed,
                gt_cpa_distance_m=gt_cpa,
            )
        else:
            fit = fit_orbit_from_audio(
                stft_db=sample.stft_db,
                family=family,
                use_amplitude=args.use_amplitude,
                gt_xy=gt_xy,
                gt_speed_mps=gt_speed,
                gt_cpa_distance_m=gt_cpa,
            )
    elif args.wav is not None:
        fit = fit_orbit_from_audio(
            wav_path=args.wav,
            family=family,
            use_amplitude=args.use_amplitude,
        )
    else:
        raise SystemExit("provide --phase1 or --wav")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = fit.to_jsonable()
    # Store predicted path for later orbit rendering.
    payload["xy_pred"] = fit.xy_pred.tolist()
    args.out.write_text(json.dumps(payload, indent=2))
    plot_fit_overlay(fit, args.plot, gt_xy=gt_xy, title=f"Parametric {fit.family} fit")
    print(json.dumps(fit.to_jsonable(), indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
