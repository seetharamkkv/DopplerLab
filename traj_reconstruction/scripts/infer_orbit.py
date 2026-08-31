#!/usr/bin/env python3
"""Infer observer-centered trajectory orbit from audio (Phase 3b).

Uses either:
  --mode flexible  physics freeform fit (no checkpoint)
  --mode seq        trained ridge-track 1D CNN (f_obs + A_env)
  --mode cnn        trained complex-STFT 2D CNN
  --mode mlp        legacy flattened log-magnitude MLP

Always audio/STFT only — no metadata.

Examples:
  python scripts/infer_orbit.py --phase1 data/tier1_smoke/audio_clips/sample_0000002 \\
      --mode flexible --out outputs/infer_flexible.json
  python scripts/infer_orbit.py --phase1 ... --mode cnn \\
      --checkpoint checkpoints/orbit_cnn_path2d_1000.npz --out outputs/infer_cnn.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traj_reconstruction import load_phase1_sample, orbit_align, xy_from_state
from traj_reconstruction.flexible import fit_flexible_from_audio
from traj_reconstruction.orbit_cnn import infer_learned_orbit, load_orbit_model
from traj_reconstruction.parametric import plot_fit_overlay
from traj_reconstruction.parametric import FitResult


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1", type=Path, default=None)
    p.add_argument("--wav", type=Path, default=None)
    p.add_argument("--mode", choices=("flexible", "seq", "cnn", "mlp"), default="cnn")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Best-weights file (reloaded from disk each run; safe during training)",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/infer_orbit.json"))
    p.add_argument("--plot", type=Path, default=Path("outputs/infer_orbit.png"))
    args = p.parse_args()

    gt_xy = None
    stft = None
    wav = args.wav
    n_frames = None

    if args.phase1 is not None:
        sample = load_phase1_sample(args.phase1)
        stft = sample.stft_db
        wav = sample.wav_path
        n_frames = sample.n_frames
        src = sample.canonical_state_frames
        if src is None:
            src = sample.state_frames
        gt_xy = xy_from_state(src)

    if args.mode == "flexible":
        if wav is not None:
            fit = fit_flexible_from_audio(wav_path=wav, gt_xy=gt_xy)
        elif stft is not None:
            fit = fit_flexible_from_audio(stft_db=stft, gt_xy=gt_xy)
        else:
            raise SystemExit("provide --phase1 or --wav")
        xy = fit.xy_pred
        payload = fit.to_jsonable()
        _write_plot(fit, args.plot, gt_xy=gt_xy, title="Flexible freeform orbit")
    else:
        from traj_reconstruction.paths import (
            DEFAULT_ORBIT_CNN_BEST,
            DEFAULT_ORBIT_MLP_BEST,
            DEFAULT_ORBIT_SEQ_BEST,
        )

        if args.checkpoint is not None:
            ckpt = args.checkpoint
        elif args.mode == "mlp":
            ckpt = DEFAULT_ORBIT_MLP_BEST
        elif args.mode == "cnn":
            ckpt = DEFAULT_ORBIT_CNN_BEST
        else:
            ckpt = DEFAULT_ORBIT_SEQ_BEST
        model = load_orbit_model(ckpt)
        if wav is not None:
            xy = infer_learned_orbit(model, wav_path=wav, stft_db=stft)
        elif stft is not None:
            xy = infer_learned_orbit(model, stft_db=stft)
        else:
            raise SystemExit("provide --phase1 or --wav")
        payload = {
            "mode": args.mode,
            "checkpoint": str(ckpt),
            "n_frames": int(xy.shape[0]),
            "note": "Output is an observer-centered trajectory orbit; heading is free.",
        }
        if gt_xy is not None:
            n = min(len(xy), len(gt_xy))
            orb = orbit_align(xy[:n], gt_xy[:n])
            payload["orbit_rms"] = orb.rms
        dummy = FitResult(
            family=f"orbit_{args.mode}",
            params={},
            xy_pred=xy,
            frame_times=np.arange(len(xy), dtype=np.float64),
            f_hat_hz=np.zeros(len(xy)),
            A_hat=np.zeros(len(xy)),
            residual_rms_f=0.0,
            residual_rms_A=0.0,
            success=True,
            message=args.mode,
        )
        _write_plot(dummy, args.plot, gt_xy=gt_xy, title=f"{args.mode.upper()} orbit inference")

    payload["xy_pred"] = np.asarray(xy).tolist()
    payload["mirror_ambiguous"] = True
    payload["heading_absolute"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in payload if k != "xy_pred"}, indent=2))
    print(f"wrote {args.out}")


def _write_plot(fit: FitResult, out_path: Path, *, gt_xy, title: str) -> None:
    try:
        plot_fit_overlay(fit, out_path, gt_xy=gt_xy, title=title)
    except ImportError:
        print("skipping plot (install matplotlib); JSON was still written")
        return
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
