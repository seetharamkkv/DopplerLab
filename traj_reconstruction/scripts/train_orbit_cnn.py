#!/usr/bin/env python3
"""Train STFT→orbit 2D CNN on a simulated Phase 1 batch.

Inputs: WAV complex spectrogram (relative dB + phase). No simulator metadata.
Resume only from ``complex_stft_cnn_v1`` checkpoints — log-mag MLP files are ignored.

Examples:
  python scripts/train_orbit_cnn.py \\
      --batch /path/to/traj_reconstruction_1000 \\
      --epochs 250 --min-epochs 200 --patience 20 --resume \\
      --checkpoint checkpoints/orbit_cnn_path2d_1000.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traj_reconstruction.orbit_cnn import train_orbit_cnn
from traj_reconstruction.paths import DEFAULT_ORBIT_CNN_BEST


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--min-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_ORBIT_CNN_BEST,
        help="Best-weights file (frontend loads this; updated atomically)",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume only if last/best is a complex-STFT CNN (default: on)",
    )
    p.add_argument(
        "--holdout-families",
        nargs="*",
        default=["u_turn"],
        help=(
            "Families held out for validation (default: u_turn). "
            "If none match the batch (e.g. DopplerSim 2D whiteboard), a random 20 percent val split is used."
        ),
    )
    args = p.parse_args()

    print("Training OrbitCNN on SIMULATED batch (complex STFT → canonical path)…")
    print(f"best checkpoint (frontend): {args.checkpoint}")
    print(f"resume={args.resume}  min_epochs={args.min_epochs}  patience={args.patience}")
    _, report = train_orbit_cnn(
        args.batch,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        lr=args.lr,
        seed=args.seed,
        holdout_families=tuple(args.holdout_families),
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )
    skip = {"history"}
    print(json.dumps({k: report[k] for k in report if k not in skip}, indent=2))
    print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
