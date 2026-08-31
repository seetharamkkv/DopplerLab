#!/usr/bin/env python3
"""Train an orbit model on a simulated Phase 1 batch (audio/STFT only).

Default architecture is the complex-STFT 2D CNN. Pass ``--arch mlp`` only to
retrain the legacy flattened log-magnitude MLP.

Examples:
  python scripts/train_orbit_mlp.py --batch /path/to/traj_reconstruction_1000
  python scripts/train_orbit_mlp.py --arch mlp --batch data/tier1_train --epochs 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traj_reconstruction.flexible import train_orbit_mlp
from traj_reconstruction.orbit_cnn import train_orbit_cnn
from traj_reconstruction.paths import DEFAULT_ORBIT_CNN_BEST, DEFAULT_ORBIT_MLP_BEST


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=("cnn", "mlp"), default="cnn")
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--min-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from last.npz / existing best (CNN only if complex-STFT)",
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
    holdout = tuple(args.holdout_families)

    if args.arch == "cnn":
        ckpt = args.checkpoint or DEFAULT_ORBIT_CNN_BEST
        lr = 3e-4 if args.lr is None else args.lr
        print("Training OrbitCNN on SIMULATED batch (complex STFT → canonical path)…")
        print(f"best checkpoint (frontend): {ckpt}")
        print(f"resume={args.resume}  min_epochs={args.min_epochs}  patience={args.patience}")
        _, report = train_orbit_cnn(
            args.batch,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            lr=lr,
            seed=args.seed,
            holdout_families=holdout,
            checkpoint_path=ckpt,
            resume=args.resume,
        )
    else:
        ckpt = args.checkpoint or DEFAULT_ORBIT_MLP_BEST
        lr = 3e-3 if args.lr is None else args.lr
        print("Training OrbitMLP on SIMULATED batch only (STFT → canonical path)…")
        print(f"best checkpoint (frontend): {ckpt}")
        print(f"resume={args.resume}")
        _, report = train_orbit_mlp(
            args.batch,
            epochs=args.epochs,
            lr=lr,
            seed=args.seed,
            holdout_families=holdout,
            checkpoint_path=ckpt,
            resume=args.resume,
        )
    print(json.dumps({k: report[k] for k in report if k != "history"}, indent=2))
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
