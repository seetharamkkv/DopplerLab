#!/usr/bin/env python3
"""Train STFT→orbit MLP on a simulated Tier-1 freehand batch (Phase 3b).

Example:
  python scripts/build_tier1_batch.py --out data/tier1_train \\
      --n-straight 40 --n-arc 40 --n-s-curve 40 --n-u-turn 20 --n-multi-cpa 20
  python scripts/train_orbit_mlp.py --batch data/tier1_train \\
      --epochs 60 --checkpoint checkpoints/orbit_mlp.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traj_reconstruction.flexible import train_orbit_mlp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/orbit_mlp.npz"))
    p.add_argument(
        "--holdout-families",
        nargs="*",
        default=["u_turn"],
        help="Families held out for validation (default: u_turn)",
    )
    args = p.parse_args()

    print("Training OrbitMLP on SIMULATED batch only (STFT → canonical path)…")
    _, report = train_orbit_mlp(
        args.batch,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        holdout_families=tuple(args.holdout_families),
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps({k: report[k] for k in report if k != "history"}, indent=2))
    print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
