#!/usr/bin/env python3
"""Train hybrid PINN-style direction model (CNN mel + physics conditioning)."""

from __future__ import annotations

import argparse
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, HybridConfig
from idmt_experiments.hybrid.eval import run_eval
from idmt_experiments.hybrid.train import train_split


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train hybrid direction model (CNN mel + kinematic_v3 conditioning)"
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--run-name", type=str, default="hybrid_mel_left_v3_ep60")
    p.add_argument("--mono-source", choices=["left", "right"], default="left")
    p.add_argument("--feature-set", default="kinematic_v3", choices=["kinematic_v3", "kinematic_full"])
    p.add_argument("--n-classes", type=int, default=3, choices=[2, 3])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--preempt", action="store_true", help="Enable early stopping (default: run all epochs)")
    p.add_argument(
        "--resume-training",
        action="store_true",
        help="Continue from last.pt if present; set --epochs higher to train further",
    )
    p.add_argument("--physics-embed-dim", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--split", default="eusipco")
    p.add_argument("--eval-split", choices=["test", "valid"], default="test")
    p.add_argument("--skip-eval", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = HybridConfig(
        mono_source=args.mono_source,
        n_classes=args.n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        preempt=args.preempt,
        feature_set=args.feature_set,
        physics_embed_dim=args.physics_embed_dim,
        split_name=args.split,
        include_no_vehicle=args.n_classes == 3,
    )

    print("=" * 72)
    print("IDMT DIRECTION — HYBRID TRAINING (CNN mel + physics conditioning)")
    print("=" * 72)
    print(f"  run_name        : {args.run_name}")
    print(f"  mono_source     : {cfg.mono_source}")
    print(f"  feature_set     : {cfg.feature_set}")
    print(f"  n_classes       : {cfg.n_classes}")
    print(f"  epochs          : {cfg.epochs}")
    print(f"  preempt         : {cfg.preempt}")
    print(f"  physics_embed   : {cfg.physics_embed_dim}")
    print(f"  resume          : {args.resume_training}")
    print()

    run_dir = train_split(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        cfg=cfg,
        device=args.device,
        resume=args.resume_training,
    )

    if not args.skip_eval:
        print("\n" + "=" * 72)
        print(f"AUTO EVAL ({args.eval_split})")
        print("=" * 72)
        run_eval(run_dir=run_dir, data_dir=args.data_dir, split=args.eval_split, device=args.device)

    print("\nDone.")


if __name__ == "__main__":
    main()
