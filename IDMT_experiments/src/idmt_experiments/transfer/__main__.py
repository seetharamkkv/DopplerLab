#!/usr/bin/env python3
"""Phase B: deep mel CNN (PANNs-inspired) direction training — 2-class L2R/R2L."""

from __future__ import annotations

import argparse
from pathlib import Path

from idmt_experiments.config import DirectionConfig
from idmt_experiments.cnn.train_recipe import phase_a_recipe
from idmt_experiments.engine.train_mel import train_mel_split
from idmt_experiments.transfer.eval import run_eval
from idmt_experiments.transfer.model import build_model


def main() -> None:
    p = argparse.ArgumentParser(description="Phase B transfer training (deep mel CNN)")
    p.add_argument("--run-name", required=True)
    p.add_argument("--mono-source", choices=["mean", "left", "right"], default="left")
    p.add_argument("--feature-type", choices=["mel", "complex_stft"], default="mel")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--preempt", action="store_true", help="Early-stop on val-loss plateau")
    p.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="With --preempt: never early-stop before this epoch (e.g. 20)",
    )
    p.add_argument("--patience", type=int, default=15, help="Early-stop patience (epochs without val improvement)")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from last.pt if present")
    args = p.parse_args()

    cfg = DirectionConfig(
        task="direction",
        feature_type=args.feature_type,
        mono_source=args.mono_source,
        n_classes=2,
        epochs=args.epochs,
        preempt=args.preempt,
        min_epochs=args.min_epochs,
        weight_decay=1e-3,
        patience=args.patience,
    )
    recipe = phase_a_recipe()

    print("=" * 72)
    print("PHASE B - Deep Mel CNN (2-class L2R/R2L)")
    print(f"  feature     : {args.feature_type}")
    print("=" * 72)

    ckpt = train_mel_split(
        run_name=args.run_name,
        cfg=cfg,
        model_factory=lambda c: build_model(c, "deep_mel_cnn"),
        checkpoint_subdir="transfer/direction",
        device=args.device,
        train_recipe=recipe,
        resume=args.resume,
        extra_ckpt={"backbone": "deep_mel_cnn", "phase": "B"},
    )

    if not args.skip_eval:
        print("\nAUTO EVAL (test)")
        run_eval(ckpt, output_subdir="transfer/direction", device=args.device)
    print("\nDone.")


if __name__ == "__main__":
    main()
