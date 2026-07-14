#!/usr/bin/env python3
"""Train physics-informed IDMT direction classifier (L2R / R2L only)."""

from __future__ import annotations

import argparse
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, PhysicsConfig
from idmt_experiments.physics.eval import run_eval
from idmt_experiments.physics.train import train_split


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train physics direction classifier (L2R/R2L)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--run-name", type=str, default="physics_lr_2class_left_v2")
    p.add_argument("--mono-source", choices=["left", "right"], default="left")
    p.add_argument(
        "--classifier",
        choices=["logistic", "logistic_antisym", "gbt", "mlp"],
        default="logistic",
    )
    p.add_argument(
        "--feature-set",
        default="kinematic_v2",
        choices=["kinematic_v1", "kinematic_v2", "kinematic_v3", "kinematic_full"],
    )
    p.add_argument(
        "--include-no-vehicle",
        action="store_true",
        help="Include background clips (disabled by default — 2-class vehicle only)",
    )
    p.add_argument("--no-speed-scale", action="store_true", help="Omit speed-scaled feature terms")
    p.add_argument("--mic", default="SE")
    p.add_argument("--channel", default="CH34")
    p.add_argument("--split", default="eusipco")
    p.add_argument("--eval-split", choices=["test", "valid"], default="test")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--interventions", action="store_true", help="Run Tier-4 tests after eval")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = PhysicsConfig(
        mono_source=args.mono_source,
        include_no_vehicle=args.include_no_vehicle,
        classifier=args.classifier,
        feature_set=args.feature_set,
        use_speed_scaled_features=not args.no_speed_scale,
        mic_filter=args.mic,
        channel_filter=args.channel,
        split_name=args.split,
        n_classes=2,
    )

    print("=" * 72)
    print("IDMT DIRECTION — PHYSICS TRAINING (L2R / R2L only)")
    print("=" * 72)
    print(f"  run_name           : {args.run_name}")
    print(f"  mono_source        : {cfg.mono_source}")
    print(f"  include_no_vehicle : {cfg.include_no_vehicle}")
    print(f"  feature_set        : {cfg.feature_set}")
    print(f"  classifier         : {cfg.classifier}")
    print()

    run_dir = train_split(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        cfg=cfg,
    )

    if not args.skip_eval:
        print("\n" + "=" * 72)
        print(f"AUTO EVAL ({args.eval_split})")
        print("=" * 72)
        run_eval(
            run_dir=run_dir,
            data_dir=args.data_dir,
            split=args.eval_split,
            run_intervention_tests=args.interventions,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
