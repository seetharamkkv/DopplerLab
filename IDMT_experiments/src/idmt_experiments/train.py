#!/usr/bin/env python3
"""Train IDMT classifier (direction or vehicle type) with optional auto-eval."""

from __future__ import annotations

import argparse
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, DirectionConfig
from idmt_experiments.src.direction.eval import run_eval, run_eval_location_loo
from idmt_experiments.src.direction.train import train_location_loo, train_split
from idmt_experiments.src.splits import build_location_loo_splits


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train IDMT CNN + auto eval")
    p.add_argument("--data-dir", type=Path, default=None, help="IDMT_Traffic root")
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--task", choices=["direction", "vehicle"], default="direction")
    p.add_argument(
        "--mode",
        choices=["eusipco", "location_loo", "weather_holdout"],
        default="eusipco",
        help="eusipco = official paper split; location_loo = leave-one-site-out; weather_holdout = train dry test wet",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--feature-type", choices=["mel", "cc", "stereo_mel"], default="mel")
    p.add_argument("--n-classes", type=int, default=None, help="direction: 2|3; vehicle: 5 (default)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--preempt", action="store_true", help="Early-stop when val loss plateaus")
    p.add_argument(
        "--norm-max-samples",
        type=int,
        default=512,
        help="Train clips used to fit per-bin normalization (leak-safe subsample; use 0 for all)",
    )
    p.add_argument("--mic", default="SE", help="Microphone filter (SE or ME)")
    p.add_argument("--channel", default="CH34", help="Channel pair filter")
    p.add_argument("--force-retrain", action="store_true", help="LOO: retrain all folds")
    p.add_argument(
        "--retrain-folds",
        nargs="+",
        default=None,
        metavar="FOLD",
        help="LOO: retrain named folds only",
    )
    p.add_argument("--no-resume", action="store_true", help="LOO: do not skip completed folds")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-split", choices=["test", "valid"], default="test")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--no-swap-test", action="store_true", help="Skip channel-swap check in eval")
    return p


def _default_n_classes(task: str) -> int:
    return 5 if task == "vehicle" else 3


def main() -> None:
    args = build_parser().parse_args()
    n_classes = args.n_classes if args.n_classes is not None else _default_n_classes(args.task)
    split_name = "weather_holdout" if args.mode == "weather_holdout" else "eusipco"

    cfg = DirectionConfig(
        task=args.task,
        feature_type=args.feature_type,
        n_classes=n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        preempt=args.preempt,
        mic_filter=args.mic,
        channel_filter=args.channel,
        split_name=split_name,
        norm_fit_max_samples=None if args.norm_max_samples == 0 else args.norm_max_samples,
    )

    task_title = "DIRECTION" if args.task == "direction" else "VEHICLE TYPE"
    print("=" * 72)
    print(f"IDMT {task_title} — TRAINING")
    print("=" * 72)
    print(f"  task        : {args.task}")
    print(f"  mode        : {args.mode}")
    print(f"  feature     : {cfg.feature_type}")
    print(f"  classes     : {cfg.n_classes}")
    print(f"  mic/channel : {cfg.mic_filter} / {cfg.channel_filter}")
    print(f"  preempt     : {cfg.preempt}")
    if args.mode == "location_loo":
        print(f"  resume      : {not args.no_resume}")
    if cfg.feature_type == "cc":
        print("  note        : CC precomputes features before epoch 1 (~20–40 min CPU on full train)")
    print()

    if args.mode == "location_loo":
        run_dir = train_location_loo(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            run_name=args.run_name,
            cfg=cfg,
            device=args.device,
            resume=not args.no_resume,
            force_retrain=args.force_retrain,
            retrain_folds=args.retrain_folds,
        )
        n_folds = len(list(run_dir.glob("fold_*.pt")))
        n_expected = len(
            build_location_loo_splits(
                args.data_dir,
                mic_filter=cfg.mic_filter,
                channel_filter=cfg.channel_filter,
            )
        )
        if not args.skip_eval and n_folds >= n_expected:
            print("\n" + "=" * 72)
            print("AUTO EVAL (location LOO pooled)")
            print("=" * 72)
            run_eval_location_loo(
                run_dir=run_dir,
                data_dir=args.data_dir,
                device=args.device,
                run_swap_test=not args.no_swap_test,
            )
        elif not args.skip_eval:
            print(f"\nLOO eval skipped — only {n_folds}/{n_expected} fold checkpoints")
    else:
        ckpt = train_split(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            run_name=args.run_name,
            cfg=cfg,
            device=args.device,
            split_name=cfg.split_name,
        )
        if not args.skip_eval:
            print("\n" + "=" * 72)
            print(f"AUTO EVAL ({args.eval_split})")
            print("=" * 72)
            run_eval(
                checkpoint=ckpt,
                data_dir=args.data_dir,
                device=args.device,
                split=args.eval_split,
                run_swap_test=not args.no_swap_test,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
