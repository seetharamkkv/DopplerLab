#!/usr/bin/env python3
"""Train IDMT classifier (direction or vehicle type) with optional auto-eval.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
CLI defaults (--feature-type mel, --mode eusipco, DirectionConfig fields) define the
reference runs. Do not change default behaviour without re-benchmarking all three runs.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, DirectionConfig
from idmt_experiments.cnn.eval import run_eval, run_eval_location_loo
from idmt_experiments.cnn.train import train_location_loo, train_split
from idmt_experiments.cnn.train_recipe import TrainRecipe, phase_a_recipe
from idmt_experiments.src.splits import build_location_loo_splits


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train IDMT CNN + auto eval")
    p.add_argument("--data-dir", type=Path, default=None, help="IDMT_Traffic root")
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--task", choices=["direction", "vehicle", "weather"], default="direction")
    p.add_argument(
        "--mode",
        choices=["eusipco", "location_loo", "weather_holdout", "weather_site", "weather_pooled"],
        default="eusipco",
        help=(
            "eusipco = official paper split (direction/vehicle); weather uses weather_site; "
            "weather_pooled = all sites (location-confounded ablation only)"
        ),
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--feature-type", choices=["mel", "cc", "stereo_mel", "complex_stft"], default="mel")
    p.add_argument(
        "--mono-source",
        choices=["mean", "left", "right"],
        default="mean",
        help="Mel only: mean=(L+R)/2, left/right=single channel from stereo pair (no downmix)",
    )
    p.add_argument("--n-classes", type=int, default=None, help="direction: 2|3; vehicle: 5 (default)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--preempt", action="store_true", help="Early-stop when val loss plateaus")
    p.add_argument(
        "--phase-a",
        action="store_true",
        help="Phase A recipe: SpecAugment + balanced sampler + focal loss + label smooth + grad clip + cosine LR",
    )
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
    p.add_argument(
        "--resume-training",
        action="store_true",
        help="eusipco: continue from last.pt if present; set --epochs higher to train further",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-split", choices=["test", "valid"], default="test")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--interventions", action="store_true", help="Run intervention battery after eval")
    p.add_argument("--no-swap-test", action="store_true", help="Skip channel-swap check in eval")
    return p


def _default_n_classes(task: str) -> int:
    if task == "vehicle":
        return 5
    if task == "weather":
        return 2
    return 3


def main() -> None:
    args = build_parser().parse_args()
    if args.task == "weather" and args.mode == "weather_holdout":
        raise SystemExit(
            "Weather classification cannot use weather_holdout (direction generalization only). "
            "Use default --mode eusipco (maps to weather_site) or --mode weather_pooled for ablation."
        )
    if args.mono_source != "mean" and args.feature_type not in ("mel", "complex_stft"):
        raise SystemExit("--mono-source left/right requires --feature-type mel or complex_stft")
    n_classes = args.n_classes if args.n_classes is not None else _default_n_classes(args.task)
    if args.task == "weather":
        split_name = "weather_stratified" if args.mode == "weather_pooled" else "weather_site"
    else:
        split_name = "weather_holdout" if args.mode == "weather_holdout" else "eusipco"

    train_recipe = phase_a_recipe() if args.phase_a else TrainRecipe()
    weight_decay = args.weight_decay
    if args.phase_a and args.weight_decay == 1e-4:
        weight_decay = 1e-3

    cfg = DirectionConfig(
        task=args.task,
        feature_type=args.feature_type,
        mono_source=args.mono_source,
        n_classes=n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=weight_decay,
        patience=args.patience,
        preempt=args.preempt,
        mic_filter=args.mic,
        channel_filter=args.channel,
        split_name=split_name,
        norm_fit_max_samples=None if args.norm_max_samples == 0 else args.norm_max_samples,
    )

    task_title = {
        "direction": "DIRECTION",
        "vehicle": "VEHICLE TYPE",
        "weather": "ROAD WEATHER (DRY/WET)",
    }[args.task]
    print("=" * 72)
    print(f"IDMT {task_title} — TRAINING")
    print("=" * 72)
    print(f"  task        : {args.task}")
    print(f"  mode        : {args.mode}")
    print(f"  feature     : {cfg.feature_type}")
    if cfg.feature_type in ("mel", "complex_stft"):
        print(f"  mono source : {cfg.mono_source}")
    print(f"  classes     : {cfg.n_classes}")
    print(f"  split       : {cfg.split_name}")
    print(f"  mic/channel : {cfg.mic_filter} / {cfg.channel_filter}")
    print(f"  preempt     : {cfg.preempt}")
    print(f"  epochs      : {cfg.epochs}")
    if args.mode != "location_loo":
        print(f"  resume      : {args.resume_training}")
    if args.task == "weather" and cfg.split_name == "weather_site":
        print("  note        : site-controlled split (Schleusinger-Allee only — no location confound)")
    elif args.task == "weather" and cfg.split_name == "weather_stratified":
        print("  note        : POOLED split — location confounds dry/wet; ablation only")
    if args.mode == "location_loo":
        print(f"  resume      : {not args.no_resume}")
    if args.phase_a:
        print(f"  phase-a      : on (SpecAugment, balanced sampler, focal, cosine LR)")
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
            resume=args.resume_training,
            train_recipe=train_recipe,
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
                run_intervention_tests=args.interventions,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
