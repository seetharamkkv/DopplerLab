#!/usr/bin/env python3
"""
Train the VS13 vehicle-length CNN (Phase B).

See length_estimation/README.md for dataset layout, LOVO vs split modes, and reports.

Training automatically runs evaluation when finished.

Examples
--------
# Recommended: VS13 train/valid split (~320 train / ~80 valid)
python -m length_estimation.train

# Full 13-fold LOVO (slow — one model per held-out vehicle)
python -m length_estimation.train --mode lovo

# Custom run name and SSQ spectrograms
python -m length_estimation.train --run-name my_ssq_run --spec-type ssq --epochs 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

from length_estimation.config import DEFAULT_CHECKPOINT_DIR, PhaseBConfig
from length_estimation.src.phase_b.eval import run_eval
from length_estimation.src.phase_b.train import train_lovo, train_split


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train VS13 length CNN (Phase B) + auto eval")
    p.add_argument("--data-dir", type=Path, default=None, help="VS13 root (default: data/vs13)")
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument(
        "--mode",
        choices=["split", "lovo"],
        default="split",
        help="split = VS13 train/valid; lovo = leave-one-vehicle-out",
    )
    p.add_argument("--run-name", type=str, default=None, help="Checkpoint folder name")
    p.add_argument("--spec-type", choices=["mel", "ssq"], default="mel")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument(
        "--preempt",
        action="store_true",
        help="Early-stop when val MAE plateaus (default: run all epochs, still save best.pt)",
    )
    p.add_argument(
        "--force-retrain",
        action="store_true",
        help="LOVO: retrain all folds even if fold_*.pt already exists",
    )
    p.add_argument(
        "--retrain-folds",
        nargs="+",
        default=None,
        metavar="VEHICLE",
        help="LOVO: retrain only these held-out vehicles (e.g. KiaSportage after interrupt)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="LOVO: do not skip existing fold checkpoints (overwrites unless you delete files)",
    )
    p.add_argument("--no-speed", action="store_true", help="Disable speed auxiliary input")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument(
        "--eval-split",
        choices=["valid", "train"],
        default="valid",
        help="Split used for post-train eval (split mode only)",
    )
    p.add_argument("--skip-eval", action="store_true", help="Train only, do not run eval")
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = PhaseBConfig(
        spec_type=args.spec_type,
        target="length_m",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        preempt=args.preempt,
        include_speed=not args.no_speed,
    )

    print("=" * 72)
    print("VS13 VEHICLE LENGTH — PHASE B TRAINING")
    print("=" * 72)
    print(f"  mode       : {args.mode}")
    print(f"  target     : length_m only (wheelbase excluded)")
    print(f"  spec       : {cfg.spec_type}")
    print(f"  speed aux  : {cfg.include_speed}")
    print(f"  epochs     : {cfg.epochs}")
    print(f"  preempt    : {cfg.preempt}  ({'early stop on' if cfg.preempt else 'full schedule'})")
    if args.mode == "lovo":
        print(f"  resume     : {not args.no_resume}  (skip completed fold_*.pt)")
    print()

    if args.mode == "split":
        ckpt = train_split(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            run_name=args.run_name,
            cfg=cfg,
            device=args.device,
        )
        if not args.skip_eval:
            print("\n" + "=" * 72)
            print("AUTO EVAL (valid split)")
            print("=" * 72)
            run_eval(checkpoint=ckpt, data_dir=args.data_dir, device=args.device, split=args.eval_split)
    else:
        run_dir = train_lovo(
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
        if not args.skip_eval and n_folds >= 13:
            print("\n" + "=" * 72)
            print("AUTO EVAL (LOVO pooled)")
            print("=" * 72)
            run_eval(run_dir=run_dir, data_dir=args.data_dir, device=args.device)
        elif not args.skip_eval:
            print(f"\nLOVO eval skipped — only {n_folds}/13 fold checkpoints (re-run to finish, then eval)")

    print("\nDone.")


if __name__ == "__main__":
    main()
