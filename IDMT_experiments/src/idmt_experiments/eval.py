#!/usr/bin/env python3
"""Evaluate IDMT classifier checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, DirectionConfig, checkpoint_subdir
from idmt_experiments.src.direction.eval import run_eval, run_eval_location_loo
from idmt_experiments.src.direction.train import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate IDMT CNN")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--task", choices=["direction", "vehicle"], default=None, help="Default: read from checkpoint")
    p.add_argument("--mode", choices=["eusipco", "location_loo", "weather_holdout"], default="eusipco")
    p.add_argument("--checkpoint", type=Path, default=None, help="best.pt for single-checkpoint modes")
    p.add_argument("--run-dir", type=Path, default=None, help="Run dir with fold_*.pt for LOO")
    p.add_argument("--run-name", type=str, default=None, help="Resolve under checkpoints/<task>/")
    p.add_argument("--split", choices=["test", "valid", "train"], default="test")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-swap-test", action="store_true", help="Skip channel-swap causality check")
    return p


def _resolve_checkpoint(args, task: str) -> Path | None:
    if args.checkpoint:
        return args.checkpoint
    if args.run_dir:
        return None
    if args.run_name:
        cfg = DirectionConfig(task=task)
        return DEFAULT_CHECKPOINT_DIR / checkpoint_subdir(cfg) / args.run_name / "best.pt"
    return None


def main() -> None:
    args = build_parser().parse_args()
    task = args.task or "direction"
    run_dir = args.run_dir
    if args.run_name and not run_dir and args.mode == "location_loo":
        cfg = DirectionConfig(task=task)
        run_dir = DEFAULT_CHECKPOINT_DIR / checkpoint_subdir(cfg) / args.run_name

    if args.mode == "location_loo":
        if not run_dir:
            raise SystemExit("location_loo eval requires --run-dir or --run-name")
        run_eval_location_loo(
            run_dir=run_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            run_swap_test=not args.no_swap_test,
        )
    else:
        ckpt = _resolve_checkpoint(args, task)
        if not ckpt:
            raise SystemExit("eval requires --checkpoint or --run-name")
        if args.task is None:
            _, cfg, _, _ = load_checkpoint(ckpt, "cpu")
            task = cfg.task
        run_eval(
            checkpoint=ckpt,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            split=args.split,
            run_swap_test=not args.no_swap_test,
        )


if __name__ == "__main__":
    main()
