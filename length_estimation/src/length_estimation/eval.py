#!/usr/bin/env python3
"""
Evaluate Phase B length CNN (split or LOVO).

See length_estimation/README.md for LOVO protocol, metrics, and report locations.

Examples
--------
# LOVO pooled eval — each fold's held-out car never appeared in that fold's training
python -m length_estimation.eval --mode lovo --run-name lovo_mel_v1

# Train/valid split eval (single best.pt; all vehicles may appear in training)
python -m length_estimation.eval --mode split --run-name mel_length_20260616_004732

# Explicit paths
python -m length_estimation.eval --mode lovo --run-dir length_estimation/checkpoints/length_cnn/lovo_mel_v1
python -m length_estimation.eval --mode split --checkpoint length_estimation/checkpoints/length_cnn/mel_length_20260616_004732/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from length_estimation.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR
from length_estimation.src.phase_b.eval import run_eval


def _resolve_lovo_run_dir(
    run_dir: Path | None,
    run_name: str | None,
    checkpoint_dir: Path,
) -> Path:
    if run_dir is not None:
        path = Path(run_dir)
    elif run_name is not None:
        path = Path(checkpoint_dir) / "length_cnn" / run_name
    else:
        print("LOVO eval requires --run-dir or --run-name")
        sys.exit(2)
    if not path.is_dir():
        print(f"LOVO run directory not found: {path}")
        sys.exit(1)
    n_folds = len(list(path.glob("fold_*.pt")))
    if n_folds == 0:
        print(f"No fold_*.pt checkpoints in: {path}")
        sys.exit(1)
    if n_folds < 13:
        print(f"WARN: only {n_folds}/13 fold checkpoints — eval will use available folds only")
    return path


def _resolve_split_checkpoint(
    checkpoint: Path | None,
    run_name: str | None,
    checkpoint_dir: Path,
) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
    elif run_name is not None:
        path = Path(checkpoint_dir) / "length_cnn" / run_name / "best.pt"
    else:
        print("Split eval requires --checkpoint or --run-name")
        sys.exit(2)
    if not path.is_file():
        print(f"Checkpoint not found: {path}")
        sys.exit(1)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate VS13 length CNN (Phase B) — split or LOVO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["split", "lovo"],
        default="lovo",
        help="split = single best.pt on train/valid; lovo = pooled 13-fold unseen-vehicle eval",
    )
    p.add_argument("--data-dir", type=Path, default=None, help="VS13 root (default: data/vs13)")
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--output-dir", type=Path, default=None, help=f"default: {DEFAULT_OUTPUT_DIR}/phase_b/<run>")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")

    # Split mode
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to best.pt (split mode)")
    p.add_argument(
        "--split",
        choices=["valid", "train"],
        default="valid",
        help="Clip split to evaluate (split mode only)",
    )

    # LOVO mode
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory with fold_*.pt checkpoints (lovo mode)",
    )

    # Either mode
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Checkpoint folder name under checkpoints/length_cnn/",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 72)
    print("VS13 VEHICLE LENGTH — PHASE B EVALUATION")
    print("=" * 72)
    print(f"  mode       : {args.mode}")
    if args.mode == "lovo":
        run_dir = _resolve_lovo_run_dir(args.run_dir, args.run_name, args.checkpoint_dir)
        print(f"  run_dir    : {run_dir}")
        print(f"  protocol   : each clip predicted by the fold where its vehicle was held out")
        print()
        run_eval(
            run_dir=run_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        ckpt = _resolve_split_checkpoint(args.checkpoint, args.run_name, args.checkpoint_dir)
        print(f"  checkpoint : {ckpt}")
        print(f"  split      : {args.split}")
        print()
        run_eval(
            checkpoint=ckpt,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            split=args.split,
        )

    out_name = args.run_name or (args.run_dir.name if args.run_dir else None)
    if out_name is None and args.mode == "split":
        out_name = args.checkpoint.parent.name if args.checkpoint else None
    if out_name:
        report = (args.output_dir or DEFAULT_OUTPUT_DIR) / "phase_b" / out_name / "eval_summary.txt"
        if report.exists():
            print(f"\nReport -> {report}")

    print("\nDone.")


if __name__ == "__main__":
    main()
