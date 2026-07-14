#!/usr/bin/env python3
"""CNN eval entry point with optional intervention battery.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Critical path for published monoaural metrics (vehicle bal. acc 81.5% / 79.3% / 73.6%).
Do not change default behaviour, numerics, or evaluation outputs without re-benchmarking
all three reference runs. Refactoring for maintainability is OK only if metrics stay
bit-identical. New work: separate --run-name or new modules.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR
from idmt_experiments.cnn.eval import run_eval


def _resolve_checkpoint(checkpoint_dir: Path, run_name: str) -> Path:
    for sub in (
        checkpoint_dir / "cnn" / "direction" / run_name / "best.pt",
        checkpoint_dir / "direction" / run_name / "best.pt",
    ):
        if sub.exists():
            return sub
    return checkpoint_dir / "cnn" / "direction" / run_name / "best.pt"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CNN direction eval / interventions")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--split", choices=["test", "valid"], default="test")
    p.add_argument("--device", default="auto")
    p.add_argument("--interventions", action="store_true", help="Time-reverse + channel-swap battery")
    p.add_argument("--no-swap-test", action="store_true", help="Skip legacy channel-swap pass in eval")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ckpt = _resolve_checkpoint(Path(args.checkpoint_dir), args.run_name)
    run_eval(
        checkpoint=ckpt,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        split=args.split,
        run_swap_test=not args.no_swap_test,
        run_intervention_tests=args.interventions,
    )


if __name__ == "__main__":
    main()
