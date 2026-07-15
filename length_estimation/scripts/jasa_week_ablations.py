#!/usr/bin/env python3
"""Thin launcher for JASA week ablations (install package first: pip install -e length_estimation/src)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as scripts/jasa_week_ablations.py without editable install
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from length_estimation.jasa_week import run_ablation1, run_full_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="JASA week length ablations (LOVO on features.csv)")
    p.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Path to features.csv (default: outputs/features.csv)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: outputs/jasa_week/)",
    )
    p.add_argument(
        "--ablation1-only",
        action="store_true",
        help="Run only Ablation 1 (L vs Wb target decision)",
    )
    args = p.parse_args()

    if args.ablation1_only:
        run_ablation1(features_path=args.features, output_dir=args.output_dir)
    else:
        run_full_pipeline(features_path=args.features, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
