#!/usr/bin/env python3
"""Phase C: late fusion of left + right mono 2-class models."""

from __future__ import annotations

import argparse
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR
from idmt_experiments.fusion.eval import run_fusion_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Phase C fusion eval")
    p.add_argument("--left-checkpoint", type=Path, required=True)
    p.add_argument("--right-checkpoint", type=Path, required=True)
    p.add_argument("--run-name", default="fusion_2class_100ep")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    print("=" * 72)
    print("PHASE C - Late fusion (left + right mono, 2-class)")
    print("=" * 72)
    run_fusion_eval(
        args.left_checkpoint,
        args.right_checkpoint,
        run_name=args.run_name,
        device=args.device,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
