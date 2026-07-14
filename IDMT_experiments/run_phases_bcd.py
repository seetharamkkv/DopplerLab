#!/usr/bin/env python3
"""Run Phases B and C sequentially (100 epochs each, 2-class direction)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
SRC = Path(__file__).resolve().parent / "src"
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=SRC, check=True, env=ENV)


def main() -> None:
    print("=" * 72)
    print("Phases B -> C (100 epochs, 2-class L2R/R2L, CPU)")
    print("=" * 72)

    # Phase B — deep mel CNN left + right (for fusion)
    run([
        PYTHON, "-m", "idmt_experiments.transfer",
        "--run-name", "deep_mel_2class_left_100ep",
        "--mono-source", "left", "--epochs", "100",
    ])
    run([
        PYTHON, "-m", "idmt_experiments.transfer",
        "--run-name", "deep_mel_2class_right_100ep",
        "--mono-source", "right", "--epochs", "100",
    ])

    # Phase C — late fusion (no extra training)
    ckpt = Path(__file__).resolve().parent / "checkpoints"
    run([
        PYTHON, "-m", "idmt_experiments.fusion",
        "--left-checkpoint", str(ckpt / "transfer/direction/deep_mel_2class_left_100ep/best.pt"),
        "--right-checkpoint", str(ckpt / "transfer/direction/deep_mel_2class_right_100ep/best.pt"),
        "--run-name", "fusion_2class_100ep",
    ])

    # Comparison table
    run([PYTHON, "-m", "idmt_experiments.scripts.compare_phases_bcd", "--refresh"])
    print("\nAll phases complete.")


if __name__ == "__main__":
    main()
