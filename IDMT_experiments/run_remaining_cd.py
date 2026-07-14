#!/usr/bin/env python3
"""Wait for the resumed Phase B right run to finish, then run Phase C and the table.

The Phase B right model (deep_mel_2class_right_100ep) is resumed in a separate process.
This runner blocks until that run writes its train_summary.json, then chains late fusion (C)
and refreshes the comparison table.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}

RIGHT_SUMMARY = ROOT / "checkpoints/transfer/direction/deep_mel_2class_right_100ep/train_summary.json"
CKPT = ROOT / "checkpoints"


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=SRC, check=True, env=ENV)


def wait_for_right(timeout_hours: float = 12.0) -> None:
    start = time.time()
    # Require the summary to be fresh (written after this runner started).
    launched_at = start
    print(f"Waiting for Phase B right to finish -> {RIGHT_SUMMARY}", flush=True)
    while True:
        if RIGHT_SUMMARY.exists() and RIGHT_SUMMARY.stat().st_mtime >= launched_at:
            print("Phase B right complete.", flush=True)
            return
        if time.time() - start > timeout_hours * 3600:
            raise TimeoutError("Timed out waiting for Phase B right to finish.")
        time.sleep(60)


def main() -> None:
    wait_for_right()

    # Phase C - late fusion (no extra training)
    run([
        PYTHON, "-m", "idmt_experiments.fusion",
        "--left-checkpoint", str(CKPT / "transfer/direction/deep_mel_2class_left_100ep/best.pt"),
        "--right-checkpoint", str(CKPT / "transfer/direction/deep_mel_2class_right_100ep/best.pt"),
        "--run-name", "fusion_2class_100ep",
    ])

    # Comparison table
    run([PYTHON, "-m", "idmt_experiments.scripts.compare_phases_bcd", "--refresh"])
    print("\nRemaining phases (C, table) complete.", flush=True)


if __name__ == "__main__":
    main()
