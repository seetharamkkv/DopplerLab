#!/usr/bin/env python3
"""After deep/shallow cpx left+right finish, run fusion evals (idempotent)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CKPT = ROOT / "checkpoints"
OUT = ROOT / "outputs"
VENV_PYTHON = Path(r"D:\Antigravity\venv\Scripts\python.exe")
PYTHON = str(VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable))
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
LOG = OUT / "cpx_fusion_after.log"

FUSIONS: list[tuple[str, Path, Path, str]] = [
    (
        "fusion_cpx_2class_100ep",
        CKPT / "transfer/direction/deep_cpx_2class_left_100ep/best.pt",
        CKPT / "transfer/direction/deep_cpx_2class_right_100ep/best.pt",
        "deep cpx L+R",
    ),
    (
        "fusion_cpx_baseline_2class",
        CKPT / "cnn/direction/cpx_3class_left/best.pt",
        CKPT / "cnn/direction/cpx_3class_right/best.pt",
        "shallow cpx L+R",
    ),
]


def log(msg: str) -> None:
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def wait_for(path: Path, label: str, timeout_hours: float = 48.0) -> None:
    start = time.time()
    log(f"waiting for {label} -> {path}")
    while not path.exists():
        if time.time() - start > timeout_hours * 3600:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(120)


def run_fusion(run_name: str, left: Path, right: Path, label: str) -> None:
    out_eval = OUT / "fusion" / "direction" / f"{run_name}_eval" / "eval_metrics.json"
    if out_eval.exists():
        log(f"SKIP {run_name} (eval exists)")
        return
    wait_for(left, f"{label} left best.pt")
    wait_for(right, f"{label} right best.pt")
    cmd = [
        PYTHON, "-m", "idmt_experiments.fusion",
        "--left-checkpoint", str(left),
        "--right-checkpoint", str(right),
        "--run-name", run_name,
    ]
    log(">>> " + " ".join(cmd))
    with LOG.open("a", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=SRC, check=True, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    log(f"DONE {run_name}")


def main() -> None:
    log("=" * 72)
    log("Cpx fusion watcher started")
    log(f"python: {PYTHON}")
    for run_name, left, right, label in FUSIONS:
        run_fusion(run_name, left, right, label)
    log("All cpx fusion steps complete.")


if __name__ == "__main__":
    main()
