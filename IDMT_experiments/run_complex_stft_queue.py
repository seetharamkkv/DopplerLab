#!/usr/bin/env python3
"""Run complex-STFT benchmarks sequentially (train + fusion). Skips completed jobs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CKPT = Path(os.environ["IDMT_CHECKPOINT_DIR"]) if os.environ.get("IDMT_CHECKPOINT_DIR") else ROOT / "checkpoints"
OUT = Path(os.environ["IDMT_OUTPUT_DIR"]) if os.environ.get("IDMT_OUTPUT_DIR") else ROOT / "outputs"
VENV_PYTHON = Path(r"D:\Antigravity\venv\Scripts\python.exe")
PYTHON = str(VENV_PYTHON if VENV_PYTHON.is_file() else Path(sys.executable))
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
DEVICE = os.environ.get("IDMT_DEVICE", "cpu")

# Remaining deep runs: early-stop after at least 20 epochs (patience=15).
# complex_stft + deep CNN can OOM at batch 32 on small GPUs — override with IDMT_DEVICE=cuda.
DEEP_EARLY = [
    "--preempt", "--min-epochs", "20", "--patience", "15",
    "--device", DEVICE,
]

# (label, command argv, done-check path or None)
JOBS: list[tuple[str, list[str], Path | None]] = [
    (
        "cpx_3class (shallow mean)",
        [PYTHON, "-m", "idmt_experiments.train", "--run-name", "cpx_3class",
         "--feature-type", "complex_stft", "--preempt", "--resume-training"],
        CKPT / "cnn/direction/cpx_3class/train_summary.json",
    ),
    (
        "cpx_3class_left",
        [PYTHON, "-m", "idmt_experiments.train", "--run-name", "cpx_3class_left",
         "--feature-type", "complex_stft", "--mono-source", "left",
         "--preempt", "--resume-training"],
        CKPT / "cnn/direction/cpx_3class_left/train_summary.json",
    ),
    (
        "cpx_3class_right",
        [PYTHON, "-m", "idmt_experiments.train", "--run-name", "cpx_3class_right",
         "--feature-type", "complex_stft", "--mono-source", "right",
         "--preempt", "--resume-training"],
        CKPT / "cnn/direction/cpx_3class_right/train_summary.json",
    ),
    (
        "deep_cpx_2class_mean",
        [PYTHON, "-m", "idmt_experiments.transfer", "--run-name", "deep_cpx_2class_mean",
         "--feature-type", "complex_stft", "--mono-source", "mean", "--epochs", "100", "--resume",
         *DEEP_EARLY],
        CKPT / "transfer/direction/deep_cpx_2class_mean/train_summary.json",
    ),
    (
        "deep_cpx_2class_left_100ep",
        [PYTHON, "-m", "idmt_experiments.transfer", "--run-name", "deep_cpx_2class_left_100ep",
         "--feature-type", "complex_stft", "--mono-source", "left", "--epochs", "100", "--resume",
         *DEEP_EARLY],
        CKPT / "transfer/direction/deep_cpx_2class_left_100ep/train_summary.json",
    ),
    (
        "deep_cpx_2class_right_100ep",
        [PYTHON, "-m", "idmt_experiments.transfer", "--run-name", "deep_cpx_2class_right_100ep",
         "--feature-type", "complex_stft", "--mono-source", "right", "--epochs", "100", "--resume",
         *DEEP_EARLY],
        CKPT / "transfer/direction/deep_cpx_2class_right_100ep/train_summary.json",
    ),
    (
        "fusion_cpx_2class_100ep",
        [PYTHON, "-m", "idmt_experiments.fusion",
         "--left-checkpoint", str(CKPT / "transfer/direction/deep_cpx_2class_left_100ep/best.pt"),
         "--right-checkpoint", str(CKPT / "transfer/direction/deep_cpx_2class_right_100ep/best.pt"),
         "--run-name", "fusion_cpx_2class_100ep"],
        OUT / "fusion/direction/fusion_cpx_2class_100ep_eval/eval_metrics.json",
    ),
    (
        "fusion_cpx_baseline_2class",
        [PYTHON, "-m", "idmt_experiments.fusion",
         "--left-checkpoint", str(CKPT / "cnn/direction/cpx_3class_left/best.pt"),
         "--right-checkpoint", str(CKPT / "cnn/direction/cpx_3class_right/best.pt"),
         "--run-name", "fusion_cpx_baseline_2class"],
        OUT / "fusion/direction/fusion_cpx_baseline_2class_eval/eval_metrics.json",
    ),
]


def is_done(marker: Path | None) -> bool:
    return marker is not None and marker.is_file()


def run(cmd: list[str], log_path: Path) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n>>> " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=SRC, check=True, env=ENV, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    log_path = OUT / "complex_stft_queue.log"
    print("=" * 72)
    print("Complex STFT queue (skip-if-done; deep runs: preempt after min 20 ep)")
    print(f"  python : {PYTHON}")
    print(f"  log    : {log_path}")
    print("=" * 72)
    pending = [(i, label, cmd, marker) for i, (label, cmd, marker) in enumerate(JOBS, 1) if not is_done(marker)]
    if not pending:
        print("\nAll queue jobs already complete.", flush=True)
        return
    print(f"\n  pending: {len(pending)}/{len(JOBS)} jobs", flush=True)
    for i, label, cmd, marker in pending:
        print("\n" + "#" * 72)
        print(f"JOB {i}/{len(JOBS)}: {label}")
        print("#" * 72, flush=True)
        run(cmd, log_path)
    print("\nComplex STFT queue complete.", flush=True)


if __name__ == "__main__":
    main()
