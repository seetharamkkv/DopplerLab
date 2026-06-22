#!/usr/bin/env python3
"""
Run a fixed queue of IDMT benchmarks sequentially (train + auto-eval).

Designed for ~3 h unattended CPU runs with clear progress banners.

Example
-------
cd IDMT_experiments
python -m idmt_experiments.run_batch
python -m idmt_experiments.run_batch --only classical_vehicle_cc vehicle_cc_eusipco
python -m idmt_experiments.run_batch --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR


@dataclass
class BatchJob:
    name: str
    kind: str  # classical | train
    task: str = "direction"
    feature_type: str = "cc"
    mode: str = "eusipco"  # eusipco | weather_holdout | location_loo
    n_classes: int | None = None
    epochs: int = 10
    preempt: bool = True
    batch_size: int = 32
    skip_if_done: bool = True
    done_marker: str | None = None  # path relative to repo outputs/ or checkpoints/
    est_minutes: int = 30
    extra_train_args: list[str] = field(default_factory=list)


# ~3 h CPU queue (SE/CH34): vehicle identity + weather generalization
DEFAULT_QUEUE: list[BatchJob] = [
    BatchJob(
        name="classical_vehicle_cc",
        kind="classical",
        task="vehicle",
        feature_type="cc",
        n_classes=5,
        mode="eusipco",
        done_marker="baselines/classical_vehicle_cc_5class.json",
        est_minutes=20,
    ),
    BatchJob(
        name="vehicle_cc_eusipco",
        kind="train",
        task="vehicle",
        feature_type="cc",
        n_classes=5,
        mode="eusipco",
        epochs=10,
        done_marker="vehicle/vehicle_cc_eusipco/eval_summary.txt",
        est_minutes=90,
    ),
    BatchJob(
        name="direction_cc_weather",
        kind="train",
        task="direction",
        feature_type="cc",
        n_classes=3,
        mode="weather_holdout",
        epochs=10,
        done_marker="direction/direction_cc_weather/eval_summary.txt",
        est_minutes=70,
    ),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _marker_path(job: BatchJob) -> Path | None:
    if not job.done_marker:
        return None
    marker = Path(job.done_marker)
    if marker.is_absolute():
        return marker
    if marker.parts[0] in ("baselines", "direction", "vehicle", "splits", "figures"):
        return DEFAULT_OUTPUT_DIR / marker
    if marker.parts[0] == "checkpoints":
        return DEFAULT_CHECKPOINT_DIR / Path(*marker.parts[1:])
    return DEFAULT_OUTPUT_DIR / marker


def _is_done(job: BatchJob) -> bool:
    path = _marker_path(job)
    if path is None:
        ckpt = DEFAULT_CHECKPOINT_DIR / job.task / job.name / "best.pt"
        return ckpt.exists()
    return path.exists()


def _run_cmd(argv: list[str]) -> int:
    print("\n>>>", " ".join(argv), flush=True)
    return subprocess.call(argv)


def _classical_argv(job: BatchJob, data_dir: Path | None) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "idmt_experiments.run",
        "classical",
        "--task",
        job.task,
        "--feature-type",
        job.feature_type,
        "--mode",
        job.mode,
    ]
    if job.n_classes is not None:
        argv.extend(["--n-classes", str(job.n_classes)])
    if data_dir:
        argv.extend(["--data-dir", str(data_dir)])
    return argv


def _train_argv(job: BatchJob, data_dir: Path | None) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "idmt_experiments.train",
        "--task",
        job.task,
        "--mode",
        job.mode,
        "--feature-type",
        job.feature_type,
        "--run-name",
        job.name,
        "--epochs",
        str(job.epochs),
        "--batch-size",
        str(job.batch_size),
    ]
    if job.preempt:
        argv.append("--preempt")
    if job.n_classes is not None:
        argv.extend(["--n-classes", str(job.n_classes)])
    if data_dir:
        argv.extend(["--data-dir", str(data_dir)])
    argv.extend(job.extra_train_args)
    return argv


def run_job(job: BatchJob, *, data_dir: Path | None, dry_run: bool) -> dict:
    started = datetime.now(timezone.utc)
    t0 = time.time()

    if job.skip_if_done and _is_done(job):
        msg = f"SKIP (already done): {job.name}"
        print(msg, flush=True)
        return {
            "name": job.name,
            "status": "skipped",
            "message": msg,
            "elapsed_s": 0.0,
        }

    if job.kind == "classical":
        argv = _classical_argv(job, data_dir)
    elif job.kind == "train":
        argv = _train_argv(job, data_dir)
    else:
        raise ValueError(f"Unknown job kind: {job.kind}")

    if dry_run:
        print(f"DRY-RUN would execute: {' '.join(argv)}")
        return {"name": job.name, "status": "dry_run", "argv": argv, "elapsed_s": 0.0}

    rc = _run_cmd(argv)
    elapsed = time.time() - t0
    status = "ok" if rc == 0 else "failed"
    return {
        "name": job.name,
        "status": status,
        "return_code": rc,
        "elapsed_s": round(elapsed, 1),
        "started_at": started.isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run IDMT benchmark queue sequentially")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--only", nargs="+", default=None, metavar="JOB", help="Subset of job names")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    p.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    p.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "batch_summary.json",
        help="Write JSON summary here",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    queue = DEFAULT_QUEUE
    if args.only:
        names = set(args.only)
        queue = [j for j in queue if j.name in names]
        missing = names - {j.name for j in queue}
        if missing:
            raise SystemExit(f"Unknown job names: {sorted(missing)}")

    if args.force:
        for job in queue:
            job.skip_if_done = False

    total_est = sum(j.est_minutes for j in queue)
    print("=" * 72)
    print("IDMT BATCH RUNNER")
    print("=" * 72)
    print(f"  jobs          : {len(queue)}")
    print(f"  est. runtime  : ~{total_est} min (CPU, CC features)")
    print(f"  data-dir      : {args.data_dir or '(default IDMT_Traffic)'}")
    print(f"  summary       : {args.summary_out}")
    print()
    for i, job in enumerate(queue, 1):
        done = "done" if _is_done(job) and job.skip_if_done else "pending"
        print(f"  [{i}/{len(queue)}] {job.name:<28} {job.kind:<10} ~{job.est_minutes:>3} min  [{done}]")
    print()

    results: list[dict] = []
    batch_t0 = time.time()
    for i, job in enumerate(queue, 1):
        print("\n" + "#" * 72)
        print(f"BATCH JOB {i}/{len(queue)}: {job.name}")
        print(f"  kind={job.kind}  task={job.task}  mode={job.mode}  feature={job.feature_type}")
        print(f"  est ~{job.est_minutes} min")
        print("#" * 72, flush=True)
        result = run_job(job, data_dir=args.data_dir, dry_run=args.dry_run)
        results.append(result)
        if result.get("status") == "failed":
            print(f"\n*** JOB FAILED: {job.name} (rc={result.get('return_code')}) — stopping batch ***")
            break

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_s": round(time.time() - batch_t0, 1),
        "jobs": results,
    }
    if not args.dry_run:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote batch summary -> {args.summary_out}")

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        raise SystemExit(1)
    print("\nBatch complete.")


if __name__ == "__main__":
    main()
