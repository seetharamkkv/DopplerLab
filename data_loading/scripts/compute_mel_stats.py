#!/usr/bin/env python3
"""Fit mel mean/std on fit_uids only and refuse if val/test would leak in."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passby_data.config import DEFAULT_REAL_ROOT, AudioConfig  # noqa: E402
from passby_data.leakage import assert_no_leakage  # noqa: E402
from passby_data.mel_stats import compute_mel_stats  # noqa: E402
from passby_data.splits import load_split, partition_paths  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--real_root", type=Path, default=DEFAULT_REAL_ROOT)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    split = load_split(args.split)
    fit_uids = list(split.fit_uids or split.train_uids)
    assert_no_leakage(split, stats_uids=fit_uids, eval_uids=split.test_uids)

    paths, _ = partition_paths(args.real_root, fit_uids)
    stats = compute_mel_stats(paths, cfg=AudioConfig(), save_path=args.out)
    print(f"Wrote {args.out} from {len(paths)} fit clips")
    print(f"mean shape={stats.mean.shape} std shape={stats.std.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
