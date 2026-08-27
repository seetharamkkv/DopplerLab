"""Train / val / test splits for simulated freehand batches."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from traj_reconstruction.dataset import DatasetError


def _read_manifest(batch_dir: Path) -> list[dict[str, str]]:
    path = batch_dir / "dataset.csv"
    if not path.is_file():
        raise DatasetError(f"missing dataset.csv under {batch_dir}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_splits(
    batch_dir: Path | str,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    holdout_families: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Create split JSON files.

    Default: random split. If ``holdout_families`` is set, those families go
    entirely to test (for path-family generalization); remaining randomly split.
    """
    batch_dir = Path(batch_dir)
    rows = _read_manifest(batch_dir)
    if not rows:
        raise DatasetError("empty dataset.csv")

    holdout = set(holdout_families or [])
    test_ids = [r["sample_id"] for r in rows if r.get("path_family") in holdout]
    rest = [r for r in rows if r.get("path_family") not in holdout]

    rng = np.random.default_rng(int(seed))
    order = list(range(len(rest)))
    rng.shuffle(order)
    n = len(order)
    n_test_extra = int(round(n * float(test_fraction))) if not holdout else 0
    n_val = int(round(n * float(val_fraction)))
    test_extra = [rest[i]["sample_id"] for i in order[:n_test_extra]]
    val_ids = [rest[i]["sample_id"] for i in order[n_test_extra : n_test_extra + n_val]]
    train_ids = [rest[i]["sample_id"] for i in order[n_test_extra + n_val :]]
    test_ids = list(dict.fromkeys(test_ids + test_extra))

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    out = {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "holdout_families": sorted(holdout),
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    (batch_dir / "splits.json").write_text(json.dumps(out, indent=2))
    for name, ids in splits.items():
        (batch_dir / f"split_{name}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    return splits
