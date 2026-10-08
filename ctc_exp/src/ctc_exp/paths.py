"""Load local dataset roots from ``ref_docs/dataset_paths.json``."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATHS_FILE = REPO_ROOT / "ref_docs" / "dataset_paths.json"

KEYS = ("vs13", "dopplersim_test_dataset", "dopplersim_unconstrained")


def load_dataset_paths(path: Path | None = None) -> dict[str, Path]:
    src = path or DEFAULT_PATHS_FILE
    data = json.loads(src.read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for key in KEYS:
        if key not in data:
            raise KeyError(f"missing {key!r} in {src}")
        out[key] = Path(data[key]).expanduser().resolve()
    return out
