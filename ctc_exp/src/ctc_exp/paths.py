"""Load dataset roots from environment variables or a local JSON file."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATHS_FILE = REPO_ROOT / "dataset_paths.json"
LEGACY_PATHS_FILE = REPO_ROOT / "ref_docs" / "dataset_paths.json"
EXAMPLE_PATHS_FILE = REPO_ROOT / "dataset_paths.example.json"

KEYS = ("vs13", "dopplersim_test_dataset", "dopplersim_unconstrained")
ENV = {
    "vs13": "DOPPLER_VS13",
    "dopplersim_test_dataset": "DOPPLER_SIM_TEST",
    "dopplersim_unconstrained": "DOPPLER_SIM_UNCONSTRAINED",
}


def load_dataset_paths(path: Path | None = None) -> dict[str, Path]:
    env_paths = {key: os.environ.get(name, "").strip() for key, name in ENV.items()}
    if all(env_paths.values()):
        return {key: Path(value).expanduser() for key, value in env_paths.items()}

    candidates = [path, DEFAULT_PATHS_FILE, LEGACY_PATHS_FILE]
    src = next((item for item in candidates if item is not None and Path(item).is_file()), None)
    if src is None:
        missing = ", ".join(f"{ENV[key]} or {key}" for key in KEYS)
        raise SystemExit(
            "Dataset roots are not configured. Copy "
            f"{EXAMPLE_PATHS_FILE.name} to dataset_paths.json, or set {missing}."
        )
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for key in KEYS:
        if key not in data:
            raise KeyError(f"missing {key!r} in {src}")
        out[key] = Path(data[key]).expanduser()
    return out
