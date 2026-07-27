#!/usr/bin/env python3
"""Leave-one-vehicle-out folds (unknown-car claim).

Example:
  python scripts/make_lovo_split.py --real_root ../speed_estimation/passby  # or your real_root
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passby_data.cli import make_lovo_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(make_lovo_main())
