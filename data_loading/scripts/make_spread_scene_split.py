#!/usr/bin/env python3
"""Scene split with opt-in near-twin spreading (default min-speed-gap=2).

Example:
  python scripts/make_spread_scene_split.py --real_root ../speed_estimation/passby  # or your real_root
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passby_data.cli import make_spread_scene_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(make_spread_scene_main())
