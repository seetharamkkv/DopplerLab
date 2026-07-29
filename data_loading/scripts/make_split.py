#!/usr/bin/env python3
"""Build a leakage-aware pass-by split (thin wrapper around passby_data.cli)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passby_data.cli import make_split_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(make_split_main())
