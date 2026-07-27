#!/usr/bin/env python3
"""Verify a split JSON has no train/test / stats / eval leakage."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passby_data.cli import verify_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(verify_main())
