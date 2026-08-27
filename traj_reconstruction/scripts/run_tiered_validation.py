#!/usr/bin/env python3
"""Run Phase 4 tiered simulated validation scorecard.

Example:
  python scripts/run_tiered_validation.py --out outputs/tiered_validation
  python scripts/run_tiered_validation.py --out outputs/tiered_validation \\
      --snr 30,20,10,0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traj_reconstruction.validate import run_all_tiers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/tiered_validation"))
    p.add_argument(
        "--snr",
        type=str,
        default="30,20,10,5,0",
        help="Comma-separated SNR grid in dB for Tier 3",
    )
    p.add_argument("--worst", type=int, default=3)
    args = p.parse_args()
    snr = [float(x) for x in args.snr.split(",") if x.strip()]

    print("Running SIMULATED Tier 1–5 validation (orbit RMS headline)…")
    report = run_all_tiers(out_dir=args.out, snr_grid_db=snr, save_worst=args.worst)
    print(json.dumps(report["summary"], indent=2))
    print("Gates:")
    for tier, g in report["gates"].items():
        status = "GO" if g.get("go") else ("WAIVER" if g.get("waiver") else "NO-GO")
        print(f"  {tier}: {status} — {g.get('note')}")
    print(f"Report: {args.out / 'tiered_validation.md'}")


if __name__ == "__main__":
    main()
