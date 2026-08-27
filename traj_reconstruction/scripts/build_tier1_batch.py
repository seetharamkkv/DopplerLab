#!/usr/bin/env python3
"""Build a simulated Tier-1 freehand Phase 1 batch (audio + orbit GT).

Example (smoke):
  python scripts/build_tier1_batch.py --out data/tier1_smoke --n-straight 4 --n-arc 4

Example (larger):
  python scripts/build_tier1_batch.py --out data/tier1_v1 --n-straight 100 --n-arc 100 \\
      --n-s-curve 50 --n-u-turn 30 --n-multi-cpa 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traj_reconstruction import audit_batch, build_tier1_batch, write_splits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Output batch directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-straight", type=int, default=20)
    p.add_argument("--n-arc", type=int, default=20)
    p.add_argument("--n-s-curve", type=int, default=10)
    p.add_argument("--n-u-turn", type=int, default=5)
    p.add_argument("--n-multi-cpa", type=int, default=5)
    p.add_argument("--speed-min", type=float, default=5.0)
    p.add_argument("--speed-max", type=float, default=40.0)
    p.add_argument("--cpa-min", type=float, default=5.0)
    p.add_argument("--cpa-max", type=float, default=80.0)
    p.add_argument("--f0", type=float, default=500.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--holdout-families",
        nargs="*",
        default=["u_turn"],
        help="Path families reserved for test split (default: u_turn)",
    )
    args = p.parse_args()

    n_per = {
        "straight": args.n_straight,
        "arc": args.n_arc,
        "s_curve": args.n_s_curve,
        "u_turn": args.n_u_turn,
        "multi_cpa": args.n_multi_cpa,
    }
    families = [k for k, v in n_per.items() if v > 0]
    n_per = {k: v for k, v in n_per.items() if v > 0}

    print("Building SIMULATED Tier-1 freehand batch (pure tone + freehand paths)…")
    out = build_tier1_batch(
        args.out,
        n_per_family=n_per,
        speed_range=(args.speed_min, args.speed_max),
        cpa_range=(args.cpa_min, args.cpa_max),
        seed=args.seed,
        families=families,
        resume=not args.no_resume,
        f0_hz=args.f0,
    )
    splits = write_splits(out, holdout_families=args.holdout_families, seed=args.seed)
    report = audit_batch(out)
    print(f"Wrote {out}")
    print(f"Splits: { {k: len(v) for k, v in splits.items()} }")
    print(f"Audit ok={report['ok']} families={report['family_counts']}")
    if report["issues"]:
        print("Issues:")
        print(json.dumps(report["issues"], indent=2))


if __name__ == "__main__":
    main()
