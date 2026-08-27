#!/usr/bin/env python3
"""Demo: orbit metric is invariant to rotation (simulated path, no audio model).

Usage:
  python scripts/demo_orbit_metric.py
  python scripts/demo_orbit_metric.py --phase1 /path/to/DopplerSim/renders/<id>/phase1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from traj_reconstruction import load_phase1_sample, orbit_align, orbit_family, xy_from_state


def _synthetic_xy(n: int = 240) -> np.ndarray:
    t = np.linspace(0.0, 5.0, n)
    # Mild arc in the plane (simulated freehand-like).
    x = 25.0 * (t - 2.5)
    y = 12.0 + 0.015 * x**2
    return np.column_stack([x, y])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1",
        type=Path,
        default=None,
        help="Optional DopplerSim Phase 1 folder (simulated). "
        "If omitted, uses a synthetic arc.",
    )
    args = parser.parse_args()

    if args.phase1 is not None:
        sample = load_phase1_sample(args.phase1)
        src = sample.canonical_state_frames
        if src is None:
            src = sample.state_frames
        xy = xy_from_state(src)
        label = f"sim Phase 1 ({sample.path_type}) @ {sample.root}"
    else:
        xy = _synthetic_xy()
        label = "synthetic arc (no Phase 1 path given)"

    print(f"Trajectory reconstruction orbit demo — SIMULATED ONLY")
    print(f"Source: {label}")
    print(f"Points: {xy.shape[0]}")

    for deg in (0, 45, 90, 180):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        rot = np.column_stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])
        res = orbit_align(rot, xy)
        print(f"  rotate {deg:3d}° → orbit RMS = {res.rms:.3e}  reflected={res.reflected}")

    mir = xy.copy()
    mir[:, 0] *= -1.0
    res_m = orbit_align(mir, xy)
    print(f"  mirror x     → orbit RMS = {res_m.rms:.3e}  reflected={res_m.reflected}")

    family = orbit_family(xy, n_rotations=8, include_mirror=True)
    print(f"Rotational family size (8 angles × {{path, mirror}}): {len(family)}")
    print("Done. Headline metric is orbit RMS, not world-frame XY.")


if __name__ == "__main__":
    main()
