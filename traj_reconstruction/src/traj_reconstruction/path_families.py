"""Programmatic freehand polylines for simulated Tier-1 batches.

Mic at origin. Paths are mic-centric world coordinates (meters).
Compatible with DopplerSim path2d constant-speed resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traj_reconstruction.contract import (
    PATH_FAMILY_ARC,
    PATH_FAMILY_MULTI_CPA,
    PATH_FAMILY_S_CURVE,
    PATH_FAMILY_STRAIGHT,
    PATH_FAMILY_U_TURN,
    PATH_FAMILIES,
)


@dataclass(frozen=True)
class PathSpec:
    family: str
    polyline: np.ndarray  # (N, 2)
    speed_mps: float
    notes: str = ""


def _rot2(xy: np.ndarray, ang: float) -> np.ndarray:
    c, s = float(np.cos(ang)), float(np.sin(ang))
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([c * x - s * y, s * x + c * y])


def make_straight(
    *,
    cpa_distance_m: float,
    half_length_m: float,
    heading_rad: float = 0.0,
    n_pts: int = 64,
) -> np.ndarray:
    """Straight flyby: CPA on +y after heading rotation (pre-rotation CPA at (0, h))."""
    h = float(cpa_distance_m)
    L = float(half_length_m)
    x = np.linspace(-L, L, int(n_pts))
    y = np.full_like(x, h)
    return _rot2(np.column_stack([x, y]), float(heading_rad))


def make_arc(
    *,
    cpa_distance_m: float,
    radius_m: float,
    sweep_rad: float = np.deg2rad(60.0),
    heading_rad: float = 0.0,
    n_pts: int = 80,
) -> np.ndarray:
    """Circular arc whose closest approach is near (0, cpa_distance_m) before heading rot."""
    h = float(cpa_distance_m)
    R = float(radius_m)
    # Center of circle below CPA so min range ≈ h.
    cy = h - R
    ang0 = -0.5 * float(sweep_rad)
    angs = np.linspace(ang0, ang0 + float(sweep_rad), int(n_pts))
    x = R * np.sin(angs)
    y = cy + R * np.cos(angs)
    return _rot2(np.column_stack([x, y]), float(heading_rad))


def make_s_curve(
    *,
    cpa_distance_m: float,
    half_length_m: float,
    amplitude_m: float,
    heading_rad: float = 0.0,
    n_pts: int = 96,
) -> np.ndarray:
    """S-curve: lateral sine wiggle while mean CPA distance ≈ h."""
    h = float(cpa_distance_m)
    L = float(half_length_m)
    A = float(amplitude_m)
    x = np.linspace(-L, L, int(n_pts))
    y = h + A * np.sin(np.pi * x / L)
    return _rot2(np.column_stack([x, y]), float(heading_rad))


def make_u_turn(
    *,
    cpa_distance_m: float,
    approach_m: float,
    turn_radius_m: float,
    heading_rad: float = 0.0,
    n_pts: int = 100,
) -> np.ndarray:
    """Approach, 180° turn, recede — two legs + semicircle."""
    h = float(cpa_distance_m)
    a = float(approach_m)
    R = float(turn_radius_m)
    n_leg = max(int(n_pts) // 3, 8)
    n_turn = max(int(n_pts) - 2 * n_leg, 16)

    # Approach along x from -a to 0 at y=h.
    x1 = np.linspace(-a, 0.0, n_leg)
    y1 = np.full_like(x1, h)
    # Semicircle center (0, h+R), from south to north going through +x.
    angs = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_turn)
    x2 = R * np.cos(angs)
    y2 = (h + R) + R * np.sin(angs)
    # Recede back along -x at y = h + 2R.
    x3 = np.linspace(0.0, -a, n_leg)
    y3 = np.full_like(x3, h + 2.0 * R)
    xy = np.vstack(
        [
            np.column_stack([x1, y1]),
            np.column_stack([x2[1:], y2[1:]]),
            np.column_stack([x3[1:], y3[1:]]),
        ]
    )
    return _rot2(xy, float(heading_rad))


def make_multi_cpa(
    *,
    cpa_distance_m: float,
    leg_m: float,
    offset_m: float,
    heading_rad: float = 0.0,
    n_pts: int = 120,
) -> np.ndarray:
    """Pass once, lateral shift, pass again (two CPA-like minima)."""
    h = float(cpa_distance_m)
    L = float(leg_m)
    d = float(offset_m)
    n = max(int(n_pts) // 3, 10)
    x1 = np.linspace(-L, L, n)
    y1 = np.full_like(x1, h)
    t = np.linspace(0.0, 1.0, n)
    x2 = L * (1.0 - 2.0 * t)  # L → -L
    y2 = h + d * np.sin(np.pi * t)
    x3 = np.linspace(-L, L, n)
    y3 = np.full_like(x3, h + d)
    xy = np.vstack(
        [
            np.column_stack([x1, y1]),
            np.column_stack([x2[1:], y2[1:]]),
            np.column_stack([x3[1:], y3[1:]]),
        ]
    )
    return _rot2(xy, float(heading_rad))


def generate_path(
    family: str,
    *,
    speed_mps: float,
    cpa_distance_m: float,
    rng: np.random.Generator,
    heading_rad: float | None = None,
) -> PathSpec:
    """Sample one polyline for a named family with mild random geometry."""
    if family not in PATH_FAMILIES:
        raise ValueError(f"unknown path family {family!r}; expected one of {PATH_FAMILIES}")
    heading = float(rng.uniform(0.0, 2.0 * np.pi) if heading_rad is None else heading_rad)
    h = float(cpa_distance_m)
    v = float(speed_mps)

    if family == PATH_FAMILY_STRAIGHT:
        half = float(rng.uniform(40.0, 120.0))
        poly = make_straight(cpa_distance_m=h, half_length_m=half, heading_rad=heading)
        notes = f"half_length_m={half:.1f}"
    elif family == PATH_FAMILY_ARC:
        R = float(rng.uniform(max(h + 5.0, 30.0), max(h + 5.0, 30.0) + 80.0))
        sweep = float(rng.uniform(np.deg2rad(40.0), np.deg2rad(100.0)))
        poly = make_arc(
            cpa_distance_m=h, radius_m=R, sweep_rad=sweep, heading_rad=heading
        )
        notes = f"radius_m={R:.1f},sweep_deg={np.rad2deg(sweep):.1f}"
    elif family == PATH_FAMILY_S_CURVE:
        half = float(rng.uniform(50.0, 140.0))
        amp = float(rng.uniform(3.0, min(0.4 * h, 25.0)))
        poly = make_s_curve(
            cpa_distance_m=h,
            half_length_m=half,
            amplitude_m=amp,
            heading_rad=heading,
        )
        notes = f"half_length_m={half:.1f},amplitude_m={amp:.1f}"
    elif family == PATH_FAMILY_U_TURN:
        approach = float(rng.uniform(40.0, 100.0))
        R = float(rng.uniform(8.0, 25.0))
        poly = make_u_turn(
            cpa_distance_m=h,
            approach_m=approach,
            turn_radius_m=R,
            heading_rad=heading,
        )
        notes = f"approach_m={approach:.1f},turn_radius_m={R:.1f}"
    else:  # multi_cpa
        leg = float(rng.uniform(40.0, 90.0))
        off = float(rng.uniform(8.0, min(40.0, 1.5 * h)))
        poly = make_multi_cpa(
            cpa_distance_m=h, leg_m=leg, offset_m=off, heading_rad=heading
        )
        notes = f"leg_m={leg:.1f},offset_m={off:.1f}"

    return PathSpec(family=family, polyline=poly.astype(np.float64), speed_mps=v, notes=notes)
