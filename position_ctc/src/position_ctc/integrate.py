"""Position from velocity. Velocity is piecewise constant between bin changes."""

from __future__ import annotations

import numpy as np


def integrate_velocity(
    vx: np.ndarray,
    vy: np.ndarray,
    dt: float,
    *,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Integrate ``(vx, vy)`` with a forward Euler step. Returns ``(T, 2)``."""
    vx = np.asarray(vx, dtype=np.float64).reshape(-1)
    vy = np.asarray(vy, dtype=np.float64).reshape(-1)
    if vx.shape != vy.shape:
        raise ValueError("vx and vy length mismatch")
    if vx.size == 0:
        raise ValueError("empty velocity")
    xy = np.empty((vx.size, 2), dtype=np.float64)
    xy[0, 0] = float(x0)
    xy[0, 1] = float(y0)
    step = float(dt)
    for i in range(vx.size - 1):
        xy[i + 1, 0] = xy[i, 0] + vx[i] * step
        xy[i + 1, 1] = xy[i, 1] + vy[i] * step
    return xy


def integrate_trapezoid(
    vx: np.ndarray,
    vy: np.ndarray,
    dt: float,
    *,
    x0: float,
    y0: float,
) -> np.ndarray:
    """Trapezoid integration. ``p`` gains ``(v_t + v_{t+1}) / 2 * dt`` each frame.

    Returns ``(T, 2)`` in meters. The first sample is the anchor ``(x0, y0)``.
    """
    vx = np.asarray(vx, dtype=np.float64).reshape(-1)
    vy = np.asarray(vy, dtype=np.float64).reshape(-1)
    if vx.shape != vy.shape:
        raise ValueError("vx and vy length mismatch")
    if vx.size == 0:
        raise ValueError("empty velocity")
    xy = np.empty((vx.size, 2), dtype=np.float64)
    xy[0, 0] = float(x0)
    xy[0, 1] = float(y0)
    if vx.size == 1:
        return xy
    dt_arr = np.asarray(dt, dtype=np.float64)
    if dt_arr.ndim == 0:
        dt_arr = np.full(vx.size - 1, float(dt_arr))
    elif dt_arr.shape != (vx.size - 1,):
        raise ValueError("dt must be a scalar or one value per step")
    xy[1:, 0] = xy[0, 0] + np.cumsum(0.5 * (vx[:-1] + vx[1:]) * dt_arr)
    xy[1:, 1] = xy[0, 1] + np.cumsum(0.5 * (vy[:-1] + vy[1:]) * dt_arr)
    return xy


def hold_velocity_bins(
    vx_bins: np.ndarray,
    vy_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Piecewise-constant bin centers, plus the frame index of each change."""
    vx_bins = np.asarray(vx_bins, dtype=np.float64).reshape(-1)
    vy_bins = np.asarray(vy_bins, dtype=np.float64).reshape(-1)
    if vx_bins.shape != vy_bins.shape or vx_bins.size == 0:
        raise ValueError("velocity bins")
    held_x = np.empty_like(vx_bins)
    held_y = np.empty_like(vy_bins)
    held_x[0] = vx_bins[0]
    held_y[0] = vy_bins[0]
    changes: list[int] = []
    for i in range(1, vx_bins.size):
        changed = vx_bins[i] != vx_bins[i - 1] or vy_bins[i] != vy_bins[i - 1]
        if changed:
            changes.append(i)
            held_x[i] = vx_bins[i]
            held_y[i] = vy_bins[i]
        else:
            held_x[i] = held_x[i - 1]
            held_y[i] = held_y[i - 1]
    return held_x, held_y, changes
