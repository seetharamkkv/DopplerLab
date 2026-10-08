"""Orbit-aware geometry. Mic at origin."""

from __future__ import annotations

import numpy as np


def straight_passby_xy(
    times_s: np.ndarray,
    *,
    speed_mps: float,
    cpa_s: float,
    cpa_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-speed line: x = v (t - t_cpa), y = d_cpa. Mic at origin.

    Range is Pythagoras: r(t) = sqrt(x(t)^2 + d_cpa^2). Velocity is (v, 0).
    """
    t = np.asarray(times_s, dtype=np.float64).reshape(-1)
    v = float(speed_mps)
    x = v * (t - float(cpa_s))
    y = np.full_like(x, float(cpa_m))
    xy = np.stack([x, y], axis=1)
    vel = np.stack([np.full_like(x, v), np.zeros_like(x)], axis=1)
    return xy, vel


def xy_from_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    return np.column_stack([state[:, 0], state[:, 2]])


def _reflect_x(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    out[:, 0] *= -1.0
    return out


def orbit_rms(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """Procrustes RMS after rotation ± reflection about the origin (no translation)."""
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    best = None
    for reflected in (False, True):
        p = _reflect_x(pred) if reflected else pred
        h = p.T @ gt
        u, _, vt = np.linalg.svd(h)
        r_mat = vt.T @ u.T
        if np.linalg.det(r_mat) < 0:
            vt = vt.copy()
            vt[-1, :] *= -1.0
            r_mat = vt.T @ u.T
        aligned = (r_mat @ p.T).T
        rms = float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1))))
        if best is None or rms < best:
            best = rms
    return float(best)


def downsample_series(arr: np.ndarray, n_out: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if n_out <= 0:
        raise ValueError("n_out")
    if len(arr) == n_out:
        return arr
    t_in = np.linspace(0.0, 1.0, len(arr))
    t_out = np.linspace(0.0, 1.0, n_out)
    if arr.ndim == 1:
        return np.interp(t_out, t_in, arr)
    cols = [np.interp(t_out, t_in, arr[:, j]) for j in range(arr.shape[1])]
    return np.stack(cols, axis=1)
