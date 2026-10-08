"""Signed velocity-component bins, shared by the classifiers, CTC, and Viterbi."""

from __future__ import annotations

import numpy as np

from position_ctc.alphabet import VEL_CENTERS_MPS, VEL_MAX_MPS, VEL_MIN_MPS, VEL_STEP_MPS

N_VEL = len(VEL_CENTERS_MPS)
CENTERS_MPS = np.asarray(VEL_CENTERS_MPS, dtype=np.float64)


def velocity_bin_index(mps: np.ndarray) -> np.ndarray:
    """Nearest bin index in ``[0, N_VEL)``. Values outside the alphabet clip."""
    lo = int(round(VEL_MIN_MPS / VEL_STEP_MPS))
    hi = int(round(VEL_MAX_MPS / VEL_STEP_MPS))
    k = np.rint(np.asarray(mps, dtype=np.float64) / VEL_STEP_MPS).astype(np.int64)
    return np.clip(k, lo, hi) - lo


def collapse_runs(ids: np.ndarray) -> np.ndarray:
    """Drop consecutive repeats. ``[4, 4, 3, 3, 1]`` becomes ``[4, 3, 1]``."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return ids
    keep = np.empty(ids.size, dtype=bool)
    keep[0] = True
    keep[1:] = ids[1:] != ids[:-1]
    return ids[keep]


def trapezoid_velocity_gap(state: np.ndarray, dt: float) -> float:
    """Median gap between ``Δp/Δt`` and the trapezoidal labeled velocity, in m/s.

    ``state`` columns are ``x, vx, y, vy``. A small number means the position
    and velocity labels describe the same motion.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.shape[0] < 2:
        return 0.0
    dpx = np.diff(state[:, 0]) / float(dt)
    dpy = np.diff(state[:, 2]) / float(dt)
    vx = 0.5 * (state[:-1, 1] + state[1:, 1])
    vy = 0.5 * (state[:-1, 3] + state[1:, 3])
    err = np.concatenate([np.abs(dpx - vx), np.abs(dpy - vy)])
    return float(np.median(err))
