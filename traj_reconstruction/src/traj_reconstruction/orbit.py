"""Observer-centered orbit metrics (rotation ± reflection about the mic).

Monaural Doppler + spreading identifies trajectory **shape** up to an SO(2)
rotation (and typically a reflection) about the observer. Headline error is
Procrustes RMS after that alignment — never raw world-frame XY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from traj_reconstruction.contract import STATE_COLUMNS_2D, STATE_COLUMNS_3D


@dataclass(frozen=True)
class OrbitAlignResult:
    """Result of aligning ``pred`` to ``gt`` in the observer plane."""

    aligned_pred: np.ndarray  # (T, 2)
    rotation_rad: float
    reflected: bool
    rms: float
    length_normalized_rms: float
    cpa_index_gt: int


def xy_from_state(state: np.ndarray) -> np.ndarray:
    """Extract mic-centric XY positions from Phase 1 state (N,4) or (N,6)."""
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] not in (4, 6):
        raise ValueError(f"state must be (N, 4) or (N, 6), got {state.shape}")
    return np.column_stack([state[:, 0], state[:, 2]]).astype(np.float64)


def state_column_names(state: np.ndarray) -> tuple[str, ...]:
    state = np.asarray(state)
    if state.shape[1] == 4:
        return STATE_COLUMNS_2D
    if state.shape[1] == 6:
        return STATE_COLUMNS_3D
    raise ValueError(f"unexpected state width {state.shape[1]}")


def _rotate_xy(xy: np.ndarray, rotation_rad: float) -> np.ndarray:
    c = float(np.cos(rotation_rad))
    s = float(np.sin(rotation_rad))
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([c * x - s * y, s * x + c * y])


def _reflect_x(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    out[:, 0] *= -1.0
    return out


def canonical_xy(xy: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Gauge-fix XY so CPA lies on +y and travel tends +x at CPA.

    Matches DopplerSim ``canonical_frame_state`` for the position channels.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (T, 2), got {xy.shape}")
    if xy.shape[0] == 0:
        raise ValueError("xy is empty")

    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    cpa = int(np.argmin(r))
    if r[cpa] < 1e-9:
        return xy.copy(), {
            "rotation_rad": 0.0,
            "reflected_x": False,
            "cpa_index": cpa,
            "skipped": "cpa_on_mic_axis",
        }

    ang = float(np.arctan2(xy[cpa, 1], xy[cpa, 0]))
    rot = 0.5 * np.pi - ang
    xy2 = _rotate_xy(xy, rot)

    # Travel direction proxy: finite difference at CPA.
    if cpa + 1 < xy2.shape[0]:
        vx = float(xy2[cpa + 1, 0] - xy2[cpa, 0])
    elif cpa > 0:
        vx = float(xy2[cpa, 0] - xy2[cpa - 1, 0])
    else:
        vx = 0.0

    reflected = False
    if vx < 0.0:
        xy2 = _reflect_x(xy2)
        reflected = True

    return xy2, {
        "rotation_rad": rot,
        "reflected_x": reflected,
        "cpa_index": cpa,
        "skipped": None,
    }


def orbit_align(pred_xy: np.ndarray, gt_xy: np.ndarray) -> OrbitAlignResult:
    """Align ``pred_xy`` to ``gt_xy`` by rotation ± reflection about the origin.

    Searches the optimal planar rotation for both the unreflected and
    x-reflected prediction; returns the lower RMS alignment.
    """
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"expected (T, 2) paths, got {pred.shape}")
    if pred.shape[0] == 0:
        raise ValueError("empty trajectories")

    r_gt = np.sqrt(gt[:, 0] ** 2 + gt[:, 1] ** 2)
    cpa_gt = int(np.argmin(r_gt))
    path_len = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    path_len = max(path_len, 1e-12)

    best: OrbitAlignResult | None = None
    for reflected in (False, True):
        p = _reflect_x(pred) if reflected else pred
        # Kabsch / Procrustes in 2D about fixed origin (no translation).
        # R = argmin ||R p - g|| ; using SVD of p^T g.
        h = p.T @ gt  # 2x2
        u, _, vt = np.linalg.svd(h)
        r_mat = vt.T @ u.T
        if np.linalg.det(r_mat) < 0:
            vt = vt.copy()
            vt[-1, :] *= -1.0
            r_mat = vt.T @ u.T
        aligned = (r_mat @ p.T).T
        rms = float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1))))
        rot = float(np.arctan2(r_mat[1, 0], r_mat[0, 0]))
        cand = OrbitAlignResult(
            aligned_pred=aligned.astype(np.float64),
            rotation_rad=rot,
            reflected=reflected,
            rms=rms,
            length_normalized_rms=rms / path_len,
            cpa_index_gt=cpa_gt,
        )
        if best is None or cand.rms < best.rms:
            best = cand

    assert best is not None
    return best


def orbit_family(
    xy: np.ndarray,
    *,
    n_rotations: int = 36,
    include_mirror: bool = True,
) -> list[np.ndarray]:
    """Discrete samples of the rotational (± mirror) family about the origin."""
    xy = np.asarray(xy, dtype=np.float64)
    if n_rotations < 1:
        raise ValueError("n_rotations must be >= 1")
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_rotations), endpoint=False)
    family: list[np.ndarray] = [_rotate_xy(xy, float(a)) for a in angles]
    if include_mirror:
        mir = _reflect_x(xy)
        family.extend(_rotate_xy(mir, float(a)) for a in angles)
    return family
