"""Position and distance errors for a path around a microphone at the origin.

A single omnidirectional mic cannot see absolute heading. Position error is
the RMS after the best rotation and reflection about the origin. Distance
error is the range from the microphone, which that alignment does not change.
"""

from __future__ import annotations

import numpy as np


def _reflect_x(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    out[:, 0] *= -1.0
    return out


def align_path(pred_xy: np.ndarray, gt_xy: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """Rotate and optionally reflect ``pred`` about the origin to match ``gt``.

    Returns the aligned path, whether a reflection was used, and the RMS in meters.
    """
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    best_rms = None
    best_xy = pred
    best_reflected = False
    for reflected in (False, True):
        aligned = _align(pred if not reflected else _reflect_x(pred), gt)
        rms = float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1))))
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_xy = aligned
            best_reflected = reflected
    return best_xy, best_reflected, float(best_rms)


def orbit_rms(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """RMS in meters after rotation and reflection about the microphone."""
    return align_path(pred_xy, gt_xy)[2]


def _align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    cross = pred.T @ gt
    u, _, vt = np.linalg.svd(cross)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return (rotation @ pred.T).T


def range_m(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return np.sqrt(np.sum(xy * xy, axis=1))


def score_path(pred_xy: np.ndarray, gt_xy: np.ndarray) -> dict[str, float]:
    """Orbit position error and observer-range error for one clip."""
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    pred_r = range_m(pred)
    gt_r = range_m(gt)
    gap = pred_r - gt_r
    return {
        "orbit_rms_m": orbit_rms(pred, gt),
        "canonical_rms_m": float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1)))),
        "range_mae_m": float(np.mean(np.abs(gap))),
        "range_rmse_m": float(np.sqrt(np.mean(gap * gap))),
        "cpa_abs_m": float(abs(pred_r.min() - gt_r.min())),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    out: dict[str, float] = {"n": float(len(rows))}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_median"] = float(np.median(values))
    return out
