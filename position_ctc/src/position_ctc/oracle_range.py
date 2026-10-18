"""Geometric path recovery from a range history. No audio.

Exact R(t) does not determine a planar path. This module reconstructs paths
under explicit motion families so the leftover angular ambiguity is
measurable. Orbit alignment is applied only when scoring, never inside the
decoder.
"""

from __future__ import annotations

import numpy as np

from position_ctc.metrics import range_m, score_path

EPS = 1e-8


def path_from_polar(radius: np.ndarray, theta: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


def bearing(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))


def radial_rate(xy: np.ndarray, vel: np.ndarray | None = None, dt: float | None = None) -> np.ndarray:
    """Range rate on the label clock.

    If velocity columns are supplied, this is (p · v) / R. Otherwise it is a
    central difference of ||p||.
    """
    xy = np.asarray(xy, dtype=np.float64)
    radius = range_m(xy)
    if vel is not None:
        vel = np.asarray(vel, dtype=np.float64)
        return (xy * vel).sum(axis=1) / np.maximum(radius, EPS)
    if dt is None or dt <= 0:
        raise ValueError("dt is required when velocity is omitted")
    q = np.gradient(radius, dt)
    return q


def angular_rate_magnitude(radius: np.ndarray, rad_rate: np.ndarray, speed: float) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    rad_rate = np.asarray(rad_rate, dtype=np.float64).reshape(-1)
    speed = float(max(speed, 0.0))
    inside = np.maximum(speed * speed - rad_rate * rad_rate, 0.0)
    return np.sqrt(inside) / np.maximum(radius, EPS)


def integrate_bearing(
    radius: np.ndarray,
    rad_rate: np.ndarray,
    speed: float,
    signs: np.ndarray,
    theta0: float,
    dt: float,
) -> np.ndarray:
    """p(t) = R(t)[cos theta, sin theta] with theta-dot = sign * |theta-dot|."""
    omega = angular_rate_magnitude(radius, rad_rate, speed)
    signs = np.asarray(signs, dtype=np.float64).reshape(-1)
    theta = np.empty_like(omega)
    theta[0] = float(theta0)
    if theta.size > 1:
        theta[1:] = float(theta0) + np.cumsum(signs[:-1] * omega[:-1] * float(dt))
    return path_from_polar(radius, theta)


def fit_straight_pass(radius: np.ndarray, dt: float) -> dict[str, float | np.ndarray]:
    """Least-squares R^2(t) = b^2 + s^2 (t - t_c)^2.

    The reconstructed path is the canonical straight pass x=s(t-t_c), y=b.
    """
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    n = radius.size
    t = np.arange(n, dtype=np.float64) * float(dt)
    design = np.column_stack([np.ones(n), t, t * t])
    coeff, *_ = np.linalg.lstsq(design, radius * radius, rcond=None)
    c0, c1, c2 = [float(v) for v in coeff]
    if c2 <= EPS:
        b = float(np.median(radius))
        return {
            "speed_mps": 0.0,
            "cpa_m": b,
            "t_cpa_s": float(t[int(np.argmin(radius))]),
            "residual_rmse_m": float(np.sqrt(np.mean((radius - b) ** 2))),
            "xy": np.stack([np.zeros(n), np.full(n, b)], axis=1),
        }
    t_cpa = -0.5 * c1 / c2
    b2 = c0 - c2 * t_cpa * t_cpa
    b = float(np.sqrt(max(b2, 0.0)))
    speed = float(np.sqrt(c2))
    xy = np.stack([speed * (t - t_cpa), np.full(n, b)], axis=1)
    fitted_r = range_m(xy)
    return {
        "speed_mps": speed,
        "cpa_m": b,
        "t_cpa_s": float(t_cpa),
        "residual_rmse_m": float(np.sqrt(np.mean((fitted_r - radius) ** 2))),
        "xy": xy,
    }


def constant_speed(vel: np.ndarray) -> float:
    speed = np.linalg.norm(np.asarray(vel, dtype=np.float64), axis=1)
    moving = speed > 0.5
    if not np.any(moving):
        return float(np.median(speed))
    return float(np.median(speed[moving]))


def heading_jump_count(vel: np.ndarray, min_deg: float = 25.0) -> int:
    vel = np.asarray(vel, dtype=np.float64)
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > 0.5
    if int(moving.sum()) < 3:
        return 0
    head = np.unwrap(np.arctan2(vel[moving, 1], vel[moving, 0]))
    step = np.abs(np.diff(head))
    return int(np.sum(step > np.deg2rad(min_deg)))


def _true_signs(xy: np.ndarray, dt: float) -> np.ndarray:
    signs = np.sign(np.gradient(bearing(xy), float(dt)))
    signs[signs == 0.0] = 1.0
    return signs


def flip_sign_path(
    xy: np.ndarray,
    vel: np.ndarray,
    dt: float,
    theta0: float | None = None,
) -> np.ndarray:
    """Global sign flip of theta-dot. Orbit alignment treats this as a gauge."""
    radius = range_m(xy)
    q = radial_rate(xy, vel)
    speed = constant_speed(vel)
    if theta0 is None:
        theta0 = float(bearing(xy)[0])
    return integrate_bearing(radius, q, speed, -_true_signs(xy, dt), theta0, dt)


def branch_at_index(
    xy: np.ndarray,
    vel: np.ndarray,
    dt: float,
    index: int,
) -> np.ndarray:
    """Keep the true sign until ``index``, then reverse it.

    That is not a global rotation or reflection. The radial history is unchanged.
    """
    radius = range_m(xy)
    q = radial_rate(xy, vel)
    speed = constant_speed(vel)
    signs = _true_signs(xy, dt).copy()
    cut = int(np.clip(index, 1, max(len(signs) - 1, 1)))
    signs[cut:] *= -1.0
    return integrate_bearing(radius, q, speed, signs, float(bearing(xy)[0]), dt)


def first_heading_cut(vel: np.ndarray, min_deg: float = 25.0) -> int | None:
    vel = np.asarray(vel, dtype=np.float64)
    speed = np.linalg.norm(vel, axis=1)
    moving = np.flatnonzero(speed > 0.5)
    if moving.size < 3:
        return None
    head = np.unwrap(np.arctan2(vel[moving, 1], vel[moving, 0]))
    step = np.abs(np.diff(head))
    hits = np.flatnonzero(step > np.deg2rad(min_deg))
    if hits.size == 0:
        return None
    return int(moving[int(hits[0]) + 1])


def recover_with_true_signs(xy: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
    radius = range_m(xy)
    q = radial_rate(xy, vel)
    speed = constant_speed(vel)
    theta = bearing(xy)
    return integrate_bearing(radius, q, speed, _true_signs(xy, dt), float(theta[0]), dt)


def recover_with_constant_sign(
    xy: np.ndarray,
    vel: np.ndarray,
    dt: float,
    sign: float = 1.0,
) -> np.ndarray:
    radius = range_m(xy)
    q = radial_rate(xy, vel)
    speed = constant_speed(vel)
    signs = np.full(len(radius), float(np.sign(sign) or 1.0))
    return integrate_bearing(radius, q, speed, signs, float(bearing(xy)[0]), dt)


def recover_without_speed(xy: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
    """Use max|q| as a speed floor. Curved motion then understates |theta-dot|."""
    radius = range_m(xy)
    q = radial_rate(xy, vel)
    speed = float(max(np.max(np.abs(q)), EPS))
    theta = bearing(xy)
    return integrate_bearing(radius, q, speed, _true_signs(xy, dt), float(theta[0]), dt)


def score_decoder(pred_xy: np.ndarray, gt_xy: np.ndarray) -> dict[str, float]:
    scores = score_path(pred_xy, gt_xy)
    scores["radial_mae_m"] = scores["range_mae_m"]
    return scores
