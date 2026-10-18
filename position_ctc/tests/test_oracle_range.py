"""Oracle range recovers a straight pass and leaves a curved path ambiguous."""

import numpy as np

from position_ctc.metrics import orbit_rms, range_m
from position_ctc.oracle_range import (
    branch_at_index,
    fit_straight_pass,
    flip_sign_path,
    recover_with_true_signs,
    recover_without_speed,
)


DT = 0.01161


def _straight(speed: float = 15.0, cpa: float = 8.0, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float64) * DT
    t_cpa = t[n // 2]
    x = speed * (t - t_cpa)
    y = np.full(n, cpa)
    xy = np.stack([x, y], axis=1)
    vel = np.stack([np.full(n, speed), np.zeros(n)], axis=1)
    return xy, vel


def _corner(speed: float = 15.0, n_each: int = 200) -> tuple[np.ndarray, np.ndarray]:
    step = speed * DT
    x1 = np.arange(n_each, dtype=np.float64) * step - 20.0
    y1 = np.full(n_each, 8.0)
    x2 = np.full(n_each, x1[-1])
    y2 = 8.0 + np.arange(1, n_each + 1) * step
    xy = np.stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])], axis=1)
    vel = np.zeros_like(xy)
    vel[:n_each, 0] = speed
    vel[n_each:, 1] = speed
    return xy, vel


def test_straight_pass_is_recovered_from_range():
    xy, _ = _straight()
    fitted = fit_straight_pass(range_m(xy), DT)
    assert fitted["residual_rmse_m"] < 0.05
    assert abs(fitted["speed_mps"] - 15.0) < 0.05
    assert abs(fitted["cpa_m"] - 8.0) < 0.05
    assert orbit_rms(fitted["xy"], xy) < 0.05


def test_true_signs_and_speed_recover_the_path():
    xy, vel = _corner()
    pred = recover_with_true_signs(xy, vel, DT)
    assert orbit_rms(pred, xy) < 0.2
    assert np.mean(np.abs(range_m(pred) - range_m(xy))) < 1e-6


def test_global_sign_flip_is_removed_by_orbit_alignment():
    xy, vel = _corner()
    alt = flip_sign_path(xy, vel, DT)
    assert np.mean(np.abs(range_m(alt) - range_m(xy))) < 1e-6
    assert orbit_rms(alt, xy) < 0.1


def test_sign_switch_at_a_corner_is_a_different_shape():
    xy, vel = _corner()
    alt = branch_at_index(xy, vel, DT, 200)
    assert np.mean(np.abs(range_m(alt) - range_m(xy))) < 1e-6
    assert orbit_rms(alt, xy) > 5.0


def test_straight_pass_flip_is_a_gauge_transform():
    xy, vel = _straight()
    alt = flip_sign_path(xy, vel, DT)
    assert np.mean(np.abs(range_m(alt) - range_m(xy))) < 1e-6
    assert orbit_rms(alt, xy) < 0.05


def test_speed_omission_hurts_a_turn():
    xy, vel = _corner()
    pred = recover_without_speed(xy, vel, DT)
    assert np.mean(np.abs(range_m(pred) - range_m(xy))) < 1e-6
    assert orbit_rms(pred, xy) > orbit_rms(recover_with_true_signs(xy, vel, DT), xy)
