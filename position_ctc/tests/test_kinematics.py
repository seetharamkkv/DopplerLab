"""Feasible-turn cap: a kink is penalized, a gentle arc is not."""

import math

import numpy as np
import torch

from position_ctc.features import DATA_HOP, DATA_SR
from position_ctc.kinematics import (
    A_LAT_MAX_MPS2,
    R_MIN_M,
    kinematic_penalty,
    project_path,
    violation_fraction,
)

DT = DATA_HOP / float(DATA_SR)


def _polyline(speed: float, headings: list[float], frames_each: int) -> np.ndarray:
    pos = np.zeros(2, dtype=np.float64)
    rows = [pos.copy()]
    step = speed * DT
    for heading in headings:
        direction = np.array([math.cos(heading), math.sin(heading)])
        for _ in range(frames_each):
            pos = pos + direction * step
            rows.append(pos.copy())
    return np.asarray(rows, dtype=np.float32)


def test_radius_and_lateral_cap_match_the_cited_models():
    assert abs(R_MIN_M - 2.70 / math.tan(math.radians(35.0))) < 1e-9
    assert abs(A_LAT_MAX_MPS2 - 0.5 * 9.80665) < 1e-9


def test_straight_and_gentle_arc_are_feasible():
    straight = _polyline(15.0, [0.0], 40)
    assert violation_fraction(straight, DT) == 0.0
    # 80 m radius at 15 m/s is 2.8 m/s^2, under the 0.5 g cap and far above R_min.
    arc = []
    pos = np.zeros(2, dtype=np.float64)
    radius = 80.0
    speed = 15.0
    heading = 0.0
    yaw = speed / radius
    for _ in range(80):
        heading += yaw * DT
        pos = pos + speed * DT * np.array([math.cos(heading), math.sin(heading)])
        arc.append(pos.copy())
    assert violation_fraction(np.asarray(arc, dtype=np.float32), DT) < 0.02


def test_one_frame_corner_is_infeasible_until_projected():
    corner = _polyline(15.0, [0.0, math.pi / 2], 20)
    assert violation_fraction(corner, DT) > 0.0
    feasible = project_path(corner, DT)
    assert violation_fraction(feasible, DT) == 0.0
    assert abs(np.linalg.norm(feasible[0] - corner[0])) < 1e-5


def test_penalty_is_differentiable_and_zero_on_a_straight_line():
    straight = torch.tensor(_polyline(15.0, [0.0], 30)[None, :, :], dtype=torch.float32)
    lengths = torch.tensor([straight.shape[1]])
    assert float(kinematic_penalty(straight, DT, lengths)) < 1e-6
    corner = torch.tensor(_polyline(15.0, [0.0, math.pi / 2], 12)[None, :, :], dtype=torch.float32, requires_grad=True)
    loss = kinematic_penalty(corner, DT, torch.tensor([corner.shape[1]]))
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert corner.grad is not None
    assert torch.isfinite(corner.grad).all()
