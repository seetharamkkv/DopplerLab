"""Feasible-turn limits for a predicted planar path.

The bound is a passenger-car capability limit, not a comfort preference.

Geometric lock, kinematic bicycle model (Kong, Pfeiffer, Schildbach, and
Borrelli, IEEE IV 2015): yaw rate is ``v * tan(delta) / L``, so the tightest
centerline radius is ``R_min = L / tan(delta_max)``. With a 2.70 m wheelbase
and 35 degrees of road-wheel steer, ``R_min`` is 3.86 m.

Lateral acceleration, Polack, Altché, d'Andréa-Novel, and de La Fortelle,
IEEE IV 2017: a kinematic bicycle stays consistent with a 9-DoF model when
``a_y <= 0.5 * mu * g``. Their dynamic simulator saturated near 0.84 g, which
is a higher, tire-limited number. The planner in Altché, Polack, and de La
Fortelle (arXiv:1804.08290, 2018) enforces the 0.5 μ g bound through a
speed-dependent steering limit. This module uses dry pavement, ``mu = 1``.

Reymond, Kemeny, Droulez, and Berthoz, Human Factors 2001, show that drivers
choose a speed-dependent comfort margin rather than one fixed acceleration.
That comfort curve is not the hard guard here.

The resulting yaw-rate cap is the tighter of the two:

    omega_max(v) = min(v / R_min, a_lat_max / v)

A persistent heading change faster than that is not a feasible passenger-car
turn. These labels are constant-speed polylines, so the ground-truth corner
itself can violate the cap. The penalty and the projection act on the
prediction. They do not rewrite the simulator path.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

G_MPS2 = 9.80665
MU_DRY = 1.0
WHEELBASE_M = 2.70
MAX_ROAD_WHEEL_STEER_RAD = math.radians(35.0)
R_MIN_M = WHEELBASE_M / math.tan(MAX_ROAD_WHEEL_STEER_RAD)
A_LAT_MAX_MPS2 = 0.5 * MU_DRY * G_MPS2
SPEED_FLOOR_MPS = 0.5
# Heading is measured on a short chord so one spectrogram hop of numerical
# wobble is not treated as a vehicle yaw rate. A tenth of a second is still
# short compared with a driver steering correction.
YAW_WINDOW_S = 0.10
# A chord yaw rate can read slightly above the per-step clip when speed
# changes inside the window. The projector aims inside the published cap
# so the chord measurement still satisfies it.
PROJECTION_MARGIN = 0.75


def _span(dt: float) -> int:
    return max(2, int(round(YAW_WINDOW_S / dt)))


def yaw_rate_limit(speed: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Maximum |yaw rate| in rad/s at each speed sample."""
    if isinstance(speed, np.ndarray):
        v = np.maximum(speed, SPEED_FLOOR_MPS)
        return np.minimum(v / R_MIN_M, A_LAT_MAX_MPS2 / v)
    v = speed.clamp(min=SPEED_FLOOR_MPS)
    return torch.minimum(v / R_MIN_M, A_LAT_MAX_MPS2 / v)


def _unwrap(delta: torch.Tensor) -> torch.Tensor:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def path_rates(
    xy: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smoothed speed, yaw rate, and lateral acceleration for a ``(B, T, 2)`` path.

    Instantaneous heading comes from each displacement. Those yaw samples are
    averaged over ``YAW_WINDOW_S`` so a one-hop wiggle is not a turn, while a
    corner that stays sharp across that window still is.
    """
    step = xy[:, 1:, :] - xy[:, :-1, :]
    speed = step.norm(dim=-1) / dt
    heading = torch.atan2(step[..., 1], step[..., 0])
    if heading.shape[1] < 2:
        empty = xy.new_zeros(xy.shape[0], 0)
        return empty, empty, empty
    yaw = _unwrap(heading[:, 1:] - heading[:, :-1]) / dt
    speed_pair = 0.5 * (speed[:, 1:] + speed[:, :-1])
    span = min(_span(dt), yaw.shape[1])
    yaw_s = F.avg_pool1d(yaw.unsqueeze(1), kernel_size=span, stride=1).squeeze(1)
    speed_s = F.avg_pool1d(speed_pair.unsqueeze(1), kernel_size=span, stride=1).squeeze(1)
    return speed_s, yaw_s, speed_s * yaw_s


def kinematic_penalty(
    xy: torch.Tensor,
    dt: float,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean squared excess of yaw rate over the feasible cap, on valid frames."""
    speed, yaw, _lateral = path_rates(xy, dt)
    limit = yaw_rate_limit(speed)
    moving = speed > SPEED_FLOOR_MPS
    excess = (F.relu(yaw.abs() - limit) / limit.clamp(min=1e-3)).clamp(max=5.0)
    # The first yaw sample sits ``span`` frames after the start. A sample is
    # valid when that whole chord lies inside the clip.
    span = _span(dt)
    time = excess.shape[1]
    frame = torch.arange(time, device=xy.device)
    valid = frame[None, :] < (lengths[:, None] - span - 1).clamp(min=0)
    mask = (valid & moving).float()
    denom = mask.sum().clamp(min=1.0)
    return (excess.square() * mask).sum() / denom


def violation_fraction(xy: np.ndarray, dt: float) -> float:
    """Fraction of moving steps whose yaw rate exceeds the cap. ``xy`` is ``(T, 2)``."""
    path = torch.as_tensor(xy, dtype=torch.float32)[None, :, :]
    speed, yaw, _lateral = path_rates(path, dt)
    limit = yaw_rate_limit(speed)
    moving = speed > SPEED_FLOOR_MPS
    if int(moving.sum()) == 0:
        return 0.0
    over = (yaw.abs() > limit + 1e-4) & moving
    return float(over.sum() / moving.sum())


def project_path(xy: np.ndarray, dt: float) -> np.ndarray:
    """Rebuild a path so successive chords respect the yaw-rate cap.

    Each accepted step keeps its length, so speed is unchanged. The heading
    change is clipped with the same midpoint speed the penalty uses. A step
    below the speed floor does not get to introduce a new heading.
    """
    path = np.asarray(xy, dtype=np.float64)
    out = np.empty_like(path)
    out[0] = path[0]
    heading = None
    previous_speed = 0.0
    for index in range(1, len(path)):
        step = path[index] - path[index - 1]
        distance = float(np.hypot(step[0], step[1]))
        speed = distance / dt
        if distance <= 0.0 or heading is None:
            if distance > 0.0:
                heading = float(np.arctan2(step[1], step[0]))
            out[index] = out[index - 1] + step
            previous_speed = speed
            continue
        desired = float(np.arctan2(step[1], step[0]))
        delta = (desired - heading + math.pi) % (2.0 * math.pi) - math.pi
        speed_mid = max(0.5 * (previous_speed + speed), SPEED_FLOOR_MPS)
        cap = PROJECTION_MARGIN * float(np.asarray(yaw_rate_limit(np.array([speed_mid])))[0]) * dt
        heading = heading + float(np.clip(delta, -cap, cap))
        out[index] = out[index - 1] + distance * np.array([math.cos(heading), math.sin(heading)])
        previous_speed = speed
    return out.astype(np.float32)
