"""Trapezoid integration from a known start reconstructs a constant-speed segment."""

import numpy as np
import torch

from position_ctc.velocity_track import integrate_trapezoid, velocity_objective


def test_trapezoid_matches_a_straight_segment():
    t = torch.linspace(0.0, 1.0, 21)
    times = t[None, :]
    vel = torch.zeros(1, 21, 2)
    vel[..., 0] = 10.0
    origin = torch.tensor([[-5.0, 3.0]])
    xy = integrate_trapezoid(vel, times, origin)
    assert torch.allclose(xy[0, 0], origin[0])
    assert torch.allclose(xy[0, -1], torch.tensor([5.0, 3.0]), atol=1e-5)


def test_objective_is_zero_on_the_labels():
    n = 40
    t = torch.arange(n, dtype=torch.float32) * 0.01161
    vx = torch.full((n,), 12.0)
    vy = torch.zeros(n)
    x = torch.cumsum(torch.cat([torch.zeros(1), 0.5 * (vx[:-1] + vx[1:]) * 0.01161]), dim=0)
    y = torch.full((n,), 8.0)
    state = torch.stack([x, vx, y, vy], dim=-1)[None]
    vel = torch.stack([vx, vy], dim=-1)[None]
    times = t[None]
    lengths = torch.tensor([n])
    loss, parts = velocity_objective(vel, times, state, lengths)
    assert float(loss) < 1e-5
    assert parts["path"] < 1e-5
