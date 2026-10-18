"""Full-model window, clock, and loss checks."""

import numpy as np
import torch

from position_ctc.data import window_bounds
from position_ctc.integrate import integrate_trapezoid
from position_ctc.motion_loss import FullObjective, strict_bin_index
from position_ctc.motion_model import FullMotionNet


def test_trailing_audio_extends_past_the_last_moving_frame():
    times = np.arange(80, dtype=np.float64) * 0.01
    state = np.zeros((80, 4), dtype=np.float32)
    state[10:40, 1] = 12.0
    state[:, 0] = 40.0
    start, stop, end = window_bounds(times, state)
    assert start == 10
    assert stop == 40
    assert end > stop
    delay = 40.0 / 343.0
    assert times[end - 1] + 1e-9 >= times[stop - 1] + delay


def test_frequency_cnn_keeps_every_time_step_and_anchors_position():
    torch.manual_seed(0)
    model = FullMotionNet(dropout=0.0)
    model.eval()
    feat = torch.randn(2, 144, 37)
    origin = torch.tensor([[-20.0, 8.0], [4.0, -3.0]])
    out = model(feat, torch.tensor([37, 20]), origin)
    assert out["vel_x"].shape == (2, 37, 31)
    assert out["ctc_y"].shape == (2, 37, 32)
    direct = origin[:, None, :] + out["disp"] - out["disp"][:, :1, :]
    assert torch.allclose(direct[:, 0], origin, atol=1e-5)


def test_velocity_targets_outside_the_alphabet_raise():
    velocity = torch.zeros(1, 4)
    velocity[0, 1] = 42.0
    valid = torch.tensor([[True, True, False, False]])
    try:
        strict_bin_index(velocity, valid)
    except RuntimeError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("out-of-range target was clipped or accepted")


def test_trapezoid_uses_each_frame_interval():
    xy = integrate_trapezoid(np.array([2.0, 2.0, 2.0]), np.zeros(3), np.array([1.0, 3.0]), x0=0.0, y0=0.0)
    assert abs(xy[1, 0] - 2.0) < 1e-9
    assert abs(xy[2, 0] - 8.0) < 1e-9


def test_vy_blend_and_encoder_gradient():
    torch.manual_seed(0)
    model = FullMotionNet(dropout=0.0)
    feat = torch.randn(2, 144, 24)
    lengths = torch.tensor([24, 16])
    origin = torch.tensor([[-30.0, 10.0], [5.0, 4.0]])
    state = torch.zeros(2, 24, 4)
    state[..., 1] = 14.0
    state[..., 3] = 0.0
    state[0, 12:, 3] = 6.0
    state[..., 0] = origin[:, :1] + torch.arange(24) * 0.01161 * 14.0
    state[..., 2] = origin[:, 1:2]
    times = torch.arange(24, dtype=torch.float32) * 0.01161
    times = times[None, :].expand(2, -1).contiguous()
    out = model(feat, lengths, origin)
    loss, parts = FullObjective()(out, state, times, lengths, aux_scale=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["ctc"] > 0.0
    assert model.cnn[0].weight.grad is not None
    assert float(model.cnn[0].weight.grad.norm()) > 0.0
    assert float(model.rnn.weight_hh_l0.grad.norm()) > 0.0
    assert parts["aux_scale"] == 1.0
