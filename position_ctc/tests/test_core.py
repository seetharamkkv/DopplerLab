"""Core velocity model: neighborhood targets, CTC, and a temporal CNN."""

import math

import torch

from position_ctc.alphabet import VEL_STEP_MPS
from position_ctc.motion_loss import CoreObjective, CoreWeights, neighborhood_log_target
from position_ctc.motion_model import CoreMotionNet


def test_temporal_cnn_restores_every_frame_and_has_no_auxiliary_heads():
    torch.manual_seed(0)
    model = CoreMotionNet(dropout=0.0)
    names = [name for name, _ in model.named_parameters()]
    assert not any(name.startswith(("pos.", "acc.", "origin.")) for name in names)
    feat = torch.randn(2, 144, 37)
    out = model(feat, torch.tensor([37, 20]))
    assert out["vel_x"].shape == (2, 37, 31)
    assert out["vel_y"].shape == (2, 37, 31)
    assert out["ctc_x"].shape == (2, 37, 32)
    assert out["ctc_y"].shape == (2, 37, 32)
    assert set(out) == {"vel_x", "vel_y", "ctc_x", "ctc_y"}


def test_neighborhood_puts_mass_on_nearby_bins():
    centers = torch.arange(31) * 2.0 - 30.0
    log_q = neighborhood_log_target(torch.tensor([0.0]), centers, sigma=2.0)
    q = log_q.exp()
    assert torch.allclose(q.sum(), torch.tensor(1.0), atol=1e-5)
    peak = int(torch.argmax(q))
    assert peak == 15
    ratio = float(q[0, peak + 1] / q[0, peak])
    assert math.isclose(ratio, math.exp(-0.5), rel_tol=1e-5)
    assert float(q[0, peak + 3]) < float(q[0, peak + 1])


def test_core_loss_ignores_the_tail_and_reaches_the_encoder():
    torch.manual_seed(0)
    model = CoreMotionNet(dropout=0.0)
    objective = CoreObjective(CoreWeights(sigma_mps=2.0, ctc=0.25))
    feat = torch.randn(2, 144, 40)
    state = torch.zeros(2, 40, 4)
    state[..., 1] = 14.0
    state[0, 10:20, 3] = 4.0
    state[1, :, 3] = -2.0
    lengths = torch.tensor([24, 24])
    pred = model(feat, torch.tensor([40, 32]))
    loss, parts = objective(pred, state, lengths)
    assert torch.isfinite(loss)
    assert parts["ctc"] > 0.0
    loss.backward()
    conv = torch.nn.utils.clip_grad_norm_(model.conv.parameters(), 1e9)
    rnn = torch.nn.utils.clip_grad_norm_(model.rnn.parameters(), 1e9)
    assert float(conv) > 0.0
    assert float(rnn) > 0.0
    # A change only in the unsupervised tail does not change the target bins.
    state_tail = state.clone()
    state_tail[:, 24:, 1] = 30.0
    pred2 = {key: value.detach() for key, value in pred.items()}
    _, parts_tail = objective(pred2, state_tail, lengths)
    assert parts_tail["vx"] == parts["vx"]


def test_velocity_outside_the_alphabet_raises():
    model = CoreMotionNet(dropout=0.0)
    objective = CoreObjective()
    feat = torch.randn(1, 144, 16)
    state = torch.zeros(1, 16, 4)
    state[..., 1] = 42.0
    pred = model(feat, torch.tensor([16]))
    try:
        objective(pred, state, torch.tensor([16]))
    except RuntimeError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("out-of-range velocity was accepted")


def test_one_bin_offset_is_cheaper_than_a_distant_bin():
    centers = torch.arange(31) * VEL_STEP_MPS - 30.0
    objective = CoreObjective(CoreWeights(sigma_mps=2.0, ctc=0.0))
    state = torch.zeros(1, 4, 4)
    logits_near = torch.full((1, 4, 31), -20.0)
    logits_far = logits_near.clone()
    logits_near[..., 16] = 20.0
    logits_far[..., 0] = 20.0
    empty = torch.zeros(1, 4, 32)
    near, _ = objective({"vel_x": logits_near, "vel_y": logits_near, "ctc_x": empty, "ctc_y": empty}, state, torch.tensor([4]))
    far, _ = objective({"vel_x": logits_far, "vel_y": logits_far, "ctc_x": empty, "ctc_y": empty}, state, torch.tensor([4]))
    assert float(near) < float(far)
    assert centers[15].item() == 0.0
