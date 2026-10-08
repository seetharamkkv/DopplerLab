"""Velocity bins, graded transitions, Viterbi, and one training step."""

import numpy as np
import torch

from position_ctc.bins import collapse_runs, trapezoid_velocity_gap, velocity_bin_index
from position_ctc.features import DATA_HOP, DATA_SR
from position_ctc.integrate import integrate_trapezoid
from position_ctc.motion_loss import (
    MotionObjective,
    dominant_balanced_nll,
    expected_distance,
    integrate_expected,
    reliable_velocity_mask,
    zero_bin_balanced_nll,
)
from position_ctc.motion_model import MotionNet
from position_ctc.plots import trajectory_figure
from position_ctc.transitions import a_max_for_jump, transition_matrix
from position_ctc.viterbi import constrained_greedy, viterbi

DT = DATA_HOP / float(DATA_SR)


def test_collapse_and_bin_index_match_the_alphabet():
    assert collapse_runs(np.array([4, 4, 4, 3, 3, 1, 1])).tolist() == [4, 3, 1]
    assert velocity_bin_index(np.array([0.0, 2.4, -30.0, 80.0])).tolist() == [15, 16, 0, 30]
    state = np.zeros((6, 4), dtype=np.float64)
    state[:, 1] = 10.0
    state[:, 0] = 10.0 * np.arange(6) * DT
    assert trapezoid_velocity_gap(state, DT) < 1e-6


def test_transition_costs_match_the_three_bin_table():
    psi = transition_matrix(DT, a_max_for_jump(6.0, DT), lam=1.0, scale_mps=2.0)
    stay = 15  # 0 m/s
    assert psi[stay, stay] == 0.0
    assert abs(psi[stay, stay + 1] + 1.0) < 1e-6
    assert abs(psi[stay, stay + 2] + 4.0) < 1e-6
    assert abs(psi[stay, stay + 3] + 9.0) < 1e-6
    assert np.isneginf(psi[stay, stay + 4])


def test_viterbi_keeps_a_path_greedy_cannot_reach():
    psi = transition_matrix(1.0, a_max_mps2=2.0, lam=1.0, scale_mps=2.0, centers=np.array([0.0, 2.0, 4.0]))
    logp = np.array(
        [
            [0.0, -5.0, -0.2],
            [-10.0, -10.0, 0.0],
        ]
    )
    assert viterbi(logp, psi).tolist() == [2, 2]
    greedy = constrained_greedy(logp, psi)
    assert greedy[0] == 0
    assert greedy[1] != 2


def test_viterbi_rejects_a_forbidden_spike():
    centers = np.arange(-4.0, 5.0, 2.0)
    psi = transition_matrix(1.0, a_max_mps2=2.0, lam=1.0, scale_mps=2.0, centers=centers)
    logp = np.full((4, centers.size), -8.0)
    logp[:, 2] = 0.0
    logp[2, 2] = -0.5
    logp[2, 4] = 0.0
    assert int(np.argmax(logp[2])) == 4
    assert viterbi(logp, psi).tolist() == [2, 2, 2, 2]


def test_trapezoid_uses_both_endpoint_velocities():
    xy = integrate_trapezoid(np.array([0.0, 10.0]), np.array([0.0, 0.0]), 1.0, x0=1.0, y0=2.0)
    assert abs(xy[1, 0] - 6.0) < 1e-9
    assert abs(xy[1, 1] - 2.0) < 1e-9


def test_a_corner_spike_is_left_out_of_the_velocity_mask():
    state = torch.zeros(1, 5, 4)
    state[0, :, 1] = 10.0
    state[0, 2, 1] = 30.0
    state[0, :, 0] = 10.0 * torch.arange(5) * DT
    valid = torch.ones(1, 5, dtype=torch.bool)
    mask = reliable_velocity_mask(state, valid, DT, jump_mps=6.0)
    assert mask[0].tolist() == [True, False, False, False, True]


def test_anchored_integral_starts_at_the_supplied_position_and_penalizes_a_flat_turn():
    time = 20
    dt = 0.1
    steps = torch.arange(time, dtype=torch.float32)
    xy = torch.stack((steps * 1.0 - 30.0, torch.linspace(8.0, 12.0, time)), dim=-1)[None]
    valid = torch.ones(1, time, dtype=torch.bool)
    vel = torch.zeros(1, time, 2)
    vel[..., 0] = 10.0
    pred = integrate_expected(vel, xy[:, :1], valid, dt)
    assert torch.allclose(pred[:, 0], xy[:, 0])
    objective = MotionObjective()
    flat = objective._integrated_path(vel, xy, valid, dt)
    matched = vel.clone()
    matched[..., 1] = (12.0 - 8.0) / (dt * (time - 1))
    good = objective._integrated_path(matched, xy, valid, dt)
    assert float(flat) > float(good)
    assert float(flat) > 0.5


def test_far_bin_costs_more_than_a_neighbor_at_the_same_correct_probability():
    centers = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0])
    truth = torch.tensor([4.0])
    near = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]])
    far = torch.tensor([[0.5, 0.0, 0.5, 0.0, 0.0]])
    near_cost = expected_distance(near, truth, centers, scale=2.0)
    far_cost = expected_distance(far, truth, centers, scale=2.0)
    assert float(far_cost) > float(near_cost)


def test_model_step_is_finite_and_conditions_on_the_initial_position():
    torch.manual_seed(0)
    model = MotionNet()
    model.eval()
    feat = torch.randn(2, 144, 40)
    feat[1, :, 16:] = 0
    lengths = torch.tensor([40, 16])
    origin = torch.tensor([[-40.0, 12.0], [10.0, -6.0]])
    out = model(feat, lengths, origin)
    assert out["vel_x"].shape == (2, 40, 31)
    assert out["ctc_x"].shape == (2, 40, 32)
    assert int(out["enc_lengths"][0]) == 10
    assert "polar" not in out and "pos" not in out
    again = model(feat, lengths, origin)
    assert torch.allclose(out["vel_y"], again["vel_y"])
    model.origin.weight.data.fill_(0.2)
    base = model(feat, lengths, origin)
    shifted = model(feat, lengths, origin + torch.tensor([30.0, -20.0]))
    assert not torch.allclose(base["vel_x"], shifted["vel_x"])
    time = torch.arange(40, dtype=torch.float32)
    state = torch.zeros(2, 40, 4)
    state[..., 1] = 10.0
    state[..., 0] = 10.0 * time * DT
    state[:, 0, 0] = origin[:, 0]
    state[..., 2] = 4.0
    state[:, 0, 2] = origin[:, 1]
    model.train()
    live = model(feat, lengths, origin)
    loss, parts = MotionObjective()(live, state, lengths, DT)
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["ctc"] >= 0.0
    assert parts["path"] >= 0.0
    assert model.conv[0].weight.grad is not None
    assert model.origin.weight.grad is not None
    assert torch.isfinite(model.conv[0].weight.grad).all()


def test_viterbi_integral_starts_at_the_anchor_and_matches_its_steps():
    from position_ctc.reconstruct import reconstruct
    from position_ctc.transitions import a_max_for_jump, transition_matrix

    frames, bins = 8, 31
    logp = np.full((frames, bins), -8.0)
    logp[:, 20] = 0.0
    logp[4:, 20] = -0.5
    logp[4:, 21] = 0.0
    dt = DT
    psi = transition_matrix(dt, a_max_for_jump(6.0, dt), lam=0.25)
    traj = reconstruct(logp, logp, dt, x0=3.0, y0=-4.0, psi=psi)
    assert abs(traj.xy[0, 0] - 3.0) < 1e-12
    assert abs(traj.xy[0, 1] + 4.0) < 1e-12
    assert abs(traj.xy_argmax[0, 0] - 3.0) < 1e-12
    assert abs(traj.xy_argmax[0, 1] + 4.0) < 1e-12
    step_x = 0.5 * (traj.vx[:-1] + traj.vx[1:]) * dt
    step_y = 0.5 * (traj.vy[:-1] + traj.vy[1:]) * dt
    assert np.max(np.abs(np.diff(traj.xy[:, 0]) - step_x)) < 1e-9
    assert np.max(np.abs(np.diff(traj.xy[:, 1]) - step_y)) < 1e-9
    jumped = np.abs(np.diff(np.rint((traj.vx + 30.0) / 2.0)))
    assert np.max(jumped) <= 3


def test_ctc_blank_is_not_a_physical_velocity_bin():
    logits = torch.zeros(1, 8, 32)
    velocity = torch.full((1, 8), -30.0)
    lengths = torch.tensor([8])
    reliable = torch.ones(1, 8, dtype=torch.bool)
    loss = MotionObjective()._ctc(logits, velocity, lengths, reliable)
    assert torch.isfinite(loss)
    assert logits.shape[-1] == 32


def test_integrated_path_loss_penalizes_a_flat_velocity():
    time = 20
    dt = 0.1
    steps = torch.arange(time, dtype=torch.float32)
    xy = torch.stack((steps * 1.0, torch.linspace(0.0, 4.0, time)), dim=-1)[None]
    vel = torch.zeros(1, time, 2)
    vel[..., 0] = 10.0
    valid = torch.ones(1, time, dtype=torch.bool)
    objective = MotionObjective()
    flat = objective._integrated_path(vel, xy, valid, dt)
    matched = vel.clone()
    matched[..., 1] = 4.0 / (dt * (time - 1))
    good = objective._integrated_path(matched, xy, valid, dt)
    assert float(flat) > float(good)
    assert float(flat) > 0.5


def test_zero_bin_balance_splits_vy_between_zero_and_nonzero():
    nll = torch.zeros(1, 10)
    nll[0, -1] = 10.0
    bins = torch.zeros(1, 10, dtype=torch.long)
    bins[0, -1] = 4
    valid = torch.ones(1, 10, dtype=torch.bool)
    valid[0, 0] = False
    balanced = zero_bin_balanced_nll(nll, bins, valid, zero_index=0)
    assert abs(float(balanced) - 5.0) < 1e-5
    only_zero = zero_bin_balanced_nll(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.long), valid[:, :4], 0)
    assert abs(float(only_zero)) < 1e-5


def test_dominant_balanced_loss_gives_the_minority_half_the_weight():
    nll = torch.zeros(1, 10)
    nll[0, -1] = 10.0
    bins = torch.zeros(1, 10, dtype=torch.long)
    bins[0, -1] = 4
    valid = torch.ones(1, 10, dtype=torch.bool)
    balanced = dominant_balanced_nll(nll, bins, valid)
    ordinary = nll.mean()
    assert abs(float(balanced) - 5.0) < 1e-5
    assert float(balanced) > float(ordinary)
    only = dominant_balanced_nll(torch.zeros_like(nll), torch.zeros_like(bins), valid)
    assert abs(float(only)) < 1e-5


def test_disabling_temporal_pool_keeps_one_feature_per_audio_frame():
    torch.manual_seed(0)
    model = MotionNet(dropout=0.0, temporal_pool=False)
    feat = torch.randn(1, 144, 40)
    lengths = torch.tensor([40])
    origin = torch.zeros(1, 2)
    out = model(feat, lengths, origin)
    assert int(out["enc_lengths"][0]) == 40
    assert out["vel_y"].shape == (1, 40, 31)


def test_trajectory_figure_writes_four_panels(tmp_path):
    t = np.linspace(0.0, 1.0, 20)
    gt = np.stack([t, np.full_like(t, 3.0)], axis=1)
    path = tmp_path / "trajectory.png"
    trajectory_figure(
        t,
        np.full_like(t, 1.0),
        np.zeros_like(t),
        gt,
        np.full_like(t, 1.2),
        np.zeros_like(t),
        gt + 0.1,
        path,
        title="synthetic",
        vx_argmax=np.full_like(t, 2.0),
        vy_argmax=np.zeros_like(t),
        xy_argmax=gt + 0.2,
    )
    assert path.is_file()
    assert path.stat().st_size > 1000
