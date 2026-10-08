from ctc_exp.tokens import collapse_repeats, has_apr_cpa_rec, token_string
import numpy as np


def test_collapse_repeats():
    assert collapse_repeats(["A", "A", "B", "B", "B"]) == ["A", "B"]


def test_straight_pass_has_apr_cpa_rec():
    times = np.linspace(0, 10, 80)
    toks = token_string(speed_mps=20.0, cpa_s=5.0, times_s=times, include_range=False)
    assert has_apr_cpa_rec(toks)
    assert "V20" in toks
    assert len(toks) < 20


def test_range_changes_along_pass():
    times = np.linspace(0, 10, 80)
    rng = np.abs(times - 5.0) * 15.0 + 0.5
    toks = token_string(speed_mps=20.0, cpa_s=5.0, times_s=times, range_m=rng, include_range=True)
    rs = [t for t in toks if t.startswith("R") and t[1:].isdigit()]
    assert len(set(rs)) >= 2


def test_centered_loss_ignores_global_offset():
    import torch
    from ctc_exp.loss import centered_traj_huber

    gt = torch.tensor([[[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]])
    pred = gt + 50.0
    mask = torch.ones(1, 3)
    loss = float(centered_traj_huber(pred, gt, mask))
    assert loss < 1e-5


def test_orbit_rms_rotation_invariant():
    import numpy as np
    from ctc_exp.track_geom import orbit_rms

    t = np.linspace(0, np.pi, 20)
    gt = np.stack([np.cos(t) * 10, np.sin(t) * 10 + 15], axis=1)
    ang = 0.7
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pred = gt @ rot.T
    assert orbit_rms(pred, gt) < 0.05


def test_speed_bin_index_covers_alphabet():
    from ctc_exp.alphabet import N_SPEED, speed_bin_index

    assert speed_bin_index(20.0) != speed_bin_index(40.0)
    assert 0 <= speed_bin_index(12.0) < N_SPEED
