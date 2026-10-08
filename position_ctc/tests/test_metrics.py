"""Orbit alignment ignores a rotation about the microphone. Range does not."""

import numpy as np

from position_ctc.metrics import orbit_rms, score_path


def test_rotation_about_the_mic_is_not_position_error():
    t = np.linspace(0, np.pi, 40)
    gt = np.stack([np.cos(t) * 20, np.sin(t) * 20 + 5], axis=1)
    ang = 0.4
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pred = gt @ rot.T
    assert orbit_rms(pred, gt) < 1e-4


def test_range_error_survives_rotation():
    gt = np.array([[0.0, 10.0], [0.0, 20.0]])
    pred = np.array([[10.0, 0.0], [30.0, 0.0]])
    scores = score_path(pred, gt)
    assert scores["range_mae_m"] == 5.0
    assert abs(scores["cpa_abs_m"] - 0.0) < 1e-6
