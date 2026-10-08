"""Phase 0: a straight pass collapses, a turn does not, position comes from velocity."""

import numpy as np

from position_ctc.alphabet import N_CLASS, quantize_velocity
from position_ctc.audio import complex_stft
from position_ctc.clock import DT, HOP, SR, frame_times
from position_ctc.tokens import motion_tokens, string_length_ok


def _straight(speed_mps: float, n: int = 400) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) * DT
    x = speed_mps * t
    state = np.column_stack(
        [
            x,
            np.full(n, speed_mps),
            np.full(n, 8.0),
            np.zeros(n),
        ]
    )
    return state


def test_clock_is_100_hz():
    times = frame_times(SR)
    assert times.size == 1 + SR // HOP
    assert abs(times[1] - times[0] - DT) < 1e-12
    assert abs(1.0 / DT - 100.0) < 0.3


def test_straight_pass_is_one_velocity_pair():
    tokens = motion_tokens(_straight(20.0), DT)
    assert tokens.tokens == ["VX+20", "VY+0"]
    assert tokens.change_frames == []
    assert string_length_ok(tokens.n_tokens)
    err = np.sqrt(np.mean((tokens.xy_from_bins[:, 0] - _straight(20.0)[:, 0]) ** 2))
    assert err < 1e-6


def test_off_bin_speed_stays_one_pair_and_drifts():
    speed = 21.0
    state = _straight(speed)
    tokens = motion_tokens(state, DT)
    assert tokens.tokens == [f"VX{quantize_velocity(speed):+.0f}", "VY+0"]
    assert tokens.change_frames == []
    drift = abs(tokens.xy_from_bins[-1, 0] - state[-1, 0])
    assert drift > 0.5


def test_turn_emits_a_short_acceleration_string():
    n = 300
    t = np.arange(n, dtype=np.float64) * DT
    speed = 10.0
    heading = np.linspace(0.0, np.pi / 2, n)
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    x = np.cumsum(vx) * DT
    y = np.cumsum(vy) * DT
    state = np.column_stack([x, vx, y, vy])
    tokens = motion_tokens(state, DT)
    assert tokens.change_frames
    assert any(tok.startswith("AX") or tok.startswith("AY") for tok in tokens.tokens)
    assert string_length_ok(tokens.n_tokens)
    assert tokens.n_tokens < n


def test_slow_ramp_acceleration_is_the_interval_not_a_spike():
    n = 200
    t = np.arange(n, dtype=np.float64) * DT
    vx = np.interp(t, [0.0, t[-1]], [0.0, 4.0])
    state = np.column_stack([vx * 0.0, vx, np.zeros(n), np.zeros(n)])
    tokens = motion_tokens(state, DT)
    acc = [float(tok[2:]) for tok in tokens.tokens if tok.startswith("AX")]
    assert acc
    assert max(abs(a) for a in acc) < 8.0


def test_stft_matches_the_clock():
    y = np.cos(2 * np.pi * 440.0 * np.arange(SR) / SR)
    spec = complex_stft(y, sr=SR)
    assert spec["times_s"].size == frame_times(SR).size
    assert spec["magnitude"].shape[1] == spec["times_s"].size
    assert spec["phase_sin"].shape == spec["phase_cos"].shape == spec["magnitude"].shape
    assert np.all(np.abs(spec["phase_sin"]) <= 1.0 + 1e-5)
    assert N_CLASS > 10
