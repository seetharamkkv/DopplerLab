"""Phase 3a parametric orbit fit tests (simulated Tier-1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traj_reconstruction.kinematics import canonical_state_frames, interpolate_state, stft_frame_times, stft_n_frames
from traj_reconstruction.orbit import orbit_align, xy_from_state
from traj_reconstruction.parametric import fit_orbit_from_audio, fit_parametric_orbit, plot_fit_overlay
from traj_reconstruction.path_families import make_arc, make_straight
from traj_reconstruction.synthesize import synthesize_tone_on_path
from traj_reconstruction.frontend import extract_ridges


def _gt_xy_helper(synth: dict, speed: float) -> tuple[np.ndarray, float, float]:
    traj = synth["trajectory"]
    n = len(synth["audio"])
    n_frames = stft_n_frames(n)
    times = stft_frame_times(n_frames)
    state = interpolate_state(traj["t"], traj["state"], times)
    can, _ = canonical_state_frames(state)
    return xy_from_state(can), float(synth["cpa_distance_m"]), float(speed)


def test_fit_straight_frequency_only():
    xy = make_straight(cpa_distance_m=15.0, half_length_m=70.0, heading_rad=0.4)
    speed = 22.0
    synth = synthesize_tone_on_path(xy, speed_mps=speed, f0_hz=500.0, pad_s=0.25)
    gt_xy, gt_cpa, _ = _gt_xy_helper(synth, speed)
    fit = fit_orbit_from_audio(
        audio=synth["audio"],
        sr=synth["sr"],
        family="straight",
        use_amplitude=True,
        gt_xy=gt_xy,
        gt_speed_mps=speed,
        gt_cpa_distance_m=gt_cpa,
    )
    assert fit.success
    assert fit.orbit is not None
    assert fit.orbit.rms < 5.0
    assert fit.metrics is not None
    assert fit.metrics["speed_rel_err"] < 0.25
    assert fit.metrics["cpa_rel_err"] < 0.35


def test_fit_arc_frequency_only():
    xy = make_arc(cpa_distance_m=18.0, radius_m=55.0, sweep_rad=np.deg2rad(70.0), heading_rad=0.2)
    speed = 18.0
    synth = synthesize_tone_on_path(xy, speed_mps=speed, f0_hz=450.0, pad_s=0.25)
    gt_xy, gt_cpa, _ = _gt_xy_helper(synth, speed)
    fit = fit_orbit_from_audio(
        audio=synth["audio"],
        sr=synth["sr"],
        family="arc",
        use_amplitude=True,
        gt_xy=gt_xy,
        gt_speed_mps=speed,
        gt_cpa_distance_m=gt_cpa,
    )
    assert fit.success
    assert fit.orbit is not None
    assert fit.orbit.rms < 10.0
    assert fit.metrics is not None
    assert fit.metrics["speed_rel_err"] < 0.4


def test_orbit_score_rotation_invariant():
    xy = make_straight(cpa_distance_m=12.0, half_length_m=50.0, heading_rad=0.0)
    synth = synthesize_tone_on_path(xy, speed_mps=20.0, f0_hz=500.0)
    gt_xy, gt_cpa, _ = _gt_xy_helper(synth, 20.0)
    fit = fit_orbit_from_audio(
        audio=synth["audio"],
        sr=synth["sr"],
        family="straight",
        gt_xy=gt_xy,
        gt_speed_mps=20.0,
        gt_cpa_distance_m=gt_cpa,
    )
    # Rotate GT; orbit RMS must stay the same.
    ang = np.deg2rad(55.0)
    c, s = np.cos(ang), np.sin(ang)
    gt_rot = np.column_stack(
        [c * gt_xy[:, 0] - s * gt_xy[:, 1], s * gt_xy[:, 0] + c * gt_xy[:, 1]]
    )
    n = min(len(fit.xy_pred), len(gt_rot))
    r0 = orbit_align(fit.xy_pred[:n], gt_xy[:n]).rms
    r1 = orbit_align(fit.xy_pred[:n], gt_rot[:n]).rms
    assert abs(r0 - r1) < 1e-6


def test_plot_fit(tmp_path: Path):
    xy = make_straight(cpa_distance_m=10.0, half_length_m=40.0)
    synth = synthesize_tone_on_path(xy, speed_mps=15.0, f0_hz=500.0)
    gt_xy, _, _ = _gt_xy_helper(synth, 15.0)
    feats = extract_ridges(audio=synth["audio"], sr=synth["sr"], n_harmonics=1)
    fit = fit_parametric_orbit(feats, family="straight", gt_xy=gt_xy)
    out = plot_fit_overlay(fit, tmp_path / "fit.png", gt_xy=gt_xy)
    assert out.is_file()
