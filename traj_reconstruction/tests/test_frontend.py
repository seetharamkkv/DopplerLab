"""Tests for audio-only ridge / envelope front end (simulated Tier-1)."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from traj_reconstruction.frontend import extract_ridges, plot_ridge_overlay
from traj_reconstruction.kinematics import compute_stft_db
from traj_reconstruction.path_families import make_straight
from traj_reconstruction.synthesize import synthesize_tone_on_path


def test_extract_signature_has_no_gt_params():
    sig = inspect.signature(extract_ridges)
    forbidden = {
        "state",
        "state_frames",
        "canonical",
        "polyline",
        "cpa",
        "speed_mps",
        "metadata",
        "path_xy",
    }
    assert forbidden.isdisjoint(sig.parameters.keys())


def test_tier1_frequency_track_matches_doppler():
    xy = make_straight(cpa_distance_m=15.0, half_length_m=60.0, heading_rad=0.0)
    synth = synthesize_tone_on_path(xy, speed_mps=25.0, f0_hz=500.0, pad_s=0.2)
    audio = synth["audio"]
    sr = synth["sr"]
    feats = extract_ridges(audio=audio, sr=sr, n_harmonics=1, f_min_hz=200.0, f_max_hz=900.0)

    # Interpolate true f_obs onto STFT frame times.
    t_audio = synth["trajectory"]["t"]
    f_true = np.interp(feats.frame_times, t_audio, synth["f_obs_hz_true"])
    valid = np.isfinite(f_true) & (feats.quality > 0.4)
    # Ignore padded silence at the very ends.
    pad = synth["trajectory"]["pad_s"][0]
    travel = synth["trajectory"]["travel_s"][0]
    in_motion = (feats.frame_times > pad) & (feats.frame_times < pad + travel)
    mask = valid & in_motion
    assert mask.sum() > 10
    err = np.abs(feats.f_obs_hz[mask] - f_true[mask])
    # Bin width ~ sr/n_fft ≈ 10.8 Hz; allow a few bins.
    assert float(np.median(err)) < 25.0
    assert float(np.mean(feats.quality[mask])) > 0.85


def test_envelope_peaks_near_cpa():
    xy = make_straight(cpa_distance_m=12.0, half_length_m=50.0, heading_rad=0.1)
    synth = synthesize_tone_on_path(xy, speed_mps=20.0, f0_hz=500.0, pad_s=0.2)
    feats = extract_ridges(audio=synth["audio"], sr=synth["sr"], n_harmonics=1)
    t_peak = float(feats.frame_times[int(np.argmax(feats.A_env))])
    # Retarded CPA ≈ geometric CPA + h/c (~35 ms here); allow 0.35 s slack.
    assert abs(t_peak - float(synth["cpa_time_sec"])) < 0.35


def test_stft_path_matches_audio_path():
    xy = make_straight(cpa_distance_m=10.0, half_length_m=40.0)
    synth = synthesize_tone_on_path(xy, speed_mps=18.0, f0_hz=400.0)
    stft = compute_stft_db(synth["audio"], sr=synth["sr"])
    a = extract_ridges(audio=synth["audio"], sr=synth["sr"], n_harmonics=1)
    b = extract_ridges(stft_db=stft, sr=synth["sr"], n_harmonics=1)
    assert a.f_obs_hz.shape == b.f_obs_hz.shape
    assert float(np.median(np.abs(a.f_obs_hz - b.f_obs_hz))) < 15.0


def test_plot_overlay(tmp_path: Path):
    xy = make_straight(cpa_distance_m=10.0, half_length_m=30.0)
    synth = synthesize_tone_on_path(xy, speed_mps=15.0, f0_hz=500.0)
    feats = extract_ridges(audio=synth["audio"], sr=synth["sr"], n_harmonics=1)
    out = plot_ridge_overlay(feats, tmp_path / "ridge.png")
    assert out.is_file() and out.stat().st_size > 1000
