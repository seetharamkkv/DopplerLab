"""Phase 3b flexible freehand orbit tests (simulated)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traj_reconstruction.batch import build_tier1_batch
from traj_reconstruction.dataset import Phase1Batch
from traj_reconstruction.flexible import (
    OrbitMLP,
    fit_flexible_from_audio,
    infer_orbit_mlp,
    orbit_mse_loss,
    train_orbit_mlp,
)
from traj_reconstruction.kinematics import (
    canonical_state_frames,
    interpolate_state,
    stft_frame_times,
    stft_n_frames,
)
from traj_reconstruction.orbit import xy_from_state
from traj_reconstruction.parametric import fit_orbit_from_audio
from traj_reconstruction.path_families import make_s_curve, make_straight
from traj_reconstruction.splits import write_splits
from traj_reconstruction.synthesize import synthesize_tone_on_path


def _gt_xy(synth: dict) -> np.ndarray:
    traj = synth["trajectory"]
    n = len(synth["audio"])
    times = stft_frame_times(stft_n_frames(n))
    state = interpolate_state(traj["t"], traj["state"], times)
    can, _ = canonical_state_frames(state)
    return xy_from_state(can)


def test_flexible_beats_parametric_on_s_curve():
    xy = make_s_curve(
        cpa_distance_m=16.0, half_length_m=70.0, amplitude_m=10.0, heading_rad=0.15
    )
    synth = synthesize_tone_on_path(xy, speed_mps=20.0, f0_hz=500.0, pad_s=0.25)
    gt = _gt_xy(synth)
    flex = fit_flexible_from_audio(
        audio=synth["audio"], sr=synth["sr"], gt_xy=gt, n_modes=3
    )
    para = fit_orbit_from_audio(
        audio=synth["audio"],
        sr=synth["sr"],
        family="straight",
        use_amplitude=True,
        gt_xy=gt,
    )
    assert flex.orbit is not None and para.orbit is not None
    # Flexible should recover freehand shape better than straight parametric.
    assert flex.orbit.rms < para.orbit.rms * 0.85 + 1.0
    assert flex.orbit.rms < 25.0


def test_flexible_competitive_on_straight():
    xy = make_straight(cpa_distance_m=14.0, half_length_m=60.0, heading_rad=0.2)
    synth = synthesize_tone_on_path(xy, speed_mps=18.0, f0_hz=480.0, pad_s=0.25)
    gt = _gt_xy(synth)
    flex = fit_flexible_from_audio(audio=synth["audio"], sr=synth["sr"], gt_xy=gt)
    para = fit_orbit_from_audio(
        audio=synth["audio"], sr=synth["sr"], family="straight", gt_xy=gt
    )
    assert flex.orbit is not None and para.orbit is not None
    # Should not be dramatically worse than 3a on straight.
    assert flex.orbit.rms < max(para.orbit.rms * 2.5, 8.0)


def test_train_mlp_smoke(tmp_path: Path):
    out = build_tier1_batch(
        tmp_path / "batch",
        n_per_family={"straight": 6, "arc": 4, "s_curve": 4, "u_turn": 3},
        families=("straight", "arc", "s_curve", "u_turn"),
        seed=3,
        resume=False,
    )
    write_splits(out, holdout_families=["u_turn"], seed=0)
    ckpt = tmp_path / "orbit_mlp.npz"
    model, report = train_orbit_mlp(
        out,
        epochs=12,
        lr=5e-3,
        seed=1,
        holdout_families=("u_turn",),
        checkpoint_path=ckpt,
    )
    assert ckpt.is_file()
    assert report["best_val_orbit_rms"] < 80.0
    loaded = OrbitMLP.load(ckpt)
    batch = Phase1Batch.from_dir(out)
    sample = batch.load(0)
    pred = infer_orbit_mlp(loaded, stft_db=sample.stft_db)
    assert pred.shape[0] == sample.n_frames and pred.shape[1] == 2


def test_gauge_invariance_of_orbit_loss():
    xy = make_straight(cpa_distance_m=10.0, half_length_m=40.0)
    synth = synthesize_tone_on_path(xy, speed_mps=15.0, f0_hz=500.0)
    gt = _gt_xy(synth)
    pred = gt + np.random.default_rng(0).normal(0, 0.5, size=gt.shape)
    ang = np.deg2rad(40.0)
    c, s = np.cos(ang), np.sin(ang)
    gt_rot = np.column_stack([c * gt[:, 0] - s * gt[:, 1], s * gt[:, 0] + c * gt[:, 1]])
    assert abs(orbit_mse_loss(pred, gt) - orbit_mse_loss(pred, gt_rot)) < 1e-6
