"""Ridge-track 1D CNN orbit model tests (simulated, no metadata)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traj_reconstruction.batch import build_tier1_batch
from traj_reconstruction.dataset import Phase1Batch, to_inference_bundle
from traj_reconstruction.frontend import extract_ridges
from traj_reconstruction.orbit_cnn import OrbitCNN
from traj_reconstruction.orbit_seq import (
    ARCH_TAG,
    N_CH,
    SEQ_T,
    OrbitSeq1D,
    _conv1d,
    _conv1d_backward,
    infer_orbit_seq,
    ridge_sequence_features,
    train_orbit_seq,
)
from traj_reconstruction.paths import DEFAULT_ORBIT_CNN_BEST
from traj_reconstruction.splits import write_splits


def test_conv1d_weight_grad_matches_finite_diff():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 20))
    w = rng.normal(size=(4, 3, 5)) * 0.05
    b = rng.normal(size=(4,)) * 0.01
    y = _conv1d(x, w, b)
    dout = rng.normal(size=y.shape)
    _dx, dw, db = _conv1d_backward(x, w, dout)
    eps = 1e-6
    w2 = w.copy()
    w2[1, 0, 2] += eps
    y2 = _conv1d(x, w2, b)
    numeric = float(np.sum((y2 - y) * dout) / eps)
    assert abs(numeric - dw[1, 0, 2]) < 2e-4


def test_ridge_sequence_keeps_amplitude_shape():
    t = np.linspace(0.0, 1.2, 22050, endpoint=False)
    env = 0.2 + 0.8 * np.exp(-((t - 0.6) ** 2) / (2 * 0.08**2))
    audio = np.sin(2 * np.pi * 500.0 * t) * env
    ridges = extract_ridges(audio=audio, sr=22050, n_harmonics=1)
    feat = ridge_sequence_features(ridges)
    assert feat.shape == (N_CH, SEQ_T)
    a = feat[2]
    assert float(np.max(a)) > 0.9
    assert float(a[0]) < float(np.max(a))
    assert float(a[-1]) < float(np.max(a))


def test_seq_rejects_cnn_checkpoint():
    if not DEFAULT_ORBIT_CNN_BEST.is_file():
        return
    assert OrbitCNN.is_checkpoint(DEFAULT_ORBIT_CNN_BEST)
    assert not OrbitSeq1D.is_checkpoint(DEFAULT_ORBIT_CNN_BEST)


def test_train_seq_smoke(tmp_path: Path):
    out = build_tier1_batch(
        tmp_path / "batch",
        n_per_family={"straight": 5, "arc": 4, "s_curve": 4, "u_turn": 3},
        families=("straight", "arc", "s_curve", "u_turn"),
        seed=6,
        resume=False,
    )
    write_splits(out, holdout_families=["u_turn"], seed=0)
    ckpt = tmp_path / "orbit_seq.npz"
    _model, report = train_orbit_seq(
        out,
        epochs=2,
        min_epochs=2,
        patience=5,
        lr=1e-3,
        seed=1,
        holdout_families=("u_turn",),
        checkpoint_path=ckpt,
        resume=False,
    )
    assert ckpt.is_file()
    last = ckpt.with_name(ckpt.stem + ".last.npz")
    _, report2 = train_orbit_seq(
        out,
        epochs=3,
        min_epochs=2,
        patience=5,
        lr=1e-3,
        seed=1,
        holdout_families=("u_turn",),
        checkpoint_path=ckpt,
        resume=True,
    )
    assert last.is_file()
    assert report2["resumed_ridge_seq"] is True
    assert report2["epoch_completed"] == 3
    assert len(report2["history"]) == 3
    loaded = np.load(ckpt, allow_pickle=True)
    assert str(np.asarray(loaded["arch"]).reshape(-1)[0]) == ARCH_TAG
    model = OrbitSeq1D.load(ckpt)
    batch = Phase1Batch.from_dir(out)
    sample = batch.load(0)
    bundle = to_inference_bundle(sample)
    pred = infer_orbit_seq(model, stft_db=bundle.stft_db, wav_path=bundle.wav_path)
    assert pred.shape[1] == 2
    assert pred.shape[0] >= 2
    assert report["best_val_orbit_rms"] < 120.0
