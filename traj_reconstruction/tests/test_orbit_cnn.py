"""Complex-STFT 2D CNN orbit model tests (simulated, no metadata inputs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traj_reconstruction.batch import build_tier1_batch
from traj_reconstruction.dataset import Phase1Batch, to_inference_bundle
from traj_reconstruction.flexible import train_orbit_mlp
from traj_reconstruction.orbit_cnn import (
    ARCH_TAG,
    FEAT_F,
    FEAT_T,
    N_CH,
    OrbitCNN,
    _conv2d,
    _conv2d_backward,
    complex_stft_features,
    infer_orbit_cnn,
    train_orbit_cnn,
)
from traj_reconstruction.splits import write_splits


def test_conv2d_weight_grad_matches_finite_diff():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 8, 9))
    w = rng.normal(size=(3, 2, 3, 3)) * 0.05
    b = rng.normal(size=(3,)) * 0.01
    y = _conv2d(x, w, b)
    dout = rng.normal(size=y.shape)
    _dx, dw, db = _conv2d_backward(x, w, dout)
    eps = 1e-6
    w2 = w.copy()
    w2[0, 1, 1, 1] += eps
    y2 = _conv2d(x, w2, b)
    numeric = float(np.sum((y2 - y) * dout) / eps)
    assert abs(numeric - dw[0, 1, 1, 1]) < 2e-4
    b2 = b.copy()
    b2[1] += eps
    yb = _conv2d(x, w, b2)
    numeric_b = float(np.sum((yb - y) * dout) / eps)
    assert abs(numeric_b - db[1]) < 2e-4


def test_complex_stft_features_keep_relative_db():
    t = np.linspace(0.0, 1.0, 22050, endpoint=False)
    audio = np.sin(2 * np.pi * 500.0 * t) * np.linspace(1.0, 0.05, t.size)
    feat = complex_stft_features(audio=audio, sr=22050)
    assert feat.shape == (N_CH, FEAT_F, FEAT_T)
    db = feat[0] * 40.0
    assert float(np.max(db)) <= 1e-5
    assert float(np.min(db)) < -10.0
    assert float(np.std(feat[3])) > 1e-6


def test_cnn_rejects_mlp_checkpoint(tmp_path: Path):
    out = build_tier1_batch(
        tmp_path / "batch",
        n_per_family={"straight": 4, "arc": 3, "s_curve": 3, "u_turn": 2},
        families=("straight", "arc", "s_curve", "u_turn"),
        seed=4,
        resume=False,
    )
    write_splits(out, holdout_families=["u_turn"], seed=0)
    mlp_ckpt = tmp_path / "legacy_mlp.npz"
    train_orbit_mlp(
        out,
        epochs=1,
        lr=5e-3,
        seed=2,
        holdout_families=("u_turn",),
        checkpoint_path=mlp_ckpt,
        resume=False,
    )
    assert mlp_ckpt.is_file()
    assert not OrbitCNN.is_checkpoint(mlp_ckpt)
    cnn_ckpt = tmp_path / "orbit_cnn.npz"
    cnn_ckpt.write_bytes(mlp_ckpt.read_bytes())
    last = cnn_ckpt.with_name(cnn_ckpt.stem + ".last.npz")
    mlp_last = mlp_ckpt.with_name(mlp_ckpt.stem + ".last.npz")
    if mlp_last.is_file():
        last.write_bytes(mlp_last.read_bytes())
    _, report = train_orbit_cnn(
        out,
        epochs=1,
        min_epochs=1,
        patience=5,
        lr=1e-3,
        seed=2,
        holdout_families=("u_turn",),
        checkpoint_path=cnn_ckpt,
        resume=True,
    )
    assert report["resumed_complex_stft"] is False
    assert OrbitCNN.is_checkpoint(cnn_ckpt)
    loaded = np.load(cnn_ckpt, allow_pickle=True)
    assert str(np.asarray(loaded["arch"]).reshape(-1)[0]) == ARCH_TAG


def test_train_cnn_smoke(tmp_path: Path):
    out = build_tier1_batch(
        tmp_path / "batch",
        n_per_family={"straight": 5, "arc": 4, "s_curve": 4, "u_turn": 3},
        families=("straight", "arc", "s_curve", "u_turn"),
        seed=5,
        resume=False,
    )
    write_splits(out, holdout_families=["u_turn"], seed=0)
    ckpt = tmp_path / "orbit_cnn.npz"
    model, report = train_orbit_cnn(
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
    _, report2 = train_orbit_cnn(
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
    assert report2["resumed_complex_stft"] is True
    assert report2["epoch_completed"] == 3
    assert len(report2["history"]) == 3
    assert report["best_val_orbit_rms"] < 120.0
    loaded = OrbitCNN.load(ckpt)
    batch = Phase1Batch.from_dir(out)
    sample = batch.load(0)
    bundle = to_inference_bundle(sample)
    pred = infer_orbit_cnn(loaded, wav_path=bundle.wav_path, stft_db=bundle.stft_db)
    assert pred.shape[1] == 2
    assert pred.shape[0] >= 2
    assert model.n_knots == 64
