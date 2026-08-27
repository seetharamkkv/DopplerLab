"""Dataset loader tests with a synthetic on-disk Phase 1 mini package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from traj_reconstruction.dataset import (
    DatasetError,
    load_phase1_sample,
    to_inference_bundle,
)
from traj_reconstruction.orbit import orbit_align, xy_from_state


def _write_mini_phase1(root: Path, *, t: int = 32, f: int = 17) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "spectrograms").mkdir()
    (root / "metadata").mkdir()
    times = np.arange(t, dtype=np.float64) * (512 / 22050)
    # Straight flyby state.
    v, h, t_cpa = 15.0, 10.0, float(times[t // 2])
    x = v * (times - t_cpa)
    y = np.full(t, h)
    vx = np.full(t, v)
    vy = np.zeros(t)
    state = np.column_stack([x, vx, y, vy]).astype(np.float32)
    # Canonical: same geometry already nearly CPA-on-+y for h>0, x crossing 0.
    stft = np.random.randn(f, t).astype(np.float32)
    np.save(root / "spectrograms" / "stft.npy", stft)
    np.save(root / "metadata" / "state_frames.npy", state)
    np.save(root / "metadata" / "canonical_state_frames.npy", state)
    np.save(root / "metadata" / "frame_times.npy", times.astype(np.float32))
    (root / "metadata" / "phase1_schema.json").write_text(
        json.dumps({"path_type": "free_path_2d", "pipeline": "path2d"})
    )
    # Empty wav marker not required for loader.
    return root


def test_load_phase1_sample(tmp_path: Path):
    root = _write_mini_phase1(tmp_path / "phase1")
    sample = load_phase1_sample(root)
    assert sample.n_frames == 32
    assert sample.path_type == "free_path_2d"
    assert sample.canonical_state_frames is not None
    assert sample.data_scope == "simulated_dopplersim_only"
    xy = xy_from_state(sample.canonical_state_frames)
    assert orbit_align(xy, xy).rms < 1e-10


def test_load_from_render_parent(tmp_path: Path):
    render = tmp_path / "render_abc"
    _write_mini_phase1(render / "phase1")
    sample = load_phase1_sample(render)
    assert sample.root == (render / "phase1").resolve()


def test_inference_bundle_strips_gt(tmp_path: Path):
    sample = load_phase1_sample(_write_mini_phase1(tmp_path / "phase1"))
    bundle = to_inference_bundle(sample)
    assert bundle.stft_db is not None
    assert not hasattr(bundle, "state_frames")
    assert not hasattr(bundle, "canonical_state_frames")


def test_misaligned_stft_raises(tmp_path: Path):
    root = _write_mini_phase1(tmp_path / "phase1", t=32, f=8)
    np.save(root / "spectrograms" / "stft.npy", np.zeros((8, 16), dtype=np.float32))
    with pytest.raises(DatasetError, match="not aligned"):
        load_phase1_sample(root)
