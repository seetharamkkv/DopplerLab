"""Contract and inference-safety tests."""

from __future__ import annotations

import pytest

from traj_reconstruction.contract import (
    ACOUSTIC_PRIMARY_RELPATH,
    CANONICAL_STATE_FRAMES_RELPATH,
    DATA_SCOPE,
    INFERENCE_ALLOWED_RELPATHS,
    INFERENCE_FORBIDDEN_RELPATHS,
    PATH_TYPES,
    STATE_FRAMES_RELPATH,
)
from traj_reconstruction.dataset import DatasetError, assert_inference_safe


def test_data_scope_is_simulated():
    assert DATA_SCOPE == "simulated_dopplersim_only"


def test_stft_allowed_state_forbidden():
    assert ACOUSTIC_PRIMARY_RELPATH in INFERENCE_ALLOWED_RELPATHS
    assert STATE_FRAMES_RELPATH in INFERENCE_FORBIDDEN_RELPATHS
    assert CANONICAL_STATE_FRAMES_RELPATH in INFERENCE_FORBIDDEN_RELPATHS


def test_path_types_include_freehand():
    assert "free_path_2d" in PATH_TYPES
    assert "free_path_3d" in PATH_TYPES
    assert "straight" in PATH_TYPES


def test_assert_inference_safe_ok():
    assert_inference_safe([ACOUSTIC_PRIMARY_RELPATH, "clip.wav"])


def test_assert_inference_safe_rejects_canonical():
    with pytest.raises(DatasetError, match="forbidden"):
        assert_inference_safe([ACOUSTIC_PRIMARY_RELPATH, CANONICAL_STATE_FRAMES_RELPATH])


def test_assert_inference_safe_rejects_metadata():
    with pytest.raises(DatasetError, match="forbidden"):
        assert_inference_safe(["metadata/anything.npy"])
