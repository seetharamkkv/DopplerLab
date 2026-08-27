"""Phase 6 leakage audit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from traj_reconstruction.batch import build_tier1_batch
from traj_reconstruction.leakage import (
    audit_forbidden_relpaths,
    audit_predict_signature,
    run_leakage_audit,
    wav_only_smoke,
)
from traj_reconstruction.splits import write_splits


def test_predict_signature_audio_only():
    report = audit_predict_signature()
    assert report.ok


def test_forbidden_paths_block_metadata():
    report = audit_forbidden_relpaths()
    assert report.ok


def test_full_audit_with_batch(tmp_path: Path):
    out = build_tier1_batch(
        tmp_path / "batch",
        n_per_family={"straight": 3, "arc": 2},
        families=("straight", "arc"),
        seed=1,
        resume=False,
    )
    write_splits(out, holdout_families=["arc"], seed=0)
    report = run_leakage_audit(out)
    assert report.ok
    assert any(f.code == "split_overlap" for f in report.findings)


def test_wav_only_smoke(tmp_path: Path):
    out = build_tier1_batch(
        tmp_path / "batch",
        n_per_family={"straight": 1},
        families=("straight",),
        seed=2,
        resume=False,
    )
    wav = next((out / "audio_clips").glob("sample_*/*.wav"))
    smoke = wav_only_smoke(wav, tmp_path / "smoke", method="parametric")
    assert smoke["ok"] is True
    assert smoke["files_in_staging"] == [wav.name]
    assert Path(smoke["artifacts"]["html"]).is_file()
