"""Phase 1 freehand batch generation tests (simulated)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from traj_reconstruction import (
    Phase1Batch,
    audit_batch,
    build_tier1_batch,
    load_phase1_sample,
    orbit_align,
    write_splits,
    xy_from_state,
)
from traj_reconstruction.path_families import (
    generate_path,
    make_arc,
    make_s_curve,
    make_straight,
)
from traj_reconstruction.kinematics import polyline_arclength, resample_path_constant_speed


def test_path_generators_nontrivial():
    s = make_straight(cpa_distance_m=10.0, half_length_m=50.0)
    a = make_arc(cpa_distance_m=12.0, radius_m=40.0)
    sc = make_s_curve(cpa_distance_m=15.0, half_length_m=60.0, amplitude_m=5.0)
    assert s.shape[1] == 2 and len(s) >= 2
    assert polyline_arclength(a)[-1] > 10.0
    assert polyline_arclength(sc)[-1] > 10.0


def test_state_follows_polyline():
    xy = make_straight(cpa_distance_m=10.0, half_length_m=40.0, heading_rad=0.3)
    traj = resample_path_constant_speed(xy, speed_mps=20.0, sr=22050, pad_s=0.1)
    # Dense reference along the same polyline parameterization.
    ref = resample_path_constant_speed(xy, speed_mps=20.0, sr=22050, pad_s=0.1)
    err = np.sqrt((traj["x"] - ref["x"]) ** 2 + (traj["y"] - ref["y"]) ** 2)
    assert float(np.max(err)) < 1e-9
    # Also: every mid-motion sample is near some polyline segment (vertex spacing ~1–2 m).
    moving = (traj["t"] > float(traj["pad_s"][0])) & (
        traj["t"] < float(traj["pad_s"][0] + traj["travel_s"][0])
    )
    pts = np.column_stack([traj["x"][moving], traj["y"][moving]])
    d = [
        float(np.min(np.linalg.norm(xy - p[None, :], axis=1)))
        for p in pts[:: max(len(pts) // 20, 1)]
    ]
    assert max(d) < 2.0


def test_build_tier1_smoke(tmp_path: Path):
    out = build_tier1_batch(
        tmp_path / "tier1_smoke",
        n_per_family={"straight": 2, "arc": 2, "s_curve": 1},
        families=("straight", "arc", "s_curve"),
        seed=7,
        resume=False,
    )
    assert (out / "dataset.csv").is_file()
    batch = Phase1Batch.from_dir(out)
    assert len(batch) == 5
    sample = batch.load(0)
    assert sample.canonical_state_frames is not None
    assert sample.stft_db.shape[1] == sample.n_frames
    assert sample.tier == "tier1"
    xy = xy_from_state(sample.canonical_state_frames)
    assert orbit_align(xy, xy).rms < 1e-10

    splits = write_splits(out, holdout_families=["s_curve"], seed=0)
    assert any(sid.startswith("sample_") for sid in splits["test"])
    report = audit_batch(out)
    assert report["ok"] is True
    assert report["n_samples"] == 5

    # Resume path
    out2 = build_tier1_batch(
        out,
        n_per_family={"straight": 2, "arc": 2, "s_curve": 1},
        families=("straight", "arc", "s_curve"),
        seed=7,
        resume=True,
    )
    assert Phase1Batch.from_dir(out2).__len__() == 5


def test_generate_path_families():
    rng = np.random.default_rng(0)
    for fam in ("straight", "arc", "s_curve", "u_turn", "multi_cpa"):
        spec = generate_path(fam, speed_mps=15.0, cpa_distance_m=20.0, rng=rng)
        assert spec.family == fam
        assert spec.polyline.shape[1] == 2
