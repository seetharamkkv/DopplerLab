"""Phase 4 tiered validation tests (simulated)."""

from __future__ import annotations

from pathlib import Path

from traj_reconstruction.path_families import make_straight
from traj_reconstruction.tiers import (
    synthesize_tier1,
    synthesize_tier2_harmonics_rpm,
    synthesize_tier3_noise,
    synthesize_tier4_multipath,
    synthesize_tier5_directivity,
)
from traj_reconstruction.validate import run_all_tiers, run_tier1


def test_tier_synthesizers_smoke():
    xy = make_straight(cpa_distance_m=12.0, half_length_m=40.0)
    t1 = synthesize_tier1(xy, speed_mps=15.0)
    t2 = synthesize_tier2_harmonics_rpm(xy, speed_mps=15.0)
    t3 = synthesize_tier3_noise(xy, speed_mps=15.0, snr_db=10.0)
    t4 = synthesize_tier4_multipath(xy, speed_mps=15.0)
    t5 = synthesize_tier5_directivity(xy, speed_mps=15.0)
    assert t1["tier"] == "tier1" and len(t1["audio"]) > 1000
    assert t2["gear_shift"] is True and t2["n_harmonics"] >= 2
    assert t3["snr_db"] == 10.0
    assert t4["reflection_gain"] > 0
    assert t5["scale_ambiguous"] is True


def test_run_tier1_has_orbit_metric():
    rows = run_tier1(speed_mps=18.0)
    assert len(rows) >= 4
    assert all(r.orbit_rms == r.orbit_rms for r in rows)  # not NaN
    assert any(r.method == "flexible" for r in rows)
    assert any(r.method == "parametric_straight" for r in rows)


def test_run_all_tiers_writes_report(tmp_path: Path):
    out = tmp_path / "val"
    report = run_all_tiers(out_dir=out, snr_grid_db=[20.0, 5.0], save_worst=1)
    assert (out / "tiered_validation.json").is_file()
    assert (out / "tiered_validation.md").is_file()
    assert (out / "snr_curve.png").is_file()
    assert set(report["summary"]) >= {"tier1", "tier2", "tier3", "tier4", "tier5"}
    assert "orbit_rms" in report["headline_metric"]
    assert report["gates"]["tier5"]["scale_ambiguous"] is True
