"""Unit tests for leakage-free pass-by splits (no TensorFlow required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from passby_data.catalog import condition_key, discover_clips, uid_to_vehicle  # noqa: E402
from passby_data.config import DEFAULT_REAL_ROOT  # noqa: E402
from passby_data.leakage import assert_no_leakage, leakage_report  # noqa: E402
from passby_data.splits import (  # noqa: E402
    adjacent_twin_uids,
    build_lovo_folds,
    build_scene_split,
    build_speed_stratified_split,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_REAL_ROOT.is_dir(),
    reason=f"Data root missing: {DEFAULT_REAL_ROOT}",
)


def test_discover_400():
    clips = discover_clips(DEFAULT_REAL_ROOT)
    assert len(clips) == 400
    assert len({c.vehicle for c in clips}) == 13


def test_scene_split_no_leakage():
    split = build_scene_split(DEFAULT_REAL_ROOT, seed=42)
    report = assert_no_leakage(
        split, stats_uids=split.fit_uids, eval_uids=split.test_uids
    )
    assert report.ok
    assert not (set(split.train_uids) & set(split.test_uids))
    assert not (set(split.fit_uids) & set(split.val_uids))
    train_keys = {condition_key(u) for u in split.train_uids}
    test_keys = {condition_key(u) for u in split.test_uids}
    assert not (train_keys & test_keys)
    assert {uid_to_vehicle(u) for u in split.train_uids} & {
        uid_to_vehicle(u) for u in split.test_uids
    }


def test_stats_on_val_is_caught():
    split = build_scene_split(DEFAULT_REAL_ROOT, seed=42)
    bad = leakage_report(split, stats_uids=split.fit_uids + split.val_uids)
    assert not bad.ok
    assert any("inner-val" in e for e in bad.errors)


def test_lovo_no_vehicle_overlap():
    folds = build_lovo_folds(DEFAULT_REAL_ROOT)
    assert len(folds) == 13
    for fold in folds:
        assert_no_leakage(fold)
        train_v = {uid_to_vehicle(u) for u in fold.train_uids}
        test_v = {uid_to_vehicle(u) for u in fold.test_uids}
        assert train_v.isdisjoint(test_v)
        assert fold.held_out_vehicle in test_v


def test_speed_stratified_clip_disjoint():
    split = build_speed_stratified_split(DEFAULT_REAL_ROOT, seed=42)
    assert_no_leakage(split, stats_uids=split.fit_uids, eval_uids=split.test_uids)
    assert not (set(split.train_uids) & set(split.test_uids))


def test_default_scene_may_have_adjacent_twins():
    split = build_scene_split(DEFAULT_REAL_ROOT, seed=42, min_speed_gap=1)
    report = leakage_report(split)
    assert report.ok
    assert split.meta.get("min_speed_gap", 1) == 1


def test_spread_scene_bans_adjacent_twins():
    split = build_scene_split(DEFAULT_REAL_ROOT, seed=42, min_speed_gap=2)
    assert_no_leakage(split, stats_uids=split.fit_uids, eval_uids=split.test_uids)
    bad = adjacent_twin_uids(split.train_uids, split.test_uids, min_speed_gap=2)
    assert bad == []
    assert split.meta["min_speed_gap"] == 2
    baseline = build_scene_split(DEFAULT_REAL_ROOT, seed=42, min_speed_gap=1)
    assert len(split.test_uids) <= len(baseline.test_uids)
    assert len(split.test_uids) >= 1
