"""Leakage-aware pass-by data loading for speed / length / SE-ResNet models.

Public API
----------
- discover / resolve UIDs
- build scene-level, LOVO, or speed-stratified splits
- fit mel stats on the *fit* partition only
- assert_no_leakage before training or eval
"""

from .catalog import (
    ClipRef,
    condition_key,
    discover_clips,
    discover_uids,
    normalize_uid,
    resolve_uid_path,
    uid_to_speed,
    uid_to_vehicle,
)
from .config import AudioConfig, DEFAULT_SPLITS_DIR, DEFAULT_REAL_ROOT
from .leakage import assert_no_leakage, leakage_report
from .loader import ModelDataBundle, load_for_training, preprocess_mel
from .mel_stats import MelStats, compute_mel_stats, load_mel_stats, save_mel_stats
from .splits import (
    SplitBundle,
    adjacent_twin_uids,
    build_inner_scene_val,
    build_lovo_folds,
    build_scene_split,
    build_speed_stratified_split,
    enforce_min_speed_gap,
    load_split,
    save_split,
)

__all__ = [
    "AudioConfig",
    "ClipRef",
    "DEFAULT_SPLITS_DIR",
    "DEFAULT_REAL_ROOT",
    "MelStats",
    "ModelDataBundle",
    "SplitBundle",
    "adjacent_twin_uids",
    "assert_no_leakage",
    "build_inner_scene_val",
    "build_lovo_folds",
    "build_scene_split",
    "build_speed_stratified_split",
    "compute_mel_stats",
    "condition_key",
    "discover_clips",
    "discover_uids",
    "enforce_min_speed_gap",
    "leakage_report",
    "load_for_training",
    "load_mel_stats",
    "load_split",
    "normalize_uid",
    "preprocess_mel",
    "resolve_uid_path",
    "save_mel_stats",
    "save_split",
    "uid_to_speed",
    "uid_to_vehicle",
]
