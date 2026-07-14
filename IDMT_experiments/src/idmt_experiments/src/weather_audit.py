"""Confound and baseline audits for IDMT dry/wet classification."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from idmt_experiments.config import WEATHER_EVAL_SITE, WEATHER_CODE_TO_IDX
from idmt_experiments.src.preprocess import ClipRecord, discover_all_clips, filter_for_task, weather_label
from idmt_experiments.src.splits import build_weather_site_split, build_weather_stratified_split


def _location_weather_table(records: list[ClipRecord]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = defaultdict(lambda: {"D": 0, "W": 0})
    for r in records:
        table[r.location][r.weather] += 1
    return {loc: dict(counts) for loc, counts in sorted(table.items())}


def _majority_baseline(y_true: np.ndarray, n_classes: int = 2) -> float:
    if len(y_true) == 0:
        return 0.0
    counts = Counter(y_true.tolist())
    majority = max(counts, key=counts.get)
    return float(np.mean(y_true == majority))


def _location_oracle_preds(records: list[ClipRecord]) -> np.ndarray:
    """Trivial confound baseline: dry everywhere except always-wet at Schleusinger."""
    preds = []
    for r in records:
        if r.location == WEATHER_EVAL_SITE:
            preds.append(WEATHER_CODE_TO_IDX["W"])
        else:
            preds.append(WEATHER_CODE_TO_IDX["D"])
    return np.array(preds, dtype=int)


def _location_oracle_train_majority_preds(
    train_records: list[ClipRecord],
    eval_records: list[ClipRecord],
) -> np.ndarray:
    """At Schleusinger, predict the train-set majority weather class."""
    site_train = [r for r in train_records if r.location == WEATHER_EVAL_SITE]
    if not site_train:
        majority_wet = WEATHER_CODE_TO_IDX["W"]
    else:
        counts = Counter(weather_label(r) for r in site_train)
        majority_wet = max(counts, key=counts.get)

    preds = []
    for r in eval_records:
        if r.location == WEATHER_EVAL_SITE:
            preds.append(majority_wet)
        else:
            preds.append(WEATHER_CODE_TO_IDX["D"])
    return np.array(preds, dtype=int)


def audit_weather_dataset(
    data_dir=None,
    *,
    mic_filter: str = "SE",
    channel_filter: str = "CH34",
) -> dict:
    """Report location confounds and naive baselines on pooled vs site-only splits."""
    all_records = discover_all_clips(data_dir, mic_filter=mic_filter, channel_filter=channel_filter)
    vehicle = filter_for_task(all_records, 2, task="weather")

    pooled_train, _pooled_val, pooled_test, pooled_meta = build_weather_stratified_split(
        data_dir, mic_filter=mic_filter, channel_filter=channel_filter
    )
    site_train, _site_val, site_test, site_meta = build_weather_site_split(
        data_dir, mic_filter=mic_filter, channel_filter=channel_filter
    )

    pooled_train = filter_for_task(pooled_train, 2, task="weather")
    pooled_test = filter_for_task(pooled_test, 2, task="weather")
    site_test = filter_for_task(site_test, 2, task="weather")

    y_pooled_test = np.array([weather_label(r) for r in pooled_test], dtype=int)
    y_site_test = np.array([weather_label(r) for r in site_test], dtype=int)

    loc_oracle_pooled = _location_oracle_preds(pooled_test)
    loc_oracle_site = _location_oracle_preds(site_test)
    loc_majority_pooled = _location_oracle_train_majority_preds(pooled_train, pooled_test)

    wet_sites = [
        loc
        for loc, counts in _location_weather_table(vehicle).items()
        if counts.get("W", 0) > 0
    ]

    return {
        "mic_filter": mic_filter,
        "channel_filter": channel_filter,
        "n_vehicle_clips": len(vehicle),
        "location_weather_counts": _location_weather_table(vehicle),
        "sites_with_wet_recordings": wet_sites,
        "location_confounded": len(wet_sites) == 1 and wet_sites[0] == WEATHER_EVAL_SITE,
        "recommended_split": "weather_site",
        "recommended_site": WEATHER_EVAL_SITE,
        "pooled_split": {
            "split_name": "weather_stratified",
            "confound_warning": pooled_meta.get("confound_warning"),
            "test_clips": len(pooled_test),
            "test_weather_clips": pooled_meta["test_weather_clips"],
            "baselines": {
                "always_dry": float(np.mean(y_pooled_test == WEATHER_CODE_TO_IDX["D"])),
                "majority_class": _majority_baseline(y_pooled_test),
                "location_oracle_always_wet_at_site": float(np.mean(loc_oracle_pooled == y_pooled_test)),
                "location_oracle_train_majority_at_site": float(
                    np.mean(loc_majority_pooled == y_pooled_test)
                ),
            },
        },
        "site_split": {
            "split_name": "weather_site",
            "location": WEATHER_EVAL_SITE,
            "test_clips": len(site_test),
            "test_weather_clips": site_meta["test_weather_clips"],
            "baselines": {
                "always_dry": float(np.mean(y_site_test == WEATHER_CODE_TO_IDX["D"])),
                "majority_class": _majority_baseline(y_site_test),
                "location_oracle_always_wet_at_site": float(np.mean(loc_oracle_site == y_site_test)),
            },
        },
    }
