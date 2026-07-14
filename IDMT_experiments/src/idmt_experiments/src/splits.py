"""Leak-safe train/valid/test splits for IDMT-Traffic.

REPRODUCIBILITY BASELINE — shared CNN dependency (mel_3class / mel_3class_left / mel_3class_right)
---------------------------------------------------------------------------------
``build_eusipco_split`` (default ``split_name='eusipco'``, seed=42, val_fraction=0.1) defines
the published train/valid/test partitions. Do not change without re-benchmarking.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from idmt_experiments.config import DEFAULT_OUTPUT_DIR, DEFAULT_SHARED_OUTPUT_DIR, WEATHER_EVAL_SITE
from idmt_experiments.src.preprocess import (
    ClipRecord,
    annotation_paths,
    discover_all_clips,
    load_file_list,
    parse_records_from_file_list,
    resolve_audio_dir,
    resolve_data_dir,
)


def _sanitize_location(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def assign_splits(records: list[ClipRecord], split_map: dict[str, str]) -> list[ClipRecord]:
    return [replace(r, split=split_map.get(r.clip_id, "unknown")) for r in records]


def _event_ids(records: list[ClipRecord]) -> set[str]:
    return {r.event_id for r in records}


def verify_no_event_leakage(train: list[ClipRecord], valid: list[ClipRecord], test: list[ClipRecord]) -> dict:
    tr, va, te = _event_ids(train), _event_ids(valid), _event_ids(test)
    overlap_tv = tr & va
    overlap_tt = tr & te
    overlap_vt = va & te
    ok = not (overlap_tv or overlap_tt or overlap_vt)
    return {
        "ok": ok,
        "n_train_events": len(tr),
        "n_valid_events": len(va),
        "n_test_events": len(te),
        "train_valid_event_overlap": sorted(overlap_tv),
        "train_test_event_overlap": sorted(overlap_tt),
        "valid_test_event_overlap": sorted(overlap_vt),
    }


def split_by_events(
    records: list[ClipRecord],
    event_to_split: dict[str, str],
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord]]:
    train, valid, test = [], [], []
    for r in records:
        split = event_to_split.get(r.event_id, "unknown")
        rec = replace(r, split=split)
        if split == "train":
            train.append(rec)
        elif split == "valid":
            valid.append(rec)
        elif split == "test":
            test.append(rec)
    return train, valid, test


def _events_by_id(records: list[ClipRecord]) -> dict[str, list[ClipRecord]]:
    by_event: dict[str, list[ClipRecord]] = {}
    for r in records:
        by_event.setdefault(r.event_id, []).append(r)
    return by_event


def _val_holdout_from_train_events(
    train_records: list[ClipRecord],
    val_fraction: float,
    seed: int,
    *,
    stratify_fn=None,
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Hold out validation events from training pool (never overlaps test).

    When stratify_fn is provided (e.g. weather code), each stratum contributes
    ~val_fraction of its events to validation so class balance is preserved.
    """
    by_event = _events_by_id(train_records)

    if stratify_fn is None:
        event_ids = sorted(by_event.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(event_ids)
        n_val = max(1, int(round(len(event_ids) * val_fraction)))
        val_events = set(event_ids[:n_val])
    else:
        strata: dict[str, list[str]] = {}
        for eid, clips in by_event.items():
            key = stratify_fn(clips[0])
            strata.setdefault(str(key), []).append(eid)
        rng = np.random.default_rng(seed)
        val_events: set[str] = set()
        for _key, eids in strata.items():
            eids = sorted(eids)
            rng.shuffle(eids)
            n_val = max(1, int(round(len(eids) * val_fraction)))
            val_events.update(eids[:n_val])

    train_out, val_out = [], []
    for eid, clips in by_event.items():
        for c in clips:
            if eid in val_events:
                val_out.append(replace(c, split="valid"))
            else:
                train_out.append(replace(c, split="train"))
    return train_out, val_out


def _weather_vehicle_records(
    data_dir=None,
    *,
    mic_filter: str = "SE",
    channel_filter: str = "CH34",
    location: str | None = None,
) -> list[ClipRecord]:
    records = discover_all_clips(data_dir, mic_filter=mic_filter, channel_filter=channel_filter)
    vehicle = [r for r in records if not r.is_background and r.weather in ("D", "W")]
    if location is not None:
        vehicle = [r for r in vehicle if r.location == location]
    return vehicle


def _event_weather_map(by_event: dict[str, list[ClipRecord]]) -> dict[str, str]:
    event_weather: dict[str, str] = {}
    for eid, clips in by_event.items():
        codes = {c.weather for c in clips}
        if len(codes) != 1:
            raise ValueError(f"Event {eid} has mixed weather codes: {codes}")
        event_weather[eid] = next(iter(codes))
    return event_weather


def _stratified_event_holdout(
    by_event: dict[str, list[ClipRecord]],
    event_weather: dict[str, str],
    *,
    test_fraction: float,
    seed: int,
    weather_codes: tuple[str, ...] = ("D", "W"),
) -> tuple[set[str], set[str], dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)
    test_events: set[str] = set()
    train_events: set[str] = set()
    per_class_counts: dict[str, dict[str, int]] = {}

    for code in weather_codes:
        events = [eid for eid, w in event_weather.items() if w == code]
        rng.shuffle(events)
        n_test = max(1, int(round(len(events) * test_fraction)))
        test_events.update(events[:n_test])
        train_events.update(events[n_test:])
        per_class_counts[code] = {
            "n_events": len(events),
            "n_test_events": n_test,
            "n_train_pool_events": len(events) - n_test,
        }

    return test_events, train_events, per_class_counts


def _partition_events_to_splits(
    by_event: dict[str, list[ClipRecord]],
    test_events: set[str],
    train_events: set[str],
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    train_pool: list[ClipRecord] = []
    test_records: list[ClipRecord] = []
    for eid, clips in by_event.items():
        for rec in clips:
            if eid in test_events:
                test_records.append(replace(rec, split="test"))
            elif eid in train_events:
                train_pool.append(replace(rec, split="train"))
            else:
                raise ValueError(f"Event {eid} not assigned to train or test")
    return train_pool, test_records


def _weather_counts(recs: list[ClipRecord]) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(r.weather for r in recs))


def _weather_split_meta_base(
    *,
    split_name: str,
    mic_filter: str,
    channel_filter: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    audit: dict,
    per_class_counts: dict,
    location: str | None = None,
    confound_warning: str | None = None,
) -> dict:
    meta = {
        "split_name": split_name,
        "mic_filter": mic_filter,
        "channel_filter": channel_filter,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "audit": audit,
        "per_class_event_counts": per_class_counts,
    }
    if location is not None:
        meta["location"] = location
    if confound_warning:
        meta["confound_warning"] = confound_warning
    return meta


def build_eusipco_split(
    data_dir=None,
    *,
    mic_filter: str = "SE",
    channel_filter: str = "CH34",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord], dict]:
    """Official EUSIPCO paper split (REPRODUCIBILITY BASELINE — CNN train/eval partitions)."""
    data_dir = resolve_data_dir(data_dir)
    paths = annotation_paths(data_dir)
    audio_dir = resolve_audio_dir(data_dir)

    train_files = set(load_file_list(paths["eusipco_train"]))
    test_files = set(load_file_list(paths["eusipco_test"]))

    train_raw = parse_records_from_file_list(
        sorted(train_files), audio_dir, mic_filter=mic_filter, channel_filter=channel_filter
    )
    test_records = parse_records_from_file_list(
        sorted(test_files), audio_dir, mic_filter=mic_filter, channel_filter=channel_filter
    )
    for i, r in enumerate(test_records):
        test_records[i] = replace(r, split="test")

    train_records, val_records = _val_holdout_from_train_events(train_raw, val_fraction, seed)
    audit = verify_no_event_leakage(train_records, val_records, test_records)
    meta = {
        "split_name": "eusipco",
        "mic_filter": mic_filter,
        "channel_filter": channel_filter,
        "val_fraction": val_fraction,
        "seed": seed,
        "audit": audit,
    }
    return train_records, val_records, test_records, meta


def build_location_loo_splits(
    data_dir=None,
    *,
    mic_filter: str = "SE",
    channel_filter: str = "CH34",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> list[tuple[str, list[ClipRecord], list[ClipRecord], list[ClipRecord], dict]]:
    records = discover_all_clips(data_dir, mic_filter=mic_filter, channel_filter=channel_filter)
    locations = sorted({r.location for r in records})
    folds: list[tuple[str, list[ClipRecord], list[ClipRecord], list[ClipRecord], dict]] = []

    for held_out in locations:
        test = [r for r in records if r.location == held_out]
        pool = [r for r in records if r.location != held_out]
        train_pool, val = _val_holdout_from_train_events(pool, val_fraction, seed)
        for i, r in enumerate(test):
            test[i] = replace(r, split="test")
        audit = verify_no_event_leakage(train_pool, val, test)
        meta = {
            "split_name": "location_loo",
            "held_out_location": held_out,
            "mic_filter": mic_filter,
            "channel_filter": channel_filter,
            "val_fraction": val_fraction,
            "seed": seed,
            "audit": audit,
        }
        folds.append((held_out, train_pool, val, test, meta))
    return folds


def build_weather_holdout_split(
    data_dir=None,
    *,
    mic_filter: str = "SE",
    channel_filter: str = "CH34",
    train_weather: str = "D",
    test_weather: str = "W",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord], dict]:
    """Train on dry (D) events; test on wet (W) events. Background clips use weather=None."""
    records = discover_all_clips(data_dir, mic_filter=mic_filter, channel_filter=channel_filter)
    dry = [r for r in records if r.weather == train_weather or (r.is_background and r.weather in (train_weather, "None", None))]
    wet = [r for r in records if r.weather == test_weather]

    # Background clips only in dry pool (no wet background in IDMT for this filter)
    dry = [r for r in dry if r.weather == train_weather or r.is_background]
    train_records, val_records = _val_holdout_from_train_events(dry, val_fraction, seed)
    test_records = [replace(r, split="test") for r in wet]
    audit = verify_no_event_leakage(train_records, val_records, test_records)
    meta = {
        "split_name": "weather_holdout",
        "train_weather": train_weather,
        "test_weather": test_weather,
        "mic_filter": mic_filter,
        "channel_filter": channel_filter,
        "val_fraction": val_fraction,
        "seed": seed,
        "audit": audit,
        "n_dry_clips": len(dry),
        "n_wet_clips": len(wet),
    }
    return train_records, val_records, test_records, meta


def build_weather_stratified_split(
    data_dir=None,
    *,
    mic_filter: str = "SE",
    channel_filter: str = "CH34",
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord], dict]:
    """Stratified dry/wet split across all recording sites (pooled).

    WARNING: In IDMT-Traffic, wet clips were recorded only at Schleusinger-Allee.
    Dry clips at the other two sites are trivially separable by location, so pooled
    accuracy is inflated. Use ``weather_site`` for publication-grade evaluation.
    """
    vehicle = _weather_vehicle_records(data_dir, mic_filter=mic_filter, channel_filter=channel_filter)
    by_event = _events_by_id(vehicle)
    event_weather = _event_weather_map(by_event)

    test_events, train_events, per_class_counts = _stratified_event_holdout(
        by_event, event_weather, test_fraction=test_fraction, seed=seed
    )
    train_pool, test_records = _partition_events_to_splits(by_event, test_events, train_events)
    train_records, val_records = _val_holdout_from_train_events(
        train_pool, val_fraction, seed, stratify_fn=lambda r: r.weather
    )
    audit = verify_no_event_leakage(train_records, val_records, test_records)

    meta = _weather_split_meta_base(
        split_name="weather_stratified",
        mic_filter=mic_filter,
        channel_filter=channel_filter,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        audit=audit,
        per_class_counts=per_class_counts,
        confound_warning=(
            "Wet recordings exist only at Schleusinger-Allee. Non-Schleusinger dry clips "
            "inflate pooled accuracy — prefer split weather_site for honest evaluation."
        ),
    )
    meta.update(
        {
            "n_train_clips": len(train_records),
            "n_valid_clips": len(val_records),
            "n_test_clips": len(test_records),
            "train_weather_clips": _weather_counts(train_records),
            "valid_weather_clips": _weather_counts(val_records),
            "test_weather_clips": _weather_counts(test_records),
        }
    )
    return train_records, val_records, test_records, meta


def build_weather_site_split(
    data_dir=None,
    *,
    location: str = WEATHER_EVAL_SITE,
    mic_filter: str = "SE",
    channel_filter: str = "CH34",
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord], dict]:
    """Stratified dry/wet split at a single site (default: Schleusinger-Allee).

    This is the leak-safe, confound-controlled split for weather classification:
    both dry and wet clips share the same microphone geometry, road layout, and
    speed limit, so the model cannot exploit location as a proxy label.
    """
    vehicle = _weather_vehicle_records(
        data_dir, mic_filter=mic_filter, channel_filter=channel_filter, location=location
    )
    if not vehicle:
        raise ValueError(f"No dry/wet vehicle clips at location {location!r}")

    by_event = _events_by_id(vehicle)
    event_weather = _event_weather_map(by_event)
    for code in ("D", "W"):
        if not any(w == code for w in event_weather.values()):
            raise ValueError(
                f"Location {location!r} has no {code} events — cannot build weather_site split"
            )

    test_events, train_events, per_class_counts = _stratified_event_holdout(
        by_event, event_weather, test_fraction=test_fraction, seed=seed
    )
    train_pool, test_records = _partition_events_to_splits(by_event, test_events, train_events)
    train_records, val_records = _val_holdout_from_train_events(
        train_pool, val_fraction, seed, stratify_fn=lambda r: r.weather
    )
    audit = verify_no_event_leakage(train_records, val_records, test_records)

    meta = _weather_split_meta_base(
        split_name="weather_site",
        mic_filter=mic_filter,
        channel_filter=channel_filter,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        audit=audit,
        per_class_counts=per_class_counts,
        location=location,
    )
    meta.update(
        {
            "n_train_clips": len(train_records),
            "n_valid_clips": len(val_records),
            "n_test_clips": len(test_records),
            "train_weather_clips": _weather_counts(train_records),
            "valid_weather_clips": _weather_counts(val_records),
            "test_weather_clips": _weather_counts(test_records),
        }
    )
    return train_records, val_records, test_records, meta


def build_split(
    split_name: str,
    data_dir=None,
    *,
    mic_filter: str = "SE",
    channel_filter: str = "CH34",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord], dict]:
    if split_name == "eusipco":
        return build_eusipco_split(
            data_dir,
            mic_filter=mic_filter,
            channel_filter=channel_filter,
            val_fraction=val_fraction,
            seed=seed,
        )
    if split_name == "weather_holdout":
        return build_weather_holdout_split(
            data_dir,
            mic_filter=mic_filter,
            channel_filter=channel_filter,
            val_fraction=val_fraction,
            seed=seed,
        )
    if split_name == "weather_stratified":
        return build_weather_stratified_split(
            data_dir,
            mic_filter=mic_filter,
            channel_filter=channel_filter,
            val_fraction=val_fraction,
            seed=seed,
        )
    if split_name == "weather_site":
        return build_weather_site_split(
            data_dir,
            mic_filter=mic_filter,
            channel_filter=channel_filter,
            val_fraction=val_fraction,
            seed=seed,
        )
    raise ValueError(f"Unknown split_name: {split_name}")


def persist_split_meta(meta: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def default_split_meta_path(split_name: str, output_dir: Path | None = None) -> Path:
    root = output_dir or DEFAULT_SHARED_OUTPUT_DIR
    return root / "splits" / f"{split_name}.json"
