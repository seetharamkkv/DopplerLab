"""Leak-safe train/valid/test splits for IDMT-Traffic."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from idmt_experiments.config import DEFAULT_OUTPUT_DIR
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


def _val_holdout_from_train_events(
    train_records: list[ClipRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Hold out validation events from training pool (never overlaps test)."""
    by_event: dict[str, list[ClipRecord]] = {}
    for r in train_records:
        by_event.setdefault(r.event_id, []).append(r)

    event_ids = sorted(by_event.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(event_ids)
    n_val = max(1, int(round(len(event_ids) * val_fraction)))
    val_events = set(event_ids[:n_val])
    train_out, val_out = [], []
    for eid, clips in by_event.items():
        for c in clips:
            if eid in val_events:
                val_out.append(replace(c, split="valid"))
            else:
                train_out.append(replace(c, split="train"))
    return train_out, val_out


def build_eusipco_split(
    data_dir=None,
    *,
    mic_filter: str = "SE",
    channel_filter: str = "CH34",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[ClipRecord], list[ClipRecord], list[ClipRecord], dict]:
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
    raise ValueError(f"Unknown split_name: {split_name}")


def persist_split_meta(meta: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def default_split_meta_path(split_name: str, output_dir: Path | None = None) -> Path:
    root = output_dir or DEFAULT_OUTPUT_DIR
    return root / "splits" / f"{split_name}.json"
