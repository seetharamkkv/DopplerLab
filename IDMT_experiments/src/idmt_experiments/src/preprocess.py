"""Parse IDMT-Traffic filenames, build clip records, manifest I/O.

REPRODUCIBILITY BASELINE — shared CNN dependency (mel_3class / mel_3class_left / mel_3class_right)
---------------------------------------------------------------------------------
``direction_label``, ``clip_label``, and ``filter_records`` define CNN labels and clip sets.
Do not change default behaviour without re-benchmarking all three reference runs.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from idmt_experiments.config import (
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_DATA_DIR,
    DirectionConfig,
    PhysicsConfig,
    VEHICLE_CODE_TO_IDX,
    WEATHER_CODE_TO_IDX,
)


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    event_id: str
    wav_path: Path
    is_background: bool
    date_time: str
    location: str
    speed_kmh: str
    sample_pos: str
    daytime: str
    weather: str
    vehicle: str
    source_direction: str
    travel_direction: str  # L2R | R2L | none
    microphone: str
    channel: str
    split: str = "unknown"  # train | valid | test | unknown


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    p = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if not p.exists():
        raise FileNotFoundError(f"IDMT data dir not found: {p}")
    return p


def resolve_audio_dir(data_dir: Path | str | None = None) -> Path:
    audio = resolve_data_dir(data_dir) / "audio"
    if not audio.is_dir():
        raise FileNotFoundError(f"IDMT audio dir not found: {audio}")
    return audio


def _travel_direction(source_direction: str) -> str:
    if source_direction == "L":
        return "L2R"
    if source_direction == "R":
        return "R2L"
    return "none"


def _parse_filename(stem: str, wav_path: Path) -> ClipRecord:
    parts = stem.split("_")
    is_background = "-BG" in stem

    if is_background:
        date_time, location, speed_kmh, sample_pos, mic, channel_raw = parts
        daytime = weather = vehicle = source_direction = "None"
        channel = channel_raw.replace("-BG", "")
        is_bg = True
    else:
        date_time, location, speed_kmh, sample_pos, daytime, weather, vehicle_direction, mic, channel = parts
        vehicle, source_direction = vehicle_direction[0], vehicle_direction[1]
        is_bg = False

    speed_kmh = speed_kmh.replace("unknownKmh", "UNK").replace("Kmh", "")
    event_id = "_".join(parts[:-2])
    travel = "none" if is_bg else _travel_direction(source_direction)

    return ClipRecord(
        clip_id=stem,
        event_id=event_id,
        wav_path=wav_path,
        is_background=is_bg,
        date_time=date_time,
        location=location,
        speed_kmh=speed_kmh,
        sample_pos=sample_pos,
        daytime=daytime,
        weather=weather,
        vehicle=vehicle if not is_bg else "none",
        source_direction=source_direction if not is_bg else "none",
        travel_direction=travel,
        microphone=mic,
        channel=channel,
    )


def load_file_list(list_path: Path) -> list[str]:
    lines = list_path.read_text(encoding="utf-8").strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def parse_records_from_file_list(
    filenames: list[str],
    audio_dir: Path,
    *,
    mic_filter: str | None = None,
    channel_filter: str | None = None,
) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for fn in filenames:
        stem = fn.replace(".wav", "")
        wav_path = audio_dir / fn
        if not wav_path.exists():
            continue
        rec = _parse_filename(stem, wav_path)
        if mic_filter and rec.microphone != mic_filter:
            continue
        if channel_filter and rec.channel != channel_filter:
            continue
        records.append(rec)
    return records


def discover_all_clips(
    data_dir: Path | str | None = None,
    *,
    mic_filter: str | None = None,
    channel_filter: str | None = None,
) -> list[ClipRecord]:
    data_dir = resolve_data_dir(data_dir)
    list_path = data_dir / "annotation" / "idmt_traffic_all.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing annotation list: {list_path}")
    filenames = load_file_list(list_path)
    return parse_records_from_file_list(
        filenames,
        resolve_audio_dir(data_dir),
        mic_filter=mic_filter,
        channel_filter=channel_filter,
    )


def records_to_dataframe(records: list[ClipRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def save_manifest(records: list[ClipRecord], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records)
    df["wav_path"] = df["wav_path"].astype(str)
    df.to_csv(path, index=False)
    return path


def direction_label(rec: ClipRecord, n_classes: int) -> int:
    """Map clip to direction class index (REPRODUCIBILITY BASELINE — CNN label mapping)."""
    if n_classes == 2:
        if rec.is_background:
            raise ValueError("2-class direction task excludes background clips")
        return 0 if rec.travel_direction == "L2R" else 1
    if rec.is_background:
        return 2
    return 0 if rec.travel_direction == "L2R" else 1


def vehicle_label(rec: ClipRecord) -> int:
    if rec.is_background:
        return 4
    code = rec.vehicle.upper()
    if code not in VEHICLE_CODE_TO_IDX:
        raise ValueError(f"Unknown vehicle code {rec.vehicle!r} in {rec.clip_id}")
    return VEHICLE_CODE_TO_IDX[code]


def weather_label(rec: ClipRecord) -> int:
    if rec.is_background:
        raise ValueError(f"Weather task excludes background clips: {rec.clip_id}")
    code = rec.weather.upper()
    if code not in WEATHER_CODE_TO_IDX:
        raise ValueError(f"Unknown weather code {rec.weather!r} in {rec.clip_id}")
    return WEATHER_CODE_TO_IDX[code]


def clip_label(rec: ClipRecord, cfg: DirectionConfig) -> int:
    if cfg.task == "vehicle":
        return vehicle_label(rec)
    if cfg.task == "weather":
        return weather_label(rec)
    return direction_label(rec, cfg.n_classes)


def filter_for_task(
    records: list[ClipRecord],
    n_classes: int,
    *,
    task: str = "direction",
) -> list[ClipRecord]:
    if task == "vehicle":
        return list(records)
    if task == "weather":
        return [r for r in records if not r.is_background and r.weather in WEATHER_CODE_TO_IDX]
    if n_classes == 2:
        return [r for r in records if not r.is_background]
    return list(records)


def filter_records(records: list[ClipRecord], cfg: DirectionConfig) -> list[ClipRecord]:
    return filter_for_task(records, cfg.n_classes, task=cfg.task)


def filter_physics_records(records: list[ClipRecord], cfg: PhysicsConfig) -> list[ClipRecord]:
    """Vehicle-only L2R/R2L when include_no_vehicle=False (default)."""
    if cfg.task != "direction":
        raise ValueError(f"Physics track supports task=direction only, got {cfg.task!r}")
    if cfg.include_no_vehicle:
        return list(records)
    return [r for r in records if not r.is_background]


def annotation_paths(data_dir: Path | None = None) -> dict[str, Path]:
    ann = (data_dir or DEFAULT_DATA_DIR) / "annotation"
    return {
        "all": ann / "idmt_traffic_all.txt",
        "eusipco_train": ann / "eusipco_2021_train.txt",
        "eusipco_test": ann / "eusipco_2021_test.txt",
    }
