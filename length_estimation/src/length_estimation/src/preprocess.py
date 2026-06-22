"""VS13 preprocessing and manifest utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from length_estimation.config import (
    CROP_T_MAX,
    CROP_T_MIN,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPECS_PATH,
    PACKAGE_ROOT,
    SR,
    VS13_VEHICLE_DIRS,
)

# Files to ignore when scanning vehicle folders
_SKIP_STEMS = frozenset({"Train_valid_split"})


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    vehicle: str
    speed_kmh: float
    cpa_time_s: float
    wav_path: Path
    length_m: float
    wheelbase_m: float
    power_kw: float
    engine_type: str
    split: str = "unknown"  # train | valid | unknown (from Train_valid_split.txt)


def resolve_data_dir(data_dir: Path | None = None) -> Path:
    """Locate VS13 root. Tries explicit path, then standard locations."""
    if data_dir is not None:
        path = Path(data_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Data directory not found: {path}")
        return path

    candidates = (
        PACKAGE_ROOT / "data" / "vs13",
        PACKAGE_ROOT / "data" / "content" / "vs13",
    )
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path

    return candidates[0]


def load_vehicle_specs(specs_path: Path | None = None) -> pd.DataFrame:
    path = specs_path or DEFAULT_SPECS_PATH
    df = pd.read_csv(path)
    return df.set_index("short_name")


def _parse_annotation(txt_path: Path) -> tuple[float, float]:
    """Parse VS13 annotation: single line 'speed cpa' or two-line format."""
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty annotation: {txt_path}")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1:
        parts = lines[0].split()
        if len(parts) < 2:
            raise ValueError(f"Expected 'speed cpa_time' in {txt_path}: {lines[0]!r}")
        return float(parts[0]), float(parts[1])

    return float(lines[0]), float(lines[1])


def _parse_clip_stem(stem: str, vehicle: str) -> float | None:
    """Extract speed from '{VehicleName}_{speed}' stem."""
    prefix = f"{vehicle}_"
    if not stem.startswith(prefix):
        return None
    speed_part = stem[len(prefix) :]
    if not speed_part.isdigit():
        return None
    return float(speed_part)


def _load_train_valid_split(vdir: Path) -> dict[str, str]:
    split_path = vdir / "Train_valid_split.txt"
    if not split_path.exists():
        return {}
    mapping: dict[str, str] = {}
    for line in split_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1].lower()
    return mapping


def discover_clips(data_dir: Path, specs_path: Path | None = None) -> list[ClipRecord]:
    """
    Scan VS13 layout::

        {data_dir}/{VehicleName}/{VehicleName}_{speed}.wav
        {data_dir}/{VehicleName}/{VehicleName}_{speed}.txt   # "speed_kmh cpa_time_s"
        {data_dir}/{VehicleName}/Train_valid_split.txt       # optional
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []

    specs = load_vehicle_specs(specs_path)
    records: list[ClipRecord] = []

    vehicle_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not vehicle_dirs:
        vehicle_dirs = [data_dir / name for name in VS13_VEHICLE_DIRS]

    for vdir in vehicle_dirs:
        if not vdir.is_dir():
            continue
        vehicle = vdir.name
        if vehicle not in specs.index:
            continue

        spec_row = specs.loc[vehicle]
        split_map = _load_train_valid_split(vdir)

        for wav_path in sorted(vdir.glob("*.wav")):
            if wav_path.stem in _SKIP_STEMS:
                continue

            speed_from_name = _parse_clip_stem(wav_path.stem, vehicle)
            if speed_from_name is None:
                continue

            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                continue

            try:
                speed_ann, cpa_time_s = _parse_annotation(txt_path)
            except ValueError:
                continue

            records.append(
                ClipRecord(
                    clip_id=wav_path.stem,
                    vehicle=vehicle,
                    speed_kmh=float(speed_ann),
                    cpa_time_s=cpa_time_s,
                    wav_path=wav_path.resolve(),
                    length_m=float(spec_row["length_m"]),
                    wheelbase_m=float(spec_row["wheelbase_m"]),
                    power_kw=float(spec_row["power_kw"]),
                    engine_type=str(spec_row["engine_type"]),
                    split=split_map.get(wav_path.stem, "unknown"),
                )
            )

    records.sort(key=lambda r: (r.vehicle, r.speed_kmh, r.clip_id))
    return records


def load_clips(
    data_dir: Path | None = None,
    specs_path: Path | None = None,
    *,
    write_manifest: bool = False,
    manifest_path: Path | None = None,
) -> list[ClipRecord]:
    """
    Discover clips from disk (called automatically by features / phase-b).

    Set ``write_manifest=True`` to also refresh ``outputs/manifest.csv``.
    """
    root = resolve_data_dir(data_dir)
    records = discover_clips(root, specs_path)
    if write_manifest and records:
        out = manifest_path or (DEFAULT_OUTPUT_DIR / "manifest.csv")
        save_manifest(records, out)
    return records


def load_audio(wav_path: Path, sr: int = SR) -> tuple[np.ndarray, int]:
    y, file_sr = librosa.load(wav_path, sr=sr, mono=True)
    return y, file_sr


def align_and_crop(
    y: np.ndarray,
    sr: int,
    cpa_time_s: float,
    t_min: float = CROP_T_MIN,
    t_max: float = CROP_T_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cropped audio and relative time axis t̃ in seconds."""
    n_samples = len(y)
    duration = n_samples / sr
    t_abs = np.arange(n_samples, dtype=np.float64) / sr

    start_s = max(0.0, cpa_time_s + t_min)
    end_s = min(duration, cpa_time_s + t_max)
    i0 = int(round(start_s * sr))
    i1 = int(round(end_s * sr))
    i0 = max(0, min(i0, n_samples))
    i1 = max(i0 + 1, min(i1, n_samples))

    y_crop = y[i0:i1]
    t_rel = t_abs[i0:i1] - cpa_time_s
    return y_crop, t_rel


def records_to_dataframe(records: list[ClipRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "clip_id": r.clip_id,
                "vehicle": r.vehicle,
                "speed_kmh": r.speed_kmh,
                "speed_mps": r.speed_kmh / 3.6,
                "cpa_time_s": r.cpa_time_s,
                "wav_path": str(r.wav_path),
                "length_m": r.length_m,
                "wheelbase_m": r.wheelbase_m,
                "power_kw": r.power_kw,
                "engine_type": r.engine_type,
                "split": r.split,
            }
            for r in records
        ]
    )


def save_manifest(records: list[ClipRecord], out_path: Path) -> pd.DataFrame:
    df = records_to_dataframe(records)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
