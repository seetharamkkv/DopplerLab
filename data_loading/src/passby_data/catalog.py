"""Pass-by clip discovery and UID helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

UID_RE = re.compile(r"^(?:real:|synth:|dense:)?([^/\\]+)[/\\]([^/\\]+\.wav)$", re.IGNORECASE)
SPEED_RE = re.compile(r"_(\d+)\.wav$", re.IGNORECASE)
_SKIP_STEMS = frozenset({"Train_valid_split"})


@dataclass(frozen=True)
class ClipRef:
    """One real (or mirrored synth) pass-by clip."""

    uid: str  # Vehicle/stem.wav
    vehicle: str
    speed_kmh: int
    wav_path: Path
    domain: str = "real"  # real | synth
    cpa_time_s: float | None = None
    official_split: str | None = None  # train | valid from Train_valid_split.txt


def normalize_uid(uid: str) -> str:
    """Canonical uid: Vehicle/stem.wav (strip domain prefixes)."""
    uid = uid.strip().replace("\\", "/")
    for prefix in ("real:", "synth:", "dense:"):
        if uid.startswith(prefix):
            uid = uid[len(prefix) :]
            break
    m = UID_RE.match(uid)
    if not m:
        raise ValueError(f"Unrecognized uid: {uid!r}")
    return f"{m.group(1)}/{m.group(2)}"


def uid_to_vehicle(uid: str) -> str:
    return normalize_uid(uid).split("/", 1)[0]


def uid_to_speed(uid: str) -> int:
    name = Path(normalize_uid(uid)).name
    m = SPEED_RE.search(name)
    if not m:
        raise ValueError(f"No integer speed in uid: {uid!r}")
    return int(m.group(1))


def condition_key(uid: str) -> str:
    """Scene key used for leakage blocking: vehicle|integer_speed_kmh."""
    return f"{uid_to_vehicle(uid)}|{uid_to_speed(uid)}"


def resolve_uid_path(dataset_root: str | Path, uid: str) -> Path:
    rel = normalize_uid(uid)
    path = Path(dataset_root) / rel
    if not path.is_file():
        raise FileNotFoundError(f"Missing wav for {rel}: {path}")
    return path


def _parse_cpa(txt_path: Path) -> float | None:
    if not txt_path.is_file():
        return None
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    try:
        if len(lines) == 1:
            parts = lines[0].split()
            return float(parts[1]) if len(parts) >= 2 else None
        return float(lines[1])
    except (ValueError, IndexError):
        return None


def _load_official_split(vehicle_dir: Path) -> dict[str, str]:
    """Map stem -> 'train'|'valid' from Train_valid_split.txt if present."""
    path = vehicle_dir / "Train_valid_split.txt"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        stem, role = parts[0], parts[1].lower()
        if role in ("train", "valid"):
            out[stem] = role
    return out


def discover_clips(
    dataset_root: str | Path,
    *,
    domain: str = "real",
    require_annotations: bool = False,
) -> list[ClipRef]:
    """Scan ``<root>/<Vehicle>/*.wav`` → ClipRef list (sorted by uid)."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    clips: list[ClipRef] = []
    for vehicle_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        vehicle = vehicle_dir.name
        official = _load_official_split(vehicle_dir)
        for wav in sorted(vehicle_dir.glob("*.wav")):
            if wav.stem in _SKIP_STEMS:
                continue
            try:
                uid = normalize_uid(f"{vehicle}/{wav.name}")
                speed = uid_to_speed(uid)
            except ValueError:
                continue
            txt = wav.with_suffix(".txt")
            cpa = _parse_cpa(txt)
            if require_annotations and cpa is None:
                continue
            clips.append(
                ClipRef(
                    uid=uid,
                    vehicle=vehicle,
                    speed_kmh=speed,
                    wav_path=wav.resolve(),
                    domain=domain,
                    cpa_time_s=cpa,
                    official_split=official.get(wav.stem),
                )
            )
    if not clips:
        raise FileNotFoundError(f"No wavs under {root}")
    return clips


def discover_uids(dataset_root: str | Path) -> list[str]:
    return [c.uid for c in discover_clips(dataset_root)]


def clips_by_uid(clips: Iterable[ClipRef]) -> dict[str, ClipRef]:
    return {c.uid: c for c in clips}
