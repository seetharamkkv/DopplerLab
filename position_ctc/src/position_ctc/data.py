"""Clip store, vehicle-disjoint splits, and random motion crops."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from position_ctc.features import N_MELS, features_from_wav

CROP_FRAMES = 320


@dataclass(frozen=True)
class Clip:
    clip_id: str
    vehicle: str
    wav_path: Path
    sample_dir: Path


def vehicle_name(wav_path: Path) -> str:
    stem = wav_path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def discover_clips(root: Path) -> list[Clip]:
    clips: list[Clip] = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sample_")):
        wavs = sorted(sample_dir.glob("*.wav"))
        state = sample_dir / "metadata" / "canonical_state_frames.npy"
        if not wavs or not state.is_file():
            continue
        clips.append(
            Clip(
                clip_id=sample_dir.name,
                vehicle=vehicle_name(wavs[0]),
                wav_path=wavs[0],
                sample_dir=sample_dir,
            )
        )
    if not clips:
        raise FileNotFoundError(f"no clips under {root}")
    return clips


def split_by_vehicle(clips: list[Clip], seed: int = 0) -> dict[str, list[int]]:
    """Hold out whole vehicles so timbre memorization cannot pass the test."""
    by_vehicle: dict[str, list[int]] = {}
    for i, clip in enumerate(clips):
        by_vehicle.setdefault(clip.vehicle, []).append(i)
    names = sorted(by_vehicle)
    rng = random.Random(seed)
    rng.shuffle(names)
    if len(names) < 4:
        ids = list(range(len(clips)))
        rng.shuffle(ids)
        n_test = max(1, int(round(0.15 * len(ids))))
        n_val = max(1, int(round(0.15 * len(ids))))
        return {"train": ids[n_test + n_val :], "val": ids[n_test : n_test + n_val], "test": ids[:n_test]}
    n_test_veh = 2 if len(names) >= 8 else 1
    n_val_veh = 2 if len(names) >= 8 else 1
    test_names = set(names[:n_test_veh])
    val_names = set(names[n_test_veh : n_test_veh + n_val_veh])
    splits = {"train": [], "val": [], "test": []}
    for name, idxs in by_vehicle.items():
        if name in test_names:
            splits["test"].extend(idxs)
        elif name in val_names:
            splits["val"].extend(idxs)
        else:
            splits["train"].extend(idxs)
    return splits


def trim_stationary(
    feat: np.ndarray,
    state: np.ndarray,
    thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the simulator's stationary pads so the first token is the moving velocity."""
    speed = np.hypot(state[:, 1], state[:, 3])
    moving = np.flatnonzero(speed > thresh)
    if moving.size == 0:
        return feat, state
    start, stop = int(moving[0]), int(moving[-1]) + 1
    return feat[:, start:stop], state[start:stop]


def load_arrays(clip: Clip) -> tuple[np.ndarray, np.ndarray]:
    feat = features_from_wav(clip.wav_path)
    state = np.load(clip.sample_dir / "metadata" / "canonical_state_frames.npy").astype(np.float32)
    n = min(feat.shape[1], state.shape[0])
    return trim_stationary(feat[:, :n], state[:n])


def augment_features(feat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Spectral tilt and a frequency mask. Gain is removed by later standardization."""
    out = feat.copy()
    mag = out[:N_MELS]
    slope = float(rng.uniform(-0.15, 0.15))
    ramp = np.linspace(-1.0, 1.0, mag.shape[0], dtype=np.float32)[:, None]
    mag += slope * ramp
    if rng.random() < 0.7:
        width = int(rng.integers(2, 8))
        start = int(rng.integers(0, max(1, mag.shape[0] - width)))
        mag[start : start + width] = mag.mean()
    out[:N_MELS] = mag
    out += rng.normal(0.0, 0.01, size=out.shape).astype(np.float32)
    return out


def standardize_mag(feat: np.ndarray) -> np.ndarray:
    out = feat.copy()
    mag = out[:N_MELS]
    out[:N_MELS] = (mag - mag.mean()) / (mag.std() + 1e-4)
    return out


def crop_pair(
    feat: np.ndarray,
    state: np.ndarray,
    rng: np.random.Generator,
    crop: int = CROP_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    n = feat.shape[1]
    if n <= crop:
        return feat, state
    start = int(rng.integers(0, n - crop + 1))
    return feat[:, start : start + crop], state[start : start + crop]
