"""Train-only mel spectrogram statistics (no val/test leakage)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import AudioConfig


@dataclass
class MelStats:
    mean: np.ndarray  # (n_mels, 1)
    std: np.ndarray  # (n_mels, 1)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": self.std.astype(np.float32).tolist(),
        }


def save_mel_stats(path: str | Path, stats: MelStats) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f)
    return path


def load_mel_stats(path: str | Path) -> MelStats:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MelStats(
        mean=np.asarray(data["mean"], dtype=np.float32),
        std=np.asarray(data["std"], dtype=np.float32),
    )


def _mel_db(path: str | Path, cfg: AudioConfig) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=cfg.sample_rate, mono=True)
    n = cfg.audio_length_samples
    if len(audio) > n:
        audio = audio[:n]
    else:
        audio = np.pad(audio, (0, n - len(audio)), "constant")
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
    )
    return librosa.power_to_db(mel, ref=np.max)


def compute_mel_stats(
    audio_paths: Sequence[str | Path],
    *,
    cfg: AudioConfig | None = None,
    save_path: str | Path | None = None,
    verbose: bool = True,
) -> MelStats:
    """Fit per-mel mean/std on *these* paths only (pass fit_uids paths)."""
    cfg = cfg or AudioConfig()
    if not audio_paths:
        raise ValueError("audio_paths is empty — cannot fit mel stats")

    mel_sums = np.zeros((cfg.n_mels, 1), dtype=np.float64)
    mel_sum_sqs = np.zeros((cfg.n_mels, 1), dtype=np.float64)
    total_frames = 0

    for i, path in enumerate(audio_paths):
        if verbose and i % 50 == 0:
            print(f"Mel stats {i}/{len(audio_paths)}...")
        try:
            mel_db = _mel_db(path, cfg)
        except Exception as exc:  # noqa: BLE001 — keep going; report path
            print(f"WARN skip {path}: {exc}")
            continue
        mel_sums += np.sum(mel_db, axis=1, keepdims=True)
        mel_sum_sqs += np.sum(mel_db**2, axis=1, keepdims=True)
        total_frames += mel_db.shape[1]

    if total_frames == 0:
        raise RuntimeError("No frames accumulated for mel stats")

    mean = (mel_sums / total_frames).astype(np.float32)
    var = mel_sum_sqs / total_frames - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    std[std < 1e-8] = 1e-8
    stats = MelStats(mean=mean, std=std)
    if save_path:
        save_mel_stats(save_path, stats)
    return stats
