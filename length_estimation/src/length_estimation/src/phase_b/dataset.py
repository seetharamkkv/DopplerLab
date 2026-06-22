"""PyTorch dataset: CPA-centred pass-by spectrogram -> vehicle length (m)."""

from __future__ import annotations

from typing import Literal

import librosa
import numpy as np

from length_estimation.config import PhaseBConfig, StftConfig
from length_estimation.src.preprocess import ClipRecord, align_and_crop, load_audio
from length_estimation.src.spectrograms import compute_log_mel, compute_synchrosqueezed

SpecType = Literal["mel", "ssq"]


def _require_torch():
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc:
        raise ImportError(
            "Phase B requires PyTorch. Install: pip install -r length_estimation/requirements.txt"
        ) from exc
    return torch, Dataset


def pad_or_crop_time(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad/center-crop along time axis (last dimension)."""
    n_frames = spec.shape[-1]
    if n_frames == target_frames:
        return spec
    if n_frames > target_frames:
        start = (n_frames - target_frames) // 2
        return spec[..., start : start + target_frames]
    pad_total = target_frames - n_frames
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(spec, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=spec.min())


def load_length_spec(
    record: ClipRecord,
    cfg: PhaseBConfig,
    stft_cfg: StftConfig | None = None,
) -> np.ndarray:
    """Load wav, CPA-align, return normalised (n_mels, T) float32 spectrogram."""
    stft_cfg = stft_cfg or StftConfig()
    y, sr = load_audio(record.wav_path)
    y, _ = align_and_crop(y, sr, record.cpa_time_s)

    if cfg.spec_type == "mel":
        spec, _, _ = compute_log_mel(y, sr, stft_cfg, n_mels=cfg.n_mels)
    else:
        power, _, _ = compute_synchrosqueezed(y, sr, stft_cfg.fmax_hz)
        spec = librosa.power_to_db(power, ref=np.max)

    spec = pad_or_crop_time(spec.astype(np.float32), cfg.target_time_frames)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec


def normalise_speed_kmh(speed_kmh: float, cfg: PhaseBConfig) -> float:
    return float(np.clip(speed_kmh / cfg.speed_kmh_max, 0.0, 1.0))


def make_dataset(records: list[ClipRecord], cfg: PhaseBConfig):
    torch, Dataset = _require_torch()

    class PassByLengthDataset(Dataset):
        def __init__(self, recs: list[ClipRecord], config: PhaseBConfig) -> None:
            self.records = recs
            self.cfg = config
            self.stft = StftConfig()

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int):
            record = self.records[idx]
            spec = load_length_spec(record, self.cfg, self.stft)
            x = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, T)
            y = torch.tensor(record.length_m, dtype=torch.float32)
            speed = (
                torch.tensor(normalise_speed_kmh(record.speed_kmh, self.cfg), dtype=torch.float32)
                if self.cfg.include_speed
                else None
            )
            meta = {
                "clip_id": record.clip_id,
                "vehicle": record.vehicle,
                "speed_kmh": record.speed_kmh,
                "split": record.split,
            }
            return x, y, speed, meta

    return PassByLengthDataset(records, cfg)


def collate_batch(batch):
    torch, _ = _require_torch()
    xs, ys, speeds, metas = zip(*batch)
    x = torch.stack(xs)
    y = torch.stack(ys)
    speed = torch.stack(speeds) if speeds[0] is not None else None
    return x, y, speed, list(metas)
