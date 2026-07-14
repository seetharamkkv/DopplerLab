"""PyTorch dataset for IDMT direction classification.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Critical path for published monoaural metrics (vehicle bal. acc 81.5% / 79.3% / 73.6%).
Do not change default behaviour, numerics, or evaluation outputs without re-benchmarking
all three reference runs. Refactoring for maintainability is OK only if metrics stay
bit-identical. New work: separate --run-name or new modules.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from idmt_experiments.config import DirectionConfig, NormStats, resolve_class_labels
from idmt_experiments.src.features import (
    extract_feature,
    load_stereo,
    normalize_feature,
    reverse_time_axis,
    swap_stereo_channels,
)
from idmt_experiments.src.preprocess import ClipRecord, clip_label, filter_records


def _require_torch():
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc:
        raise ImportError("Requires PyTorch.") from exc
    return torch, Dataset


@dataclass
class PrecomputedItem:
    x: np.ndarray
    y: int
    meta: dict


def precompute_batch(
    records: list[ClipRecord],
    cfg: DirectionConfig,
    norm_stats: NormStats | None,
    *,
    swap_channels: bool = False,
    time_reverse: bool = False,
    show_progress: bool = False,
    desc: str = "features",
) -> list[PrecomputedItem]:
    from tqdm import tqdm

    records = filter_records(records, cfg)
    labels = resolve_class_labels(cfg)
    it = tqdm(records, desc=desc, leave=True) if show_progress else records
    items: list[PrecomputedItem] = []
    for rec in it:
        y_audio, sr = load_stereo(rec.wav_path)
        if swap_channels:
            y_audio = swap_stereo_channels(y_audio)
        if time_reverse:
            y_audio = reverse_time_axis(y_audio)
        feat = extract_feature(
            y_audio, sr, cfg.feature_type, n_mels=cfg.n_mels, mono_source=cfg.mono_source
        )
        feat = normalize_feature(feat, norm_stats, cfg.feature_type)
        label = clip_label(rec, cfg)
        meta = {
            "clip_id": rec.clip_id,
            "event_id": rec.event_id,
            "location": rec.location,
            "vehicle": rec.vehicle,
            "weather": rec.weather,
            "travel_direction": rec.travel_direction,
            "split": rec.split,
            "label_name": labels[label],
            "mono_source": cfg.mono_source,
        }
        items.append(PrecomputedItem(x=feat, y=label, meta=meta))
    return items


def make_dataset(
    records: list[ClipRecord],
    cfg: DirectionConfig,
    norm_stats: NormStats | None = None,
    *,
    swap_channels: bool = False,
    time_reverse: bool = False,
    precomputed: list[PrecomputedItem] | None = None,
    show_progress: bool = False,
    desc: str = "features",
    augment: bool = False,
    time_mask_param: int = 20,
    freq_mask_param: int = 8,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
):
    torch, Dataset = _require_torch()

    class _DS(Dataset):
        def __init__(self) -> None:
            if precomputed is not None:
                self.items = precomputed
            else:
                self.items = precompute_batch(
                    records,
                    cfg,
                    norm_stats,
                    swap_channels=swap_channels,
                    time_reverse=time_reverse,
                    show_progress=show_progress,
                    desc=desc,
                )

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            item = self.items[idx]
            feat = item.x
            x = torch.from_numpy(feat).unsqueeze(0) if feat.ndim == 2 else torch.from_numpy(feat)
            if augment:
                from idmt_experiments.cnn.augment import apply_spec_augment

                x = apply_spec_augment(
                    x,
                    time_mask_param=time_mask_param,
                    freq_mask_param=freq_mask_param,
                    num_time_masks=num_time_masks,
                    num_freq_masks=num_freq_masks,
                )
            return x, torch.tensor(item.y, dtype=torch.long), item.meta

    return _DS()


def collate_batch(batch):
    torch, _ = _require_torch()
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)
