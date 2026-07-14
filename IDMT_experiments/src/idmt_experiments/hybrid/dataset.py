"""Joint mel + physics dataset for hybrid direction models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from idmt_experiments.config import HybridConfig, NormStats, PhysicsScaler, resolve_class_labels
from idmt_experiments.physics.features import extract_physics_features, feature_names
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
class HybridPrecomputedItem:
    x_mel: np.ndarray
    x_physics: np.ndarray
    y: int
    meta: dict


def fit_physics_scaler(
    records: list[ClipRecord],
    cfg: HybridConfig,
    *,
    show_progress: bool = False,
) -> PhysicsScaler:
    physics_cfg = cfg.to_physics_config()
    names = feature_names(physics_cfg)
    direction_cfg = cfg.to_direction_config()
    records = filter_records(records, direction_cfg)

    iterator = records
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(records, desc="fit physics scaler", unit="clip")

    rows: list[list[float]] = []
    for rec in iterator:
        feats = extract_physics_features(rec, physics_cfg, mono_source=cfg.mono_source)
        rows.append([feats[n] for n in names])

    if not rows:
        return PhysicsScaler(mean=[0.0] * len(names), std=[1.0] * len(names), feature_names=list(names))

    x = np.asarray(rows, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return PhysicsScaler(mean=mean.tolist(), std=std.tolist(), feature_names=list(names))


def _scale_physics(vec: np.ndarray, scaler: PhysicsScaler | None) -> np.ndarray:
    if scaler is None:
        return vec.astype(np.float32)
    mean = np.asarray(scaler.mean, dtype=np.float32)
    std = np.asarray(scaler.std, dtype=np.float32)
    return ((vec - mean) / std).astype(np.float32)


def precompute_batch(
    records: list[ClipRecord],
    cfg: HybridConfig,
    norm_stats: NormStats | None,
    physics_scaler: PhysicsScaler | None,
    *,
    swap_channels: bool = False,
    time_reverse: bool = False,
    show_progress: bool = False,
    desc: str = "hybrid features",
) -> list[HybridPrecomputedItem]:
    from tqdm import tqdm

    direction_cfg = cfg.to_direction_config()
    physics_cfg = cfg.to_physics_config()
    records = filter_records(records, direction_cfg)
    labels = resolve_class_labels(direction_cfg)
    phys_names = feature_names(physics_cfg)

    it = tqdm(records, desc=desc, leave=True) if show_progress else records
    items: list[HybridPrecomputedItem] = []
    for rec in it:
        y_audio, sr = load_stereo(rec.wav_path)
        if swap_channels:
            y_audio = swap_stereo_channels(y_audio)
        if time_reverse:
            y_audio = reverse_time_axis(y_audio)

        mel = extract_feature(
            y_audio, sr, cfg.feature_type, n_mels=cfg.n_mels, mono_source=cfg.mono_source
        )
        mel = normalize_feature(mel, norm_stats, cfg.feature_type)

        phys_feats = extract_physics_features(
            rec,
            physics_cfg,
            mono_source=cfg.mono_source,
            time_reverse=time_reverse,
        )
        phys_vec = np.asarray([phys_feats[n] for n in phys_names], dtype=np.float32)
        phys_vec = _scale_physics(phys_vec, physics_scaler)

        label = clip_label(rec, direction_cfg)
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
            "feature_set": cfg.feature_set,
        }
        items.append(HybridPrecomputedItem(x_mel=mel, x_physics=phys_vec, y=label, meta=meta))
    return items


def make_dataset(
    records: list[ClipRecord],
    cfg: HybridConfig,
    norm_stats: NormStats | None = None,
    physics_scaler: PhysicsScaler | None = None,
    *,
    swap_channels: bool = False,
    time_reverse: bool = False,
    precomputed: list[HybridPrecomputedItem] | None = None,
    show_progress: bool = False,
    desc: str = "hybrid features",
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
                    physics_scaler,
                    swap_channels=swap_channels,
                    time_reverse=time_reverse,
                    show_progress=show_progress,
                    desc=desc,
                )

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            item = self.items[idx]
            feat = item.x_mel
            x_mel = torch.from_numpy(feat).unsqueeze(0) if feat.ndim == 2 else torch.from_numpy(feat)
            x_phys = torch.from_numpy(item.x_physics)
            return x_mel, x_phys, torch.tensor(item.y, dtype=torch.long), item.meta

    return _DS()


def collate_batch(batch):
    torch, _ = _require_torch()
    xs_mel, xs_phys, ys, metas = zip(*batch)
    return torch.stack(xs_mel), torch.stack(xs_phys), torch.stack(ys), list(metas)
