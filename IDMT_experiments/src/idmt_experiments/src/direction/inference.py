"""Single-clip inference for direction model."""

from __future__ import annotations

from pathlib import Path

from idmt_experiments.config import DIRECTION_LABELS
from idmt_experiments.src.direction.train import load_checkpoint, resolve_device
from idmt_experiments.src.features import extract_feature, load_stereo, normalize_feature, swap_stereo_channels


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Requires PyTorch.") from exc
    return torch


def predict_wav(
    wav_path: Path | str,
    checkpoint: Path | str,
    *,
    device: str = "auto",
    swap_channels: bool = False,
) -> dict:
    torch = _require_torch()
    wav_path = Path(wav_path)
    checkpoint = Path(checkpoint)
    model, cfg, norm_stats, ckpt = load_checkpoint(checkpoint, resolve_device(device))
    labels = DIRECTION_LABELS if cfg.n_classes == 3 else DIRECTION_LABELS[:2]

    y, sr = load_stereo(wav_path)
    if swap_channels:
        y = swap_stereo_channels(y)
    feat = extract_feature(y, sr, cfg.feature_type, n_mels=cfg.n_mels)
    feat = normalize_feature(feat, norm_stats, cfg.feature_type)
    if feat.ndim == 2:
        x_t = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
    else:
        x_t = torch.from_numpy(feat).unsqueeze(0).float()

    model.eval()
    dev = resolve_device(device)
    model.to(dev)
    with torch.no_grad():
        logits = model(x_t.to(dev))
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(logits.argmax(dim=1).item())

    return {
        "wav_path": str(wav_path),
        "pred_label": labels[pred],
        "pred_index": pred,
        "probabilities": {labels[i]: float(prob[i]) for i in range(len(labels))},
        "swap_channels": swap_channels,
        "checkpoint": str(checkpoint),
    }
