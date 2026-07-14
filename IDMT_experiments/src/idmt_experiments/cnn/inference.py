"""Single-clip inference for direction model.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Critical path for published monoaural metrics (vehicle bal. acc 81.5% / 79.3% / 73.6%).
Do not change default behaviour, numerics, or evaluation outputs without re-benchmarking
all three reference runs. Refactoring for maintainability is OK only if metrics stay
bit-identical. New work: separate --run-name or new modules.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from idmt_experiments.config import DIRECTION_LABELS, resolve_class_labels
from idmt_experiments.cnn.train import load_checkpoint, resolve_device
from idmt_experiments.src.features import (
    extract_feature,
    load_mono,
    load_stereo,
    normalize_feature,
    select_mono_waveform,
    swap_stereo_channels,
)


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
    labels = resolve_class_labels(cfg)

    y, sr = load_stereo(wav_path)
    if swap_channels:
        y = swap_stereo_channels(y)
    feat = extract_feature(
        y, sr, cfg.feature_type, n_mels=cfg.n_mels, mono_source=cfg.mono_source
    )
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
        "mono_source": cfg.mono_source,
        "checkpoint": str(checkpoint),
    }


def predict_wav_mono(
    wav_path: Path | str,
    checkpoint: Path | str,
    *,
    device: str = "auto",
    y: np.ndarray | None = None,
    sr: int | None = None,
) -> dict:
    """Predict direction from a monoaural clip using a mel-trained checkpoint.

    The full waveform is used (no trimming or padding). The CNN ends with
    adaptive global pooling, so variable clip lengths only change the mel time axis.
    """
    torch = _require_torch()
    wav_path = Path(wav_path)
    checkpoint = Path(checkpoint)
    model, cfg, norm_stats, ckpt = load_checkpoint(checkpoint, resolve_device(device))
    if cfg.feature_type != "mel":
        raise ValueError(
            f"Mono batch inference requires a mel checkpoint (mono log-mel), got feature_type={cfg.feature_type!r}"
        )
    labels = resolve_class_labels(cfg)

    if cfg.mono_source in ("left", "right"):
        y, sr = load_stereo(wav_path)
        y = select_mono_waveform(y, cfg.mono_source)
    elif y is None or sr is None:
        y, sr = load_mono(wav_path)
    duration_s = float(len(y) / sr) if sr > 0 else 0.0
    feat = extract_feature(
        y, sr, cfg.feature_type, n_mels=cfg.n_mels, mono_source=cfg.mono_source
    )
    feat = normalize_feature(feat, norm_stats, cfg.feature_type)
    x_t = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()

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
        "mono": True,
        "mono_source": cfg.mono_source,
        "duration_s": duration_s,
        "n_mel_frames": int(feat.shape[1]),
        "checkpoint": str(checkpoint),
    }
