"""Phase B eval for transfer models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from idmt_experiments.config import DEFAULT_OUTPUT_DIR, DirectionConfig, NormStats, resolve_class_labels
from idmt_experiments.cnn.dataset import collate_batch, make_dataset
from idmt_experiments.cnn.metrics import classification_metrics, direction_intervention_flip
from idmt_experiments.cnn.train import resolve_device
from idmt_experiments.src.preprocess import filter_records, resolve_data_dir
from idmt_experiments.src.splits import build_split
from idmt_experiments.transfer.model import build_model


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Requires PyTorch.") from exc
    return torch, DataLoader


def load_transfer_checkpoint(path: Path, device: str = "cpu"):
    torch, _ = _require_torch()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = DirectionConfig.from_dict(ckpt.get("config", {}))
    backbone = ckpt.get("backbone", "deep_mel_cnn")
    model = build_model(cfg, backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    norm_stats = NormStats.from_dict(ckpt.get("norm_stats"))
    return model, cfg, norm_stats, ckpt


def _wav_to_logits(
    wav_path: Path,
    model,
    cfg: DirectionConfig,
    norm_stats: NormStats,
    device: str,
) -> tuple[np.ndarray, dict]:
    """Forward one VS13/IDMT wav; returns logits (n_classes,) and metadata."""
    torch, _ = _require_torch()
    from idmt_experiments.src.features import extract_feature, feature_to_batch_tensor, load_stereo, normalize_feature, select_mono_waveform

    y, sr = load_stereo(wav_path)
    y = select_mono_waveform(y, cfg.mono_source) if y.ndim == 2 else y
    duration_s = float(len(y) / sr) if sr > 0 else 0.0
    feat = extract_feature(
        y, sr, cfg.feature_type, n_mels=cfg.n_mels, mono_source=cfg.mono_source
    )
    feat = normalize_feature(feat, norm_stats, cfg.feature_type)
    x_t = feature_to_batch_tensor(feat)
    dev = resolve_device(device)
    model.to(dev)
    with torch.no_grad():
        logits = model(x_t.to(dev)).cpu().numpy()[0]
    return logits, {"duration_s": duration_s, "n_mel_frames": int(feat.shape[1])}


def predict_wav_transfer(
    wav_path: Path | str,
    checkpoint: Path | str,
    *,
    device: str = "auto",
) -> dict:
    """Single-clip 2-class direction inference for Phase B transfer checkpoints."""
    from idmt_experiments.config import resolve_class_labels

    wav_path = Path(wav_path)
    checkpoint = Path(checkpoint)
    model, cfg, norm_stats, _ = load_transfer_checkpoint(checkpoint, resolve_device(device))
    labels = resolve_class_labels(cfg)
    logits, meta = _wav_to_logits(wav_path, model, cfg, norm_stats, device)
    prob = np.exp(logits - logits.max())
    prob = prob / prob.sum()
    pred = int(logits.argmax())
    return {
        "wav_path": str(wav_path),
        "pred_label": labels[pred],
        "pred_index": pred,
        "probabilities": {labels[i]: float(prob[i]) for i in range(len(labels))},
        "logits": logits,
        "mono_source": cfg.mono_source,
        "duration_s": meta["duration_s"],
        "n_mel_frames": meta["n_mel_frames"],
        "checkpoint": str(checkpoint),
    }


def predict_wav_fusion(
    wav_path: Path | str,
    left_checkpoint: Path | str,
    right_checkpoint: Path | str,
    *,
    w_left: float = 0.5,
    device: str = "auto",
) -> dict:
    """Late-fusion direction prediction (Phase C) for one wav."""
    from idmt_experiments.config import resolve_class_labels

    left_r = predict_wav_transfer(wav_path, left_checkpoint, device=device)
    right_r = predict_wav_transfer(wav_path, right_checkpoint, device=device)
    labels = list(left_r["probabilities"].keys())
    logits = w_left * left_r["logits"] + (1.0 - w_left) * right_r["logits"]
    prob = np.exp(logits - logits.max())
    prob = prob / prob.sum()
    pred = int(logits.argmax())
    return {
        "wav_path": str(wav_path),
        "pred_label": labels[pred],
        "pred_index": pred,
        "probabilities": {labels[i]: float(prob[i]) for i in range(len(labels))},
        "fusion_w_left": w_left,
        "fusion_w_right": 1.0 - w_left,
        "duration_s": max(left_r["duration_s"], right_r["duration_s"]),
        "n_mel_frames": max(left_r["n_mel_frames"], right_r["n_mel_frames"]),
        "left_checkpoint": str(left_checkpoint),
        "right_checkpoint": str(right_checkpoint),
    }


def predict_logits(model, records, cfg, norm_stats, device, *, time_reverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    torch, DataLoader = _require_torch()
    device = resolve_device(device)
    model.eval()
    ds = make_dataset(records, cfg, norm_stats, time_reverse=time_reverse, show_progress=True)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)
    logits_list: list[np.ndarray] = []
    ys: list[int] = []
    with torch.no_grad():
        for x, y, _ in loader:
            logits_list.append(model(x.to(device)).cpu().numpy())
            ys.extend(y.numpy().tolist())
    return np.concatenate(logits_list, axis=0), np.asarray(ys, dtype=np.int64)


def run_eval(
    checkpoint: Path,
    *,
    data_dir=None,
    output_subdir: str,
    device: str = "auto",
    split: str = "test",
    run_flip_test: bool = True,
) -> Path:
    data_dir = resolve_data_dir(data_dir)
    checkpoint = Path(checkpoint)
    model, cfg, norm_stats, _ = load_transfer_checkpoint(checkpoint, resolve_device(device))

    train_records, val_records, test_records, _ = build_split(
        cfg.split_name, data_dir,
        mic_filter=cfg.mic_filter, channel_filter=cfg.channel_filter,
        val_fraction=cfg.val_fraction, seed=cfg.split_seed,
    )
    eval_records = {"test": test_records, "valid": val_records, "train": train_records}[split]
    eval_records = filter_records(eval_records, cfg)

    logits, y_true = predict_logits(model, eval_records, cfg, norm_stats, device)
    y_pred = logits.argmax(axis=1)
    labels = resolve_class_labels(cfg)
    metrics = classification_metrics(y_true, y_pred, labels=labels)

    flip_report = None
    if run_flip_test and cfg.task == "direction":
        logits_rev, y_true_rev = predict_logits(
            model, eval_records, cfg, norm_stats, device, time_reverse=True
        )
        pred_rev = logits_rev.argmax(axis=1)
        pred_base = y_pred
        flip_report = direction_intervention_flip(
            y_true_rev, pred_rev, pred_base, n_classes=cfg.n_classes
        )
        metrics["flip_agreement"] = flip_report.get("flip_agreement")
        metrics["flip_consistency"] = flip_report.get("flip_consistency")

    run_name = checkpoint.parent.name
    out_dir = Path(DEFAULT_OUTPUT_DIR) / output_subdir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(out_dir / "eval_predictions.csv", index=False)
    print(f"  bal_acc={metrics['balanced_accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
    if flip_report and flip_report.get("flip_agreement") is not None:
        print(f"  flip_agreement={flip_report['flip_agreement']:.4f}")
    return out_dir
