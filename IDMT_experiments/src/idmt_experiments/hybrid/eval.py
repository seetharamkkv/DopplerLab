"""Evaluation for hybrid direction models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from idmt_experiments.config import DEFAULT_OUTPUT_DIR, hybrid_checkpoint_subdir, resolve_class_labels
from idmt_experiments.cnn.metrics import channel_swap_consistency, classification_metrics
from idmt_experiments.cnn.train import resolve_device
from idmt_experiments.hybrid.dataset import collate_batch, make_dataset
from idmt_experiments.hybrid.train import load_checkpoint
from idmt_experiments.src.preprocess import ClipRecord, filter_records, resolve_data_dir
from idmt_experiments.src.splits import build_split


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Requires PyTorch.") from exc
    return torch, DataLoader


def predict_records(
    model,
    records: list[ClipRecord],
    cfg,
    norm_stats,
    physics_scaler,
    device: str,
    *,
    swap_channels: bool = False,
    time_reverse: bool = False,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    torch, DataLoader = _require_torch()
    device = resolve_device(device)
    model.eval()

    direction_cfg = cfg.to_direction_config()
    records = filter_records(records, direction_cfg)
    if progress_desc is None:
        if time_reverse:
            progress_desc = "time-reverse hybrid"
        else:
            progress_desc = f"{'swap' if swap_channels else 'eval'} hybrid"
    ds = make_dataset(
        records,
        cfg,
        norm_stats,
        physics_scaler,
        swap_channels=swap_channels,
        time_reverse=time_reverse,
        show_progress=show_progress,
        desc=progress_desc,
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch
    )

    labels = resolve_class_labels(direction_cfg)
    rows: list[dict] = []
    with torch.no_grad():
        for x_mel, x_phys, y, metas in loader:
            x_mel = x_mel.to(device)
            x_phys = x_phys.to(device)
            logits = model(x_mel, x_phys)
            pred = logits.argmax(dim=1)
            for i, meta in enumerate(metas):
                rows.append(
                    {
                        "clip_id": meta["clip_id"],
                        "event_id": meta["event_id"],
                        "location": meta["location"],
                        "vehicle": meta["vehicle"],
                        "weather": meta.get("weather", ""),
                        "travel_direction": meta["travel_direction"],
                        "split": meta["split"],
                        "y_true": int(y[i].item()),
                        "y_pred": int(pred[i].item()),
                        "label_true": labels[int(y[i].item())],
                        "label_pred": labels[int(pred[i].item())],
                        "correct": int(y[i].item()) == int(pred[i].item()),
                    }
                )
    return pd.DataFrame(rows)


def _vehicle_only_metrics(preds: pd.DataFrame, cfg) -> dict:
    direction_cfg = cfg.to_direction_config()
    vehicle = preds[preds["label_true"].isin(["L2R", "R2L"])].copy()
    if vehicle.empty:
        return {}
    y_true = vehicle["y_true"].values
    y_pred = vehicle["y_pred"].values
    labels = resolve_class_labels(direction_cfg)[:2]
    m = classification_metrics(y_true, y_pred, labels=labels)
    return {
        "n_vehicle_clips": m["n_samples"],
        "vehicle_balanced_accuracy": m["balanced_accuracy"],
        "vehicle_macro_f1": m["macro_f1"],
        "vehicle_per_class_recall": m["per_class_recall"],
        "vehicle_per_class_precision": m["per_class_precision"],
    }


def _write_eval_report(
    out_dir: Path,
    preds: pd.DataFrame,
    metrics: dict,
    cfg,
    checkpoint: Path,
    *,
    swap_metrics: dict | None = None,
    vehicle_metrics: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "eval_predictions.csv", index=False)
    payload = {**metrics}
    if vehicle_metrics:
        payload["vehicle_only"] = vehicle_metrics
    if swap_metrics:
        payload["channel_swap"] = swap_metrics
    (out_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "=" * 72,
        f"IDMT {cfg.task.upper()} — HYBRID EVALUATION",
        "=" * 72,
        "",
        f"  Checkpoint   : {checkpoint}",
        f"  Split        : {cfg.split_name}",
        f"  Feature      : {cfg.feature_type} + {cfg.feature_set}",
        f"  Mono source  : {cfg.mono_source}",
        f"  Classes      : {cfg.n_classes}",
        "",
        f"  Accuracy          : {metrics['accuracy']:.4f}",
        f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}",
        f"  Macro F1          : {metrics['macro_f1']:.4f}",
    ]
    if vehicle_metrics:
        lines.extend(
            [
                "",
                f"  Vehicle-only bal. acc : {vehicle_metrics['vehicle_balanced_accuracy']:.4f}",
                f"  Vehicle-only macro F1 : {vehicle_metrics['vehicle_macro_f1']:.4f}",
            ]
        )
    lines.append("")
    (out_dir / "eval_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_eval(
    *,
    run_dir: Path | None = None,
    checkpoint: Path | None = None,
    data_dir=None,
    output_dir: Path | None = None,
    device: str = "auto",
    split: str = "test",
    run_swap_test: bool = True,
) -> Path:
    data_dir = resolve_data_dir(data_dir)

    if checkpoint is None:
        if run_dir is None:
            raise ValueError("Provide checkpoint or run_dir")
        checkpoint = Path(run_dir) / "best.pt"

    checkpoint = Path(checkpoint)
    model, cfg, norm_stats, physics_scaler, _ckpt_meta = load_checkpoint(
        checkpoint, resolve_device(device)
    )

    train_records, val_records, test_records, _meta = build_split(
        cfg.split_name,
        data_dir,
        mic_filter=cfg.mic_filter,
        channel_filter=cfg.channel_filter,
        val_fraction=cfg.val_fraction,
        seed=cfg.split_seed,
    )

    if split == "test":
        eval_records = test_records
    elif split == "valid":
        eval_records = val_records
    elif split == "train":
        eval_records = train_records
    else:
        raise ValueError(f"Unknown split: {split}")

    eval_records = filter_records(eval_records, cfg.to_direction_config())
    print(
        f"  hybrid eval  split={cfg.split_name}/{split}  clips={len(eval_records)}  "
        f"device={resolve_device(device)}"
    )

    preds = predict_records(
        model,
        eval_records,
        cfg,
        norm_stats,
        physics_scaler,
        device,
        progress_desc=f"eval {split} hybrid",
    )
    direction_cfg = cfg.to_direction_config()
    metrics = classification_metrics(
        preds["y_true"].values,
        preds["y_pred"].values,
        labels=resolve_class_labels(direction_cfg),
    )
    vehicle_metrics = _vehicle_only_metrics(preds, cfg)

    swap_metrics = None
    if run_swap_test and cfg.task == "direction" and cfg.mono_source in ("left", "right"):
        vehicle_records = [r for r in eval_records if not r.is_background]
        if vehicle_records:
            pred_orig = predict_records(
                model,
                vehicle_records,
                cfg,
                norm_stats,
                physics_scaler,
                device,
                swap_channels=False,
                progress_desc="swap orig hybrid",
            )
            pred_swap = predict_records(
                model,
                vehicle_records,
                cfg,
                norm_stats,
                physics_scaler,
                device,
                swap_channels=True,
                progress_desc="swap flipped hybrid",
            )
            swap_metrics = channel_swap_consistency(
                pred_orig["y_pred"].values,
                pred_swap["y_pred"].values,
                n_classes=cfg.n_classes,
            )

    run_name = checkpoint.parent.name
    out_dir = (
        Path(output_dir or DEFAULT_OUTPUT_DIR)
        / hybrid_checkpoint_subdir(cfg)
        / run_name
    )
    _write_eval_report(
        out_dir,
        preds,
        metrics,
        cfg,
        checkpoint,
        swap_metrics=swap_metrics,
        vehicle_metrics=vehicle_metrics,
    )
    print(f"  -> {out_dir / 'eval_metrics.json'}")
    if vehicle_metrics:
        print(
            f"  vehicle-only bal. acc: {vehicle_metrics['vehicle_balanced_accuracy']:.4f}"
        )
    return out_dir
