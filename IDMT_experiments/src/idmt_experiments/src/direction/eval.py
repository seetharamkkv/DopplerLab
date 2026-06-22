"""Evaluation for direction CNN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from idmt_experiments.config import DEFAULT_OUTPUT_DIR, DirectionConfig, checkpoint_subdir, resolve_class_labels
from idmt_experiments.src.direction.dataset import collate_batch, make_dataset
from idmt_experiments.src.direction.metrics import channel_swap_consistency, classification_metrics
from idmt_experiments.src.direction.train import load_checkpoint, resolve_device
from idmt_experiments.src.preprocess import ClipRecord, filter_records, resolve_data_dir
from idmt_experiments.src.splits import (
    _sanitize_location,
    build_location_loo_splits,
    build_split,
)


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
    cfg: DirectionConfig,
    norm_stats,
    device: str,
    *,
    swap_channels: bool = False,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    torch, DataLoader = _require_torch()
    device = resolve_device(device)
    model.eval()

    records = filter_records(records, cfg)
    if progress_desc is None:
        progress_desc = f"{'swap' if swap_channels else 'eval'} {cfg.feature_type}"
    ds = make_dataset(
        records,
        cfg,
        norm_stats,
        swap_channels=swap_channels,
        show_progress=show_progress,
        desc=progress_desc,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)

    labels = resolve_class_labels(cfg)
    rows: list[dict] = []
    with torch.no_grad():
        for x, y, metas in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            for i, meta in enumerate(metas):
                rows.append(
                    {
                        "clip_id": meta["clip_id"],
                        "event_id": meta["event_id"],
                        "location": meta["location"],
                        "vehicle": meta["vehicle"],
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


def _write_eval_report(
    out_dir: Path,
    preds: pd.DataFrame,
    metrics: dict,
    cfg: DirectionConfig,
    checkpoint: Path,
    *,
    swap_metrics: dict | None = None,
    ckpt_meta: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "eval_predictions.csv", index=False)
    payload = {**metrics}
    if swap_metrics:
        payload["channel_swap"] = swap_metrics
    (out_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "=" * 72,
        f"IDMT {cfg.task.upper()} — EVALUATION",
        "=" * 72,
        "",
        f"  Checkpoint   : {checkpoint}",
        f"  Feature      : {cfg.feature_type}",
        f"  Classes      : {cfg.n_classes}",
        "",
        f"  Accuracy     : {metrics['accuracy']:.4f}",
        f"  Macro F1     : {metrics['macro_f1']:.4f}",
        "",
        "  Per-class F1:",
    ]
    for k, v in metrics.get("per_class_f1", {}).items():
        lines.append(f"    {k}: {v:.4f}")
    if swap_metrics and swap_metrics.get("flip_consistency") is not None:
        lines.extend(
            [
                "",
                f"  Channel-swap flip consistency : {swap_metrics['flip_consistency']:.4f} "
                f"({swap_metrics['n_correct_flips']}/{swap_metrics['n_checked']})",
            ]
        )
    lines.append("")
    (out_dir / "eval_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _swap_enabled(cfg: DirectionConfig, run_swap_test: bool) -> bool:
    return (
        run_swap_test
        and cfg.task == "direction"
        and cfg.n_classes >= 2
        and cfg.feature_type in ("cc", "stereo_mel")
    )


def run_eval(
    *,
    checkpoint: Path | None = None,
    run_dir: Path | None = None,
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
    model, cfg, norm_stats, ckpt_meta = load_checkpoint(checkpoint, resolve_device(device))

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

    eval_records = filter_records(eval_records, cfg)

    if split == "test":
        tr_ids = {r.event_id for r in train_records}
        va_ids = {r.event_id for r in val_records}
        te_ids = {r.event_id for r in eval_records}
        leak = (tr_ids | va_ids) & te_ids
        if leak:
            raise RuntimeError(f"Test split shares {len(leak)} events with train/valid — leakage!")

    n_clips = len(eval_records)
    n_swap = sum(1 for r in eval_records if not r.is_background)
    swap_passes = 2 if _swap_enabled(cfg, run_swap_test) and n_swap else 0
    print(
        f"  task={cfg.task}  split={cfg.split_name}/{split}  clips={n_clips}  "
        f"feature={cfg.feature_type}  device={resolve_device(device)}"
    )
    if swap_passes:
        print(f"  channel-swap: {n_swap} vehicle clips x2 passes after classification")

    print(f"  Pass 1/{1 + swap_passes}: classifying {split} clips...")
    preds = predict_records(
        model,
        eval_records,
        cfg,
        norm_stats,
        device,
        progress_desc=f"eval {split} {cfg.feature_type}",
    )
    metrics = classification_metrics(
        preds["y_true"].values,
        preds["y_pred"].values,
        labels=resolve_class_labels(cfg),
    )

    swap_metrics = None
    if _swap_enabled(cfg, run_swap_test):
        vehicle_records = [r for r in eval_records if not r.is_background]
        if vehicle_records:
            print(f"  Pass 2/{1 + swap_passes}: channel-swap baseline ({len(vehicle_records)} vehicle clips)...")
            pred_orig = predict_records(
                model,
                vehicle_records,
                cfg,
                norm_stats,
                device,
                swap_channels=False,
                progress_desc=f"swap orig {cfg.feature_type}",
            )
            print(f"  Pass 3/{1 + swap_passes}: channel-swap flipped ({len(vehicle_records)} vehicle clips)...")
            pred_swap = predict_records(
                model,
                vehicle_records,
                cfg,
                norm_stats,
                device,
                swap_channels=True,
                progress_desc=f"swap flipped {cfg.feature_type}",
            )
            swap_metrics = channel_swap_consistency(
                pred_orig["y_pred"].values,
                pred_swap["y_pred"].values,
                n_classes=cfg.n_classes,
            )

    run_name = checkpoint.parent.name
    out_dir = (output_dir or DEFAULT_OUTPUT_DIR) / checkpoint_subdir(cfg) / run_name
    print("  Writing report...")
    _write_eval_report(out_dir, preds, metrics, cfg, checkpoint, swap_metrics=swap_metrics, ckpt_meta=ckpt_meta)
    print((out_dir / "eval_summary.txt").read_text(encoding="utf-8"))
    return out_dir


def run_eval_location_loo(
    *,
    run_dir: Path,
    data_dir=None,
    output_dir: Path | None = None,
    device: str = "auto",
    run_swap_test: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    folds = build_location_loo_splits(data_dir)
    all_preds: list[pd.DataFrame] = []

    for location, train_records, val_records, test_records, _meta in folds:
        fold_key = _sanitize_location(location)
        ckpt = run_dir / f"fold_{fold_key}.pt"
        if not ckpt.exists():
            print(f"  skip {location}: missing {ckpt.name}")
            continue
        model, cfg, norm_stats, _ = load_checkpoint(ckpt, resolve_device(device))
        test_records = filter_records(test_records, cfg)
        print(f"  fold {location}: {len(test_records)} test clips")
        preds = predict_records(
            model,
            test_records,
            cfg,
            norm_stats,
            device,
            progress_desc=f"loo {fold_key} {cfg.feature_type}",
        )
        preds["held_out_location"] = location
        preds["fold_checkpoint"] = str(ckpt)
        all_preds.append(preds)

    if not all_preds:
        raise RuntimeError("No fold checkpoints found for LOO eval")

    pooled = pd.concat(all_preds, ignore_index=True)
    _, cfg, _, _ = load_checkpoint(run_dir / f"fold_{_sanitize_location(folds[0][0])}.pt", "cpu")
    label_names = resolve_class_labels(cfg)
    metrics = classification_metrics(
        pooled["y_true"].values,
        pooled["y_pred"].values,
        labels=label_names,
    )

    out_dir = (output_dir or DEFAULT_OUTPUT_DIR) / checkpoint_subdir(cfg) / f"{run_dir.name}_loo_eval"
    _write_eval_report(out_dir, pooled, metrics, cfg, run_dir)
    print((out_dir / "eval_summary.txt").read_text(encoding="utf-8"))
    return out_dir
