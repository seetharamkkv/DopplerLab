"""Evaluation for direction CNN.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Critical path for published monoaural metrics (vehicle bal. acc 81.5% / 79.3% / 73.6%).
Do not change default behaviour, numerics, or evaluation outputs without re-benchmarking
all three reference runs. Refactoring for maintainability is OK only if metrics stay
bit-identical. New work: separate --run-name or new modules.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from idmt_experiments.config import DEFAULT_OUTPUT_DIR, DirectionConfig, checkpoint_subdir, resolve_class_labels
from idmt_experiments.cnn.dataset import collate_batch, make_dataset
from idmt_experiments.cnn.metrics import channel_swap_consistency, classification_metrics
from idmt_experiments.cnn.interventions import run_interventions, save_interventions
from idmt_experiments.cnn.train import load_checkpoint, resolve_device
from idmt_experiments.src.preprocess import ClipRecord, filter_records, resolve_data_dir
from idmt_experiments.src.splits import (
    _sanitize_location,
    build_location_loo_splits,
    build_split,
)
from idmt_experiments.src.weather_audit import (
    _location_oracle_preds,
    _location_oracle_train_majority_preds,
    _majority_baseline,
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
    time_reverse: bool = False,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    torch, DataLoader = _require_torch()
    device = resolve_device(device)
    model.eval()

    records = filter_records(records, cfg)
    if progress_desc is None:
        if time_reverse:
            progress_desc = f"time-reverse {cfg.feature_type}"
        else:
            progress_desc = f"{'swap' if swap_channels else 'eval'} {cfg.feature_type}"
    ds = make_dataset(
        records,
        cfg,
        norm_stats,
        swap_channels=swap_channels,
        time_reverse=time_reverse,
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


def predict_labels(
    model,
    records: list[ClipRecord],
    cfg: DirectionConfig,
    norm_stats,
    device: str,
    *,
    swap_channels: bool = False,
    time_reverse: bool = False,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> np.ndarray:
    preds = predict_records(
        model,
        records,
        cfg,
        norm_stats,
        device,
        swap_channels=swap_channels,
        time_reverse=time_reverse,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    return preds["y_pred"].values.astype(np.int64)


def _subgroup_metrics(preds: pd.DataFrame, labels: tuple[str, ...]) -> dict:
    out: dict = {}
    for col in ("location", "vehicle"):
        if col not in preds.columns:
            continue
        groups = {}
        for key, grp in preds.groupby(col):
            if len(grp) == 0:
                continue
            m = classification_metrics(
                grp["y_true"].values,
                grp["y_pred"].values,
                labels=labels,
            )
            groups[str(key)] = {
                "n": m["n_samples"],
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "per_class_recall": m["per_class_recall"],
            }
        out[col] = groups
    return out


def _vehicle_only_metrics(preds: pd.DataFrame, cfg: DirectionConfig) -> dict:
    if cfg.task != "direction" or cfg.n_classes < 3:
        return {}
    vehicle = preds[preds["label_true"].isin(["L2R", "R2L"])].copy()
    if vehicle.empty:
        return {}
    labels = resolve_class_labels(cfg)[:2]
    m = classification_metrics(
        vehicle["y_true"].values,
        vehicle["y_pred"].values,
        labels=labels,
    )
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
    cfg: DirectionConfig,
    checkpoint: Path,
    *,
    swap_metrics: dict | None = None,
    ckpt_meta: dict | None = None,
    subgroup_metrics: dict | None = None,
    baseline_metrics: dict | None = None,
    interventions: dict | None = None,
    vehicle_metrics: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "eval_predictions.csv", index=False)
    payload = {**metrics}
    if vehicle_metrics:
        payload["vehicle_only"] = vehicle_metrics
    if swap_metrics:
        payload["channel_swap"] = swap_metrics
    if subgroup_metrics:
        payload["subgroup_metrics"] = subgroup_metrics
    if baseline_metrics:
        payload["naive_baselines"] = baseline_metrics
    if interventions:
        payload["interventions"] = interventions
    (out_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "=" * 72,
        f"IDMT {cfg.task.upper()} — EVALUATION",
        "=" * 72,
        "",
        f"  Checkpoint   : {checkpoint}",
        f"  Split        : {cfg.split_name}",
        f"  Feature      : {cfg.feature_type}",
        f"  Mono source  : {getattr(cfg, 'mono_source', 'mean')}",
        f"  Classes      : {cfg.n_classes}",
        "",
        f"  Accuracy          : {metrics['accuracy']:.4f}",
        f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}",
        f"  Macro F1          : {metrics['macro_f1']:.4f}",
        "",
    ]
    if vehicle_metrics:
        lines.extend(
            [
                f"  Vehicle-only bal. acc : {vehicle_metrics['vehicle_balanced_accuracy']:.4f}",
                f"  Vehicle-only macro F1 : {vehicle_metrics['vehicle_macro_f1']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
        "  Per-class recall / precision / F1:",
        ]
    )
    for k in metrics.get("per_class_f1", {}):
        rec = metrics.get("per_class_recall", {}).get(k, float("nan"))
        prec = metrics.get("per_class_precision", {}).get(k, float("nan"))
        f1 = metrics["per_class_f1"][k]
        lines.append(f"    {k}: recall={rec:.4f}  precision={prec:.4f}  f1={f1:.4f}")
    if baseline_metrics:
        lines.extend(["", "  Naive baselines (same test set):"])
        for name, val in baseline_metrics.items():
            lines.append(f"    {name}: {val:.4f}")
    if subgroup_metrics and cfg.task == "weather":
        lines.extend(["", "  By location:"])
        for loc, sm in subgroup_metrics.get("location", {}).items():
            lines.append(
                f"    {loc}: n={sm['n']} acc={sm['accuracy']:.4f} "
                f"bal_acc={sm['balanced_accuracy']:.4f} macro_f1={sm['macro_f1']:.4f}"
            )
    if swap_metrics and swap_metrics.get("flip_consistency") is not None:
        lines.extend(
            [
                "",
                f"  Channel-swap flip consistency : {swap_metrics['flip_consistency']:.4f} "
                f"({swap_metrics['n_correct_flips']}/{swap_metrics['n_checked']})",
            ]
        )
    if interventions:
        tr = interventions.get("time_reverse") or {}
        if tr.get("flip_consistency") is not None:
            lines.append(
                f"\n  Time-reverse flip consistency : {tr['flip_consistency']:.4f} "
                f"({tr['n_correct_flips']}/{tr['n_checked']})  [vs flipped true label]"
            )
        if tr.get("flip_agreement") is not None:
            lines.append(
                f"  Time-reverse flip agreement   : {tr['flip_agreement']:.4f} "
                f"({tr['n_flipped']}/{tr['n_agreement_checked']})  [mechanism]"
            )
        cs = interventions.get("channel_swap") or {}
        if cs.get("flip_consistency") is not None:
            lines.append(
                f"  Channel-swap flip consistency : {cs['flip_consistency']:.4f} "
                f"({cs['n_correct_flips']}/{cs['n_checked']})"
            )
        if cs.get("flip_agreement") is not None:
            lines.append(
                f"  Channel-swap flip agreement   : {cs['flip_agreement']:.4f} "
                f"({cs['n_flipped']}/{cs['n_agreement_checked']})"
            )
    lines.append("")
    (out_dir / "eval_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _swap_enabled(cfg: DirectionConfig, run_swap_test: bool) -> bool:
    if not run_swap_test or cfg.task != "direction" or cfg.n_classes < 2:
        return False
    if cfg.feature_type in ("cc", "stereo_mel"):
        return True
    # Left/right mel: swapping stereo channels changes which ear is used — diagnostic only.
    if cfg.feature_type in ("mel", "complex_stft") and cfg.mono_source in ("left", "right"):
        return True
    return False


def run_eval(
    *,
    checkpoint: Path | None = None,
    run_dir: Path | None = None,
    data_dir=None,
    output_dir: Path | None = None,
    device: str = "auto",
    split: str = "test",
    run_swap_test: bool = True,
    run_intervention_tests: bool = False,
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

    subgroup_metrics = None
    baseline_metrics = None
    if cfg.task == "weather":
        label_names = resolve_class_labels(cfg)
        subgroup_metrics = _subgroup_metrics(preds, label_names)
        y_true = preds["y_true"].values
        y_pred = preds["y_pred"].values
        baseline_metrics = {
            "always_dry": float(np.mean(y_true == 0)),
            "majority_class": _majority_baseline(y_true),
            "location_oracle_always_wet_at_site": float(
                np.mean(_location_oracle_preds(eval_records) == y_true)
            ),
            "location_oracle_train_majority_at_site": float(
                np.mean(
                    _location_oracle_train_majority_preds(
                        filter_records(train_records, cfg),
                        eval_records,
                    )
                    == y_true
                )
            ),
            "model_minus_location_oracle": float(
                metrics["accuracy"]
                - float(np.mean(_location_oracle_preds(eval_records) == y_true))
            ),
        }
        if cfg.split_name == "weather_stratified":
            print(
                "  WARNING: weather_stratified test mixes dry-only sites with wet site — "
                "accuracy is inflated by location confound. Prefer weather_site split."
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

    interventions_report = None
    if run_intervention_tests and cfg.task == "direction":
        vehicle_records = [r for r in eval_records if not r.is_background]
        if vehicle_records:
            print(f"  Interventions: time-reverse + channel-swap on {len(vehicle_records)} vehicle clips...")

            def _predict(recs, *, time_reverse=False, swap_channels=False):
                return predict_labels(
                    model,
                    recs,
                    cfg,
                    norm_stats,
                    device,
                    time_reverse=time_reverse,
                    swap_channels=swap_channels,
                    show_progress=True,
                )

            interventions_report = run_interventions(
                vehicle_records,
                cfg,
                _predict,
                vehicle_only=True,
            )

    run_name = checkpoint.parent.name
    out_dir = (output_dir or DEFAULT_OUTPUT_DIR) / checkpoint_subdir(cfg) / run_name
    vehicle_metrics = _vehicle_only_metrics(preds, cfg)
    print("  Writing report...")
    _write_eval_report(
        out_dir,
        preds,
        metrics,
        cfg,
        checkpoint,
        swap_metrics=swap_metrics,
        ckpt_meta=ckpt_meta,
        subgroup_metrics=subgroup_metrics,
        baseline_metrics=baseline_metrics,
        interventions=interventions_report,
        vehicle_metrics=vehicle_metrics,
    )
    if vehicle_metrics:
        print(
            f"  vehicle-only bal. acc: {vehicle_metrics['vehicle_balanced_accuracy']:.4f}",
            flush=True,
        )
    if interventions_report:
        save_interventions(interventions_report, out_dir / "interventions.json")
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
