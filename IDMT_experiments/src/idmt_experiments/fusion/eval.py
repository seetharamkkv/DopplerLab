"""Late fusion of left + right mono direction models (Phase C)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from idmt_experiments.config import DEFAULT_OUTPUT_DIR, DirectionConfig, NormStats, resolve_class_labels
from idmt_experiments.cnn.metrics import classification_metrics, direction_intervention_flip
from idmt_experiments.cnn.train import resolve_device
from idmt_experiments.src.preprocess import filter_records, resolve_data_dir
from idmt_experiments.src.splits import build_split
from idmt_experiments.transfer.eval import load_transfer_checkpoint, predict_logits


def _load_mono_checkpoint(path: Path, device: str, kind: str = "auto"):
    path = Path(path)
    if kind == "auto":
        kind = "transfer" if "transfer" in str(path).replace("\\", "/") else "cnn"
    if kind == "cnn":
        from idmt_experiments.cnn.train import load_checkpoint

        model, cfg, norm_stats, _ = load_checkpoint(path, device)
        return model, cfg, norm_stats, _
    return load_transfer_checkpoint(path, device)


def _predict_mono(model, records, cfg, norm_stats, device, *, time_reverse: bool = False):
    if hasattr(model, "conv") and not hasattr(model, "physics_mlp"):
        from idmt_experiments.transfer.eval import predict_logits

        return predict_logits(model, records, cfg, norm_stats, device, time_reverse=time_reverse)
    from idmt_experiments.transfer.eval import predict_logits

    return predict_logits(model, records, cfg, norm_stats, device, time_reverse=time_reverse)


def _search_fusion_weight(logits_l, logits_r, y_true) -> float:
    best_w = 0.5
    best_bal = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        pred = np.argmax(w * logits_l + (1.0 - w) * logits_r, axis=1)
        from sklearn.metrics import balanced_accuracy_score

        bal = balanced_accuracy_score(y_true, pred)
        if bal > best_bal:
            best_bal = bal
            best_w = float(w)
    return best_w


def run_fusion_eval(
    left_checkpoint: Path,
    right_checkpoint: Path,
    *,
    run_name: str = "fusion_2class_100ep",
    data_dir=None,
    device: str = "auto",
    split: str = "test",
    checkpoint_kind: str = "auto",
) -> Path:
    data_dir = resolve_data_dir(data_dir)
    device = resolve_device(device)

    model_l, cfg_l, norm_l, _ = _load_mono_checkpoint(left_checkpoint, device, checkpoint_kind)
    model_r, cfg_r, norm_r, _ = _load_mono_checkpoint(right_checkpoint, device, checkpoint_kind)
    if cfg_l.n_classes != cfg_r.n_classes:
        raise ValueError("Left/right checkpoints must share n_classes")

    vehicle_only = cfg_l.n_classes == 3
    logits_slice = slice(0, 2) if vehicle_only else slice(None)

    train_records, val_records, test_records, _ = build_split(
        cfg_l.split_name, data_dir,
        mic_filter=cfg_l.mic_filter, channel_filter=cfg_l.channel_filter,
        val_fraction=cfg_l.val_fraction, seed=cfg_l.split_seed,
    )
    val_records = filter_records(val_records, cfg_l)
    test_records = filter_records(test_records, cfg_l)

    logits_l_val, y_val = predict_logits(model_l, val_records, cfg_l, norm_l, device)
    logits_r_val, _ = predict_logits(model_r, val_records, cfg_r, norm_r, device)
    if vehicle_only:
        mask = y_val < 2
        logits_l_val, logits_r_val, y_val = logits_l_val[mask, :2], logits_r_val[mask, :2], y_val[mask]
    else:
        logits_l_val, logits_r_val = logits_l_val[:, logits_slice], logits_r_val[:, logits_slice]

    w_left = _search_fusion_weight(logits_l_val, logits_r_val, y_val)
    print(f"  fusion weight (left): {w_left:.3f}  (fit on valid)")

    logits_l_test, y_test = predict_logits(model_l, test_records, cfg_l, norm_l, device)
    logits_r_test, _ = predict_logits(model_r, test_records, cfg_r, norm_r, device)
    if vehicle_only:
        mask = y_test < 2
        logits_l_test, logits_r_test, y_test = logits_l_test[mask, :2], logits_r_test[mask, :2], y_test[mask]
    else:
        logits_l_test, logits_r_test = logits_l_test[:, logits_slice], logits_r_test[:, logits_slice]
    logits_fused = w_left * logits_l_test + (1.0 - w_left) * logits_r_test
    y_pred = logits_fused.argmax(axis=1)

    labels = ("L2R", "R2L")
    metrics = classification_metrics(y_test, y_pred, labels=labels)

    pred_l = logits_l_test.argmax(axis=1)
    pred_r = logits_r_test.argmax(axis=1)
    flip_cs = direction_intervention_flip(y_test, pred_r, pred_l, n_classes=2)
    metrics["channel_swap_agreement"] = flip_cs.get("flip_agreement")
    metrics["fusion"] = {"w_left": w_left, "w_right": 1.0 - w_left, "vehicle_only": vehicle_only}

    out_dir = Path(DEFAULT_OUTPUT_DIR) / "fusion" / "direction" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **metrics,
        "left_checkpoint": str(left_checkpoint),
        "right_checkpoint": str(right_checkpoint),
        "phase": "C",
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  test bal_acc={metrics['balanced_accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
    return out_dir
