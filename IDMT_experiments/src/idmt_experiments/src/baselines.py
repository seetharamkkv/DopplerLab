"""Classical baselines and day-1 exploration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from idmt_experiments.config import DEFAULT_OUTPUT_DIR, DirectionConfig, resolve_class_labels
from idmt_experiments.src.direction.metrics import channel_swap_consistency, classification_metrics
from idmt_experiments.src.features import extract_feature, fit_norm_stats, load_stereo, normalize_feature, swap_stereo_channels
from idmt_experiments.src.preprocess import ClipRecord, clip_label, filter_records
from idmt_experiments.src.splits import build_split, verify_no_event_leakage


def _log(msg: str) -> None:
    print(msg, flush=True)


def _records_to_xy(
    records: list[ClipRecord],
    cfg: DirectionConfig,
    norm_stats,
    *,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for rec in tqdm(records, desc=desc, leave=False):
        y_audio, sr = load_stereo(rec.wav_path)
        feat = extract_feature(y_audio, sr, cfg.feature_type, n_mels=cfg.n_mels)
        feat = normalize_feature(feat, norm_stats, cfg.feature_type)
        xs.append(feat.reshape(-1))
        ys.append(clip_label(rec, cfg))
    return np.stack(xs), np.array(ys, dtype=int)


def run_classical_baseline(
    data_dir=None,
    *,
    task: str = "direction",
    feature_type: str = "cc",
    n_classes: int | None = None,
    split_name: str = "eusipco",
    output_dir: Path | None = None,
) -> dict:
    if n_classes is None:
        n_classes = 5 if task == "vehicle" else 3
    cfg = DirectionConfig(task=task, feature_type=feature_type, n_classes=n_classes, split_name=split_name)
    _log(f"Classical baseline — task={task}, feature={feature_type}, classes={n_classes}, split={split_name}")
    _log("  Loading split...")

    train_records, val_records, test_records, meta = build_split(
        split_name,
        data_dir,
        mic_filter=cfg.mic_filter,
        channel_filter=cfg.channel_filter,
        val_fraction=cfg.val_fraction,
        seed=cfg.split_seed,
    )

    train_records = filter_records(train_records, cfg)
    val_records = filter_records(val_records, cfg)
    test_records = filter_records(test_records, cfg)

    audit = verify_no_event_leakage(train_records, val_records, test_records)
    if not audit["ok"]:
        raise RuntimeError(f"Split leakage: {audit}")

    _log(
        f"  Split: {len(train_records)} train / {len(val_records)} val / {len(test_records)} test clips"
    )
    _log(f"  Fitting normalization on up to {cfg.norm_fit_max_samples or 'all'} train clips...")
    norm_stats = fit_norm_stats(train_records, cfg, show_progress=True)

    _log("  Extracting train features (this takes several minutes on CPU)...")
    x_train, y_train = _records_to_xy(train_records, cfg, norm_stats, desc="train features")

    _log("  Extracting test features...")
    x_test, y_test = _records_to_xy(test_records, cfg, norm_stats, desc="test features")

    _log("  Training logistic regression...")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    labels = resolve_class_labels(cfg)
    metrics = classification_metrics(y_test, y_pred, labels=labels)

    swap_metrics = None
    if task == "direction" and n_classes >= 2 and feature_type in ("cc", "stereo_mel"):
        vehicle_test = [r for r in test_records if not r.is_background]
        if vehicle_test:
            _log(f"  Channel-swap check on {len(vehicle_test)} vehicle test clips...")
            orig_pred, swap_pred = [], []
            for rec in tqdm(vehicle_test, desc="swap check", leave=False):
                y_a, sr = load_stereo(rec.wav_path)
                f0 = normalize_feature(
                    extract_feature(y_a, sr, cfg.feature_type, n_mels=cfg.n_mels),
                    norm_stats,
                    cfg.feature_type,
                )
                f1 = normalize_feature(
                    extract_feature(swap_stereo_channels(y_a), sr, cfg.feature_type, n_mels=cfg.n_mels),
                    norm_stats,
                    cfg.feature_type,
                )
                orig_pred.append(clf.predict(scaler.transform(f0.reshape(1, -1)))[0])
                swap_pred.append(clf.predict(scaler.transform(f1.reshape(1, -1)))[0])
            swap_metrics = channel_swap_consistency(np.array(orig_pred), np.array(swap_pred), n_classes=n_classes)

    out_dir = (output_dir or DEFAULT_OUTPUT_DIR) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "logistic_regression",
        "task": task,
        "feature_type": feature_type,
        "n_classes": n_classes,
        "split_name": split_name,
        "metrics": metrics,
        "channel_swap": swap_metrics,
        "split_audit": audit,
        "split_meta": meta,
    }
    out_path = out_dir / f"classical_{task}_{feature_type}_{n_classes}class.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"  Wrote {out_path}")
    return result
