"""Evaluation for physics direction classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from idmt_experiments.cnn.metrics import classification_metrics
from idmt_experiments.config import (
    DEFAULT_OUTPUT_DIR,
    PHYSICS_DIRECTION_LABELS,
    PhysicsConfig,
    physics_checkpoint_subdir,
)
from idmt_experiments.physics.dataset import build_feature_batch
from idmt_experiments.physics.interventions import run_interventions, save_interventions
from idmt_experiments.physics.train import load_physics_model
from idmt_experiments.src.preprocess import filter_physics_records, resolve_data_dir
from idmt_experiments.src.splits import build_split


def _majority_baseline(y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"majority_class": None, "accuracy": None}
    counts = np.bincount(y, minlength=2)
    majority = int(np.argmax(counts))
    acc = float(np.mean(y == majority))
    return {
        "majority_class": PHYSICS_DIRECTION_LABELS[majority],
        "accuracy": acc,
        "balanced_accuracy": acc,
    }


def predict_records(
    records,
    cfg: PhysicsConfig,
    model,
) -> pd.DataFrame:
    batch = build_feature_batch(records, cfg, show_progress=True)
    if len(batch.y) == 0:
        return pd.DataFrame()
    y_pred = model.predict(batch.X)
    rows: list[dict] = []
    for rec, yt, yp in zip(batch.records, batch.y, y_pred):
        rows.append(
            {
                "clip_id": rec.clip_id,
                "event_id": rec.event_id,
                "location": rec.location,
                "vehicle": rec.vehicle,
                "weather": rec.weather,
                "travel_direction": rec.travel_direction,
                "split": rec.split,
                "y_true": int(yt),
                "y_pred": int(yp),
                "label_true": PHYSICS_DIRECTION_LABELS[int(yt)],
                "label_pred": PHYSICS_DIRECTION_LABELS[int(yp)],
                "correct": int(yt) == int(yp),
            }
        )
    return pd.DataFrame(rows)


def _write_eval_report(
    out_dir: Path,
    preds: pd.DataFrame,
    metrics: dict,
    cfg: PhysicsConfig,
    run_dir: Path,
    *,
    baseline_metrics: dict | None = None,
    interventions: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "eval_predictions.csv", index=False)
    payload = {**metrics}
    if baseline_metrics:
        payload["naive_baselines"] = baseline_metrics
    if interventions:
        payload["interventions"] = interventions
    (out_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "=" * 72,
        "IDMT DIRECTION — PHYSICS EVALUATION (L2R / R2L only)",
        "=" * 72,
        "",
        f"  Run dir      : {run_dir}",
        f"  Split        : {cfg.split_name}",
        f"  Mono source  : {cfg.mono_source}",
        f"  Feature set  : {cfg.feature_set}",
        f"  Classifier   : {cfg.classifier}",
        "",
        f"  Accuracy          : {metrics['accuracy']:.4f}",
        f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}",
        f"  Macro F1          : {metrics['macro_f1']:.4f}",
        "",
        "  Per-class recall / precision / F1:",
    ]
    for k in metrics.get("per_class_f1", {}):
        rec = metrics.get("per_class_recall", {}).get(k, float("nan"))
        prec = metrics.get("per_class_precision", {}).get(k, float("nan"))
        f1 = metrics["per_class_f1"][k]
        lines.append(f"    {k}: recall={rec:.4f}  precision={prec:.4f}  f1={f1:.4f}")
    if baseline_metrics:
        lines.extend(["", "  Naive baselines:"])
        for name, val in baseline_metrics.items():
            if isinstance(val, dict):
                lines.append(f"    {name}: {val}")
            else:
                lines.append(f"    {name}: {val:.4f}")
    if interventions:
        tr = interventions.get("time_reverse", {})
        if tr.get("flip_consistency") is not None:
            lines.append(
                f"\n  Time-reverse flip consistency : {tr['flip_consistency']:.4f} "
                f"({tr['n_correct_flips']}/{tr['n_checked']})  [vs flipped true label]"
            )
        if tr.get("flip_agreement") is not None:
            lines.append(
                f"  Time-reverse flip agreement   : {tr['flip_agreement']:.4f} "
                f"({tr['n_flipped']}/{tr['n_checked']})  [mechanism: decision reverses]"
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
                f"({cs['n_flipped']}/{cs['n_checked']})"
            )
    lines.append("")
    (out_dir / "eval_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_eval(
    *,
    run_dir: Path,
    data_dir=None,
    output_dir: Path | None = None,
    split: str = "test",
    run_intervention_tests: bool = False,
) -> Path:
    data_dir = resolve_data_dir(data_dir)
    run_dir = Path(run_dir)
    model, cfg, _schema = load_physics_model(run_dir)

    train_records, val_records, test_records, _meta = build_split(
        cfg.split_name,
        data_dir,
        mic_filter=cfg.mic_filter,
        channel_filter=cfg.channel_filter,
        val_fraction=cfg.val_fraction,
        seed=cfg.split_seed,
    )
    eval_records = {"train": train_records, "valid": val_records, "test": test_records}[split]
    eval_records = filter_physics_records(eval_records, cfg)

    preds = predict_records(eval_records, cfg, model)
    metrics = classification_metrics(
        preds["y_true"].values,
        preds["y_pred"].values,
        labels=PHYSICS_DIRECTION_LABELS,
    )
    baseline = _majority_baseline(preds["y_true"].values)

    interventions = None
    if run_intervention_tests:
        interventions = run_interventions(
            eval_records,
            cfg,
            lambda X: model.predict(X),
            mono_source=cfg.mono_source,
        )

    run_name = run_dir.name
    out_dir = output_dir or (DEFAULT_OUTPUT_DIR / "physics" / cfg.task / run_name)

    _write_eval_report(
        out_dir,
        preds,
        metrics,
        cfg,
        run_dir,
        baseline_metrics={"always_majority": baseline},
        interventions=interventions,
    )
    if interventions:
        save_interventions(interventions, out_dir / "interventions.json")

    print(f"Physics eval ({split}): acc={metrics['accuracy']:.4f} bal_acc={metrics['balanced_accuracy']:.4f}")
    print(f"  wrote -> {out_dir}")
    return out_dir
