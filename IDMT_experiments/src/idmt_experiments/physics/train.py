"""Train physics-informed direction classifier (logistic regression)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from idmt_experiments.cnn.metrics import classification_metrics
from idmt_experiments.config import (
    DEFAULT_CHECKPOINT_DIR,
    PHYSICS_DIRECTION_LABELS,
    PhysicsConfig,
    physics_checkpoint_subdir,
)
from idmt_experiments.physics.dataset import build_feature_batch
from idmt_experiments.src.preprocess import ClipRecord, filter_physics_records
from idmt_experiments.src.splits import (
    build_split,
    default_split_meta_path,
    persist_split_meta,
    verify_no_event_leakage,
)


_SUPPORTED_CLASSIFIERS = ("logistic", "logistic_antisym", "gbt", "mlp")


def _build_classifier(cfg: PhysicsConfig) -> Pipeline:
    if cfg.classifier not in _SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"Unsupported classifier: {cfg.classifier!r} "
            f"({' | '.join(_SUPPORTED_CLASSIFIERS)})"
        )

    if cfg.classifier in ("logistic", "logistic_antisym"):
        # logistic_antisym is the flip-enforcing "rule layer": with no mean-centering and
        # no intercept, a strictly antisymmetric feature vector x -> -x negates the
        # decision score, so the predicted direction reverses when the waveform reverses.
        antisym = cfg.classifier == "logistic_antisym"
        scaler = StandardScaler(with_mean=not antisym)
        clf = LogisticRegression(
            max_iter=cfg.max_iter,
            class_weight="balanced",
            random_state=cfg.split_seed,
            fit_intercept=not antisym,
        )
        return Pipeline([("scaler", scaler), ("clf", clf)])

    if cfg.classifier == "gbt":
        # Gradient-boosted trees: strongest option for tabular scalar physics features.
        # Captures nonlinear feature interactions logistic cannot. No scaling needed.
        clf = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=cfg.split_seed,
        )
        return Pipeline([("clf", clf)])

    # mlp: shallow nonlinear net as a second-shot comparison.
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=cfg.split_seed,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _feature_importance(model, cfg, feature_names, val_batch) -> dict:
    """Interpretability payload that works for linear and nonlinear heads.

    - Linear (logistic): signed coefficients + intercept.
    - Nonlinear (gbt/mlp): permutation importance on the validation split (magnitude of
      balanced-accuracy drop when a feature is shuffled). Keeps the physics-informed
      story regardless of the model class.
    """
    final = model.named_steps["clf"]
    names = list(feature_names)
    if hasattr(final, "coef_"):
        coef = final.coef_
        intercept = final.intercept_
        return {
            "method": "linear_coefficients",
            "coefficients": {n: float(coef[0, i]) for i, n in enumerate(names)},
            "intercept": float(intercept[0]),
            "classes": list(PHYSICS_DIRECTION_LABELS),
        }

    perm = {n: 0.0 for n in names}
    if len(val_batch.y):
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model,
            val_batch.X,
            val_batch.y,
            scoring="balanced_accuracy",
            n_repeats=10,
            random_state=cfg.split_seed,
        )
        perm = {n: float(result.importances_mean[i]) for i, n in enumerate(names)}
    return {
        "method": "permutation_importance",
        "scoring": "balanced_accuracy",
        "importances": perm,
        "classes": list(PHYSICS_DIRECTION_LABELS),
    }


def train_on_records(
    train_records: list[ClipRecord],
    val_records: list[ClipRecord],
    cfg: PhysicsConfig,
    checkpoint_dir: Path,
) -> dict:
    train_records = filter_physics_records(train_records, cfg)
    val_records = filter_physics_records(val_records, cfg)

    audit = verify_no_event_leakage(train_records, val_records, [])
    if not audit["ok"]:
        raise RuntimeError(f"Train/valid event leakage detected: {audit}")

    print("  Extracting train physics features...", flush=True)
    train_batch = build_feature_batch(train_records, cfg, show_progress=True)
    print("  Extracting val physics features...", flush=True)
    val_batch = build_feature_batch(val_records, cfg, show_progress=True)

    model = _build_classifier(cfg)
    fit_params: dict = {}
    if cfg.classifier == "gbt":
        # HistGradientBoosting has no class_weight param across sklearn versions; apply
        # balanced weights via sample_weight so minority direction is not swamped.
        from sklearn.utils.class_weight import compute_sample_weight

        fit_params["clf__sample_weight"] = compute_sample_weight("balanced", train_batch.y)
    model.fit(train_batch.X, train_batch.y, **fit_params)

    val_pred = model.predict(val_batch.X)
    val_metrics = classification_metrics(
        val_batch.y, val_pred, labels=PHYSICS_DIRECTION_LABELS
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.joblib"
    joblib.dump(model, model_path)

    schema = {
        "feature_names": list(train_batch.feature_names),
        "feature_set": cfg.feature_set,
        "n_features": len(train_batch.feature_names),
    }
    (checkpoint_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )

    importance_payload = _feature_importance(
        model, cfg, train_batch.feature_names, val_batch
    )
    (checkpoint_dir / "feature_importance.json").write_text(
        json.dumps(importance_payload, indent=2), encoding="utf-8"
    )

    summary = {
        "model": str(model_path),
        "n_train": len(train_batch.y),
        "n_val": len(val_batch.y),
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_metrics": val_metrics,
    }
    (checkpoint_dir / "best.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_physics_model(run_dir: Path):
    run_dir = Path(run_dir)
    model = joblib.load(run_dir / "model.joblib")
    schema = json.loads((run_dir / "feature_schema.json").read_text(encoding="utf-8"))
    cfg_path = run_dir / "run_config.json"
    cfg = PhysicsConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else PhysicsConfig()
    return model, cfg, schema


def train_split(
    data_dir=None,
    checkpoint_dir: Path | None = None,
    run_name: str | None = None,
    cfg: PhysicsConfig | None = None,
    *,
    split_name: str | None = None,
) -> Path:
    cfg = cfg or PhysicsConfig()
    split_name = split_name or cfg.split_name
    train_records, val_records, test_records, meta = build_split(
        split_name,
        data_dir,
        mic_filter=cfg.mic_filter,
        channel_filter=cfg.channel_filter,
        val_fraction=cfg.val_fraction,
        seed=cfg.split_seed,
    )
    meta["n_test_clips"] = len(filter_physics_records(test_records, cfg))
    persist_split_meta(meta, default_split_meta_path(split_name))

    run_name = run_name or f"physics_lr_2class_{cfg.mono_source}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR) / physics_checkpoint_subdir(cfg) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"physics direction train ({split_name}) — L2R/R2L only")
    print(f"  include_no_vehicle={cfg.include_no_vehicle}  mono={cfg.mono_source}")
    print(f"  train clips (raw): {len(train_records)}  valid: {len(val_records)}")
    print(f"  held-out test clips (raw): {len(test_records)}")
    print(f"  leakage audit ok: {meta['audit']['ok']}")
    print(f"  checkpoint -> {out_dir}")

    summary = train_on_records(train_records, val_records, cfg, out_dir)
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir
