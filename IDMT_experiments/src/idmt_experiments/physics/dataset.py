"""Build feature matrices from IDMT clips for physics direction models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from idmt_experiments.config import PhysicsConfig
from idmt_experiments.physics.features import extract_physics_features, feature_names, physics_label_index
from idmt_experiments.src.preprocess import ClipRecord, filter_physics_records


@dataclass
class FeatureBatch:
    X: np.ndarray
    y: np.ndarray
    records: list[ClipRecord]
    feature_names: tuple[str, ...]


def build_feature_batch(
    records: list[ClipRecord],
    cfg: PhysicsConfig,
    *,
    mono_source: str | None = None,
    time_reverse: bool = False,
    show_progress: bool = False,
) -> FeatureBatch:
    records = filter_physics_records(records, cfg)
    names = feature_names(cfg)
    rows: list[list[float]] = []
    labels: list[int] = []
    kept: list[ClipRecord] = []

    iterator = records
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(records, desc="physics features", unit="clip")

    for rec in iterator:
        feats = extract_physics_features(
            rec,
            cfg,
            mono_source=mono_source,
            time_reverse=time_reverse,
        )
        rows.append([feats[n] for n in names])
        labels.append(physics_label_index(rec))
        kept.append(rec)

    if not rows:
        return FeatureBatch(
            X=np.zeros((0, len(names)), dtype=np.float64),
            y=np.zeros(0, dtype=np.int64),
            records=[],
            feature_names=names,
        )
    X = np.asarray(rows, dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    if not finite.all():
        bad = int((~finite).sum())
        print(f"  WARNING: dropping {bad} clips with non-finite physics features")
        X = X[finite]
        labels_arr = np.asarray(labels, dtype=np.int64)[finite]
        kept = [r for r, ok in zip(kept, finite) if ok]
    else:
        labels_arr = np.asarray(labels, dtype=np.int64)

    return FeatureBatch(X=X, y=labels_arr, records=kept, feature_names=names)
