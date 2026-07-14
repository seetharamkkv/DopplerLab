"""Intervention tests for physics direction models (Tier 4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from idmt_experiments.config import PHYSICS_DIRECTION_LABELS, PhysicsConfig
from idmt_experiments.physics.dataset import build_feature_batch
from idmt_experiments.src.preprocess import ClipRecord


def _flip_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_base: np.ndarray | None = None,
) -> dict:
    """Flip diagnostics for a direction-reversing intervention (binary L2R/R2L).

    - ``flip_consistency``: prediction equals the flipped *true* label. Ceiling under a
      perfectly reversing model is the forward accuracy, so it conflates mechanism + skill.
    - ``flip_agreement``: prediction equals the flipped *baseline* prediction, i.e. pure
      mechanism — does the decision reverse regardless of whether it was right? This is the
      quantity the rule layer targets (should approach 1.0).
    """
    expected = 1 - y_true
    ok = y_pred == expected
    out = {
        "flip_consistency": float(np.mean(ok)) if len(ok) else None,
        "n_checked": int(len(ok)),
        "n_correct_flips": int(np.sum(ok)),
    }
    if y_pred_base is not None and len(y_pred_base) == len(y_pred):
        agree = y_pred == (1 - y_pred_base)
        out["flip_agreement"] = float(np.mean(agree)) if len(agree) else None
        out["n_flipped"] = int(np.sum(agree))
    return out


def run_interventions(
    records: list[ClipRecord],
    cfg: PhysicsConfig,
    predict_fn,
    *,
    mono_source: str | None = None,
) -> dict:
    """
    predict_fn: callable(X) -> y_pred for a feature matrix.

    Tests:
    - time_reverse: waveform reversed before feature extraction
    - channel_swap: left <-> right mono (only when mono_source is left or right)
    """
    mono = mono_source or cfg.mono_source
    base = build_feature_batch(records, cfg, mono_source=mono)
    y_pred_base = predict_fn(base.X)

    rev = build_feature_batch(records, cfg, mono_source=mono, time_reverse=True)
    y_pred_rev = predict_fn(rev.X)
    time_reverse = _flip_rate(base.y, y_pred_rev, y_pred_base)

    channel_swap: dict | None = None
    if mono in ("left", "right"):
        swapped = "right" if mono == "left" else "left"
        swap_batch = build_feature_batch(records, cfg, mono_source=swapped)
        y_pred_swap = predict_fn(swap_batch.X)
        channel_swap = _flip_rate(base.y, y_pred_swap, y_pred_base)

    return {
        "labels": list(PHYSICS_DIRECTION_LABELS),
        "mono_source": mono,
        "n_samples": int(len(base.y)),
        "time_reverse": time_reverse,
        "channel_swap": channel_swap,
    }


def save_interventions(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
