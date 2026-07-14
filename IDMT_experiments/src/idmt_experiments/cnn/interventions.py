"""Intervention tests for CNN direction models (time-reverse, channel-swap).

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Opt-in diagnostic paths only — must not alter default train/eval when interventions are
off. Critical path for published monoaural metrics (vehicle bal. acc 81.5% / 79.3% / 73.6%).
Do not change default behaviour, numerics, or evaluation outputs without re-benchmarking
all three reference runs. Refactoring for maintainability is OK only if metrics stay
bit-identical. New work: separate --run-name or new modules.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from idmt_experiments.config import DIRECTION_LABELS, DirectionConfig
from idmt_experiments.cnn.metrics import direction_intervention_flip
from idmt_experiments.src.preprocess import ClipRecord, filter_records


def run_interventions(
    records: list[ClipRecord],
    cfg: DirectionConfig,
    predict_fn,
    *,
    vehicle_only: bool = True,
) -> dict:
    """Run time-reverse and channel-swap probes on vehicle direction clips.

    ``predict_fn(records, *, time_reverse=..., swap_channels=...) -> y_pred``
    """
    records = filter_records(records, cfg)
    if vehicle_only:
        records = [r for r in records if not r.is_background]
    if not records:
        return {"n_samples": 0, "time_reverse": None, "channel_swap": None}

    y_true = np.array(
        [0 if r.travel_direction == "L2R" else 1 for r in records],
        dtype=np.int64,
    )
    y_pred_base = predict_fn(records, time_reverse=False, swap_channels=False)
    y_pred_rev = predict_fn(records, time_reverse=True, swap_channels=False)
    time_reverse = direction_intervention_flip(
        y_true, y_pred_rev, y_pred_base, n_classes=cfg.n_classes
    )

    channel_swap = None
    if cfg.feature_type in ("cc", "stereo_mel") or (
        cfg.feature_type in ("mel", "complex_stft") and cfg.mono_source in ("left", "right")
    ):
        y_pred_swap = predict_fn(records, time_reverse=False, swap_channels=True)
        channel_swap = direction_intervention_flip(
            y_true, y_pred_swap, y_pred_base, n_classes=cfg.n_classes
        )

    return {
        "labels": list(DIRECTION_LABELS[: cfg.n_classes]),
        "mono_source": cfg.mono_source,
        "feature_type": cfg.feature_type,
        "n_samples": int(len(records)),
        "vehicle_only": vehicle_only,
        "time_reverse": time_reverse,
        "channel_swap": channel_swap,
    }


def save_interventions(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
