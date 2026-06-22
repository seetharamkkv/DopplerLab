"""Classification metrics for direction task."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: tuple[str, ...],
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_labels = len(labels)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_labels)), zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(n_labels)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_labels)))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {labels[i]: float(per_class_f1[i]) for i in range(n_labels)},
        "confusion_matrix": cm.tolist(),
        "confusion_labels": list(labels),
        "n_samples": int(len(y_true)),
    }


def channel_swap_consistency(
    y_pred_orig: np.ndarray,
    y_pred_swap: np.ndarray,
    *,
    n_classes: int,
) -> dict:
    """For binary L2R/R2L: swapped stereo should flip label. For 3-class, only vehicle rows."""
    orig = np.asarray(y_pred_orig, dtype=int)
    swap = np.asarray(y_pred_swap, dtype=int)
    if n_classes == 2:
        expected = 1 - orig
        mask = np.ones(len(orig), dtype=bool)
    else:
        vehicle = orig < 2
        expected = np.where(orig == 0, 1, 0)
        mask = vehicle & (swap < 2) & (orig < 2)
        orig = orig[mask]
        swap = swap[mask]
        expected = expected[mask]
    if len(orig) == 0:
        return {"flip_consistency": None, "n_checked": 0}
    ok = swap == expected
    return {
        "flip_consistency": float(np.mean(ok)),
        "n_checked": int(len(orig)),
        "n_correct_flips": int(np.sum(ok)),
    }
