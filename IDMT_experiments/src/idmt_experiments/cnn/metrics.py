"""Classification metrics for direction task.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Critical path for published monoaural metrics (vehicle bal. acc 81.5% / 79.3% / 73.6%).
Do not change default behaviour, numerics, or evaluation outputs without re-benchmarking
all three reference runs. Refactoring for maintainability is OK only if metrics stay
bit-identical. New work: separate --run-name or new modules.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_labels)), zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(n_labels)), zero_division=0)
    per_class_precision = precision_score(
        y_true, y_pred, average=None, labels=list(range(n_labels)), zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=list(range(n_labels)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_labels)))
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "per_class_f1": {labels[i]: float(per_class_f1[i]) for i in range(n_labels)},
        "per_class_precision": {labels[i]: float(per_class_precision[i]) for i in range(n_labels)},
        "per_class_recall": {labels[i]: float(per_class_recall[i]) for i in range(n_labels)},
        "confusion_matrix": cm.tolist(),
        "confusion_labels": list(labels),
        "n_samples": int(len(y_true)),
    }


def direction_intervention_flip(
    y_true: np.ndarray,
    y_pred_intervention: np.ndarray,
    y_pred_base: np.ndarray | None = None,
    *,
    n_classes: int = 3,
) -> dict:
    """L2R/R2L flip metrics on vehicle rows (excludes no_vehicle when n_classes=3)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pi = np.asarray(y_pred_intervention, dtype=int)
    if n_classes == 2:
        mask = np.ones(len(y_true), dtype=bool)
    else:
        mask = y_true < 2
    yt = y_true[mask]
    pi = y_pi[mask]
    if len(yt) == 0:
        return {"flip_consistency": None, "n_checked": 0, "n_correct_flips": 0}
    expected_from_true = 1 - yt
    ok = pi == expected_from_true
    out = {
        "flip_consistency": float(np.mean(ok)),
        "n_checked": int(len(yt)),
        "n_correct_flips": int(np.sum(ok)),
    }
    if y_pred_base is not None:
        pb = np.asarray(y_pred_base, dtype=int)[mask]
        dir_mask = pb < 2
        if dir_mask.any():
            expected_from_pred = np.where(pb == 0, 1, 0)
            agree = pi[dir_mask] == expected_from_pred[dir_mask]
            out["flip_agreement"] = float(np.mean(agree))
            out["n_flipped"] = int(np.sum(agree))
            out["n_agreement_checked"] = int(dir_mask.sum())
    return out


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
