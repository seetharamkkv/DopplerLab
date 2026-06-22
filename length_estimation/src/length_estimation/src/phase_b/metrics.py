"""Regression + nearest-vehicle classification metrics for length predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from length_estimation.config import DEFAULT_SPECS_PATH


def nearest_vehicle(length_m: float, specs: pd.DataFrame) -> str:
    idx = (specs["length_m"] - length_m).abs().idxmin()
    return str(idx)


def enrich_with_vehicle_id(
    preds: pd.DataFrame,
    *,
    pred_col: str = "y_pred",
    true_vehicle_col: str = "vehicle",
    specs_path: Path | None = None,
) -> pd.DataFrame:
    """
    Map each predicted length to the fleet vehicle with closest catalog length.
    This is the deployable 'classification' proxy (no oracle vehicle ID at test time).
    """
    specs = pd.read_csv(specs_path or DEFAULT_SPECS_PATH).set_index("short_name")
    display = specs["display_name"].to_dict()

    out = preds.copy()
    out["predicted_vehicle"] = out[pred_col].apply(lambda x: nearest_vehicle(float(x), specs))
    out["vehicle_correct"] = out["predicted_vehicle"] == out[true_vehicle_col]
    out["predicted_car"] = out["predicted_vehicle"].map(display)
    if "actual_car" not in out.columns and true_vehicle_col in out.columns:
        out["actual_car"] = out[true_vehicle_col].map(display)
    return out


def summarize_predictions(
    preds: pd.DataFrame,
    *,
    pred_col: str = "y_pred",
    true_col: str = "y_true",
    error_col: str | None = "abs_error",
) -> dict:
    """Overall MAE + nearest-vehicle classification accuracy."""
    if error_col and error_col in preds.columns:
        mae = float(preds[error_col].mean())
    else:
        mae = float(np.mean(np.abs(preds[pred_col].astype(float) - preds[true_col].astype(float))))

    enriched = enrich_with_vehicle_id(preds, pred_col=pred_col)
    n = len(enriched)
    n_correct = int(enriched["vehicle_correct"].sum())
    acc = float(n_correct / n) if n else float("nan")

    return {
        "n_clips": n,
        "mae_m": mae,
        "rmse_m": float(np.sqrt(np.mean((enriched[pred_col] - enriched[true_col]) ** 2))),
        "vehicle_id_accuracy": acc,
        "vehicle_id_correct": n_correct,
        "vehicle_id_total": n,
    }
