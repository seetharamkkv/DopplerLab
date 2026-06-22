"""Phase A report: per-clip predictions, vehicle ID, and result_summary.txt."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from length_estimation.config import DEFAULT_SPECS_PATH
from length_estimation.src.evaluate import (
    baseline_mean_mae,
    correlation_report,
    feature_columns,
    lovo_splits,
    run_lovo_regression,
    speed_normalized_columns,
)


def _nearest_vehicle_by_length(length_m: float, specs: pd.DataFrame) -> str:
    idx = (specs["length_m"] - length_m).abs().idxmin()
    return str(idx)


def _nearest_vehicle_by_wheelbase(wb: float, specs: pd.DataFrame) -> str:
    idx = (specs["wheelbase_m"] - wb).abs().idxmin()
    return str(idx)


def run_insample_vehicle_classification(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Vehicle ID when all 13 classes appear in training (VS13 train/valid split).

    Uses Train_valid_split.txt labels. This measures acoustic separability of
    vehicle models — not LOVO generalisation to an unseen car.
    """
    from sklearn.ensemble import RandomForestClassifier

    features = features or speed_normalized_columns(df)
    features = [f for f in features if f in df.columns]

    if "split" not in df.columns or (df["split"] == "unknown").all():
        return pd.DataFrame(columns=["clip_id", "predicted_vehicle_insample", "insample_correct"])

    train = df[df["split"] == "train"]
    test = df[df["split"] == "valid"]
    if train.empty or test.empty:
        return pd.DataFrame(columns=["clip_id", "predicted_vehicle_insample", "insample_correct"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features].astype(float).values)
    X_test = scaler.transform(test[features].astype(float).values)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, train["vehicle"].values)
    y_pred = clf.predict(X_test)

    return pd.DataFrame(
        {
            "clip_id": test["clip_id"].values,
            "predicted_vehicle_insample": y_pred,
            "insample_correct": y_pred == test["vehicle"].values,
        }
    )


def run_lovo_classification(
    df: pd.DataFrame,
    features: list[str] | None = None,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    LOVO vehicle ID via kNN among training clips.

    Strict 13-class RF cannot predict a held-out vehicle name (never seen in training).
    Instead: for each test clip, find k nearest training clips in feature space and vote.
    """
    features = features or speed_normalized_columns(df)
    features = [f for f in features if f in df.columns]

    fold_rows: list[dict] = []
    pred_rows: list[dict] = []

    for held_out, train_idx, test_idx in lovo_splits(df):
        train = df.loc[train_idx]
        test = df.loc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features].astype(float).values)
        X_test = scaler.transform(test[features].astype(float).values)
        y_train = train["vehicle"].values
        y_test = test["vehicle"].values

        preds = []
        for i in range(len(X_test)):
            dists = np.linalg.norm(X_train - X_test[i], axis=1)
            nn_idx = np.argsort(dists)[:k_neighbors]
            nn_labels = y_train[nn_idx]
            # majority vote
            vals, counts = np.unique(nn_labels, return_counts=True)
            preds.append(vals[int(np.argmax(counts))])

        y_pred = np.array(preds)
        acc = accuracy_score(y_test, y_pred)
        fold_rows.append(
            {
                "held_out_vehicle": held_out,
                "accuracy": float(acc),
                "n_test": len(test),
                "n_train": len(train),
            }
        )

        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "clip_id": df.loc[idx, "clip_id"],
                    "actual_vehicle": y_test[i],
                    "predicted_vehicle": y_pred[i],
                    "vehicle_correct": bool(y_pred[i] == y_test[i]),
                    "speed_kmh": float(df.loc[idx, "speed_kmh"]),
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def _attach_split(features_df: pd.DataFrame, data_dir=None) -> pd.DataFrame:
    from length_estimation.src.preprocess import load_clips

    df = features_df.copy()
    records = load_clips(data_dir)
    split_map = {r.clip_id: r.split for r in records}
    df["split"] = df["clip_id"].map(split_map).fillna("unknown")
    return df


def load_best_lovo_predictions(output_dir: Path, target: str) -> pd.DataFrame | None:
    """Per-clip predictions for reporting (best physics model if mean baseline wins overall)."""
    for name in (f"clip_model_preds_{target}.csv", f"best_lovo_preds_{target}.csv"):
        path = Path(output_dir) / name
        if path.exists():
            return pd.read_csv(path)
    return None


def build_clip_report(features_df: pd.DataFrame, specs_path: Path | None = None, data_dir=None, output_dir: Path | None = None) -> pd.DataFrame:
    """Per-clip table: actual vehicle, predicted length, predicted vehicle."""
    features_df = _attach_split(features_df, data_dir)
    specs = pd.read_csv(specs_path or DEFAULT_SPECS_PATH).set_index("short_name")
    speed_feats = speed_normalized_columns(features_df)
    phase_a_dir = Path(output_dir) if output_dir else None

    len_preds = load_best_lovo_predictions(phase_a_dir, "length_m") if phase_a_dir else None
    wb_preds = load_best_lovo_predictions(phase_a_dir, "wheelbase_m") if phase_a_dir else None

    if len_preds is None:
        _, len_preds = run_lovo_regression(features_df, speed_feats, "length_m", model="ridge")
    if wb_preds is None:
        _, wb_preds = run_lovo_regression(features_df, speed_feats, "wheelbase_m", model="ridge")
    _, cls_preds = run_lovo_classification(features_df, speed_feats)

    len_preds = len_preds.rename(
        columns={"y_true": "actual_length_m", "y_pred": "predicted_length_m", "vehicle": "held_out_vehicle"}
    )
    wb_preds = wb_preds.rename(columns={"y_pred": "predicted_wheelbase_m"})[["clip_id", "predicted_wheelbase_m"]]

    report = features_df[["clip_id", "vehicle", "speed_kmh", "length_m", "wheelbase_m", "split"]].copy()
    report = report.rename(columns={"vehicle": "actual_vehicle", "length_m": "actual_length_m", "wheelbase_m": "actual_wheelbase_m"})

    report = report.merge(len_preds[["clip_id", "predicted_length_m", "abs_error"]], on="clip_id", how="left")
    report = report.rename(columns={"abs_error": "length_abs_error_m"})
    report = report.merge(wb_preds, on="clip_id", how="left")
    report = report.merge(
        cls_preds[["clip_id", "predicted_vehicle", "vehicle_correct"]],
        on="clip_id",
        how="left",
    )
    report = report.rename(
        columns={
            "predicted_vehicle": "predicted_vehicle_knn",
            "vehicle_correct": "knn_match",
        }
    )

    insample = run_insample_vehicle_classification(features_df, speed_feats)
    if not insample.empty:
        report = report.merge(insample, on="clip_id", how="left")
        report["predicted_vehicle_insample"] = report["predicted_vehicle_insample"].fillna(
            report["predicted_vehicle_knn"]
        )
    else:
        report["predicted_vehicle_insample"] = report["predicted_vehicle_knn"]
        report["insample_correct"] = False

    report["wheelbase_abs_error_m"] = (report["predicted_wheelbase_m"] - report["actual_wheelbase_m"]).abs()
    report["vehicle_from_length"] = report["predicted_length_m"].apply(
        lambda x: _nearest_vehicle_by_length(x, specs)
    )
    report["vehicle_from_wheelbase"] = report["predicted_wheelbase_m"].apply(
        lambda x: _nearest_vehicle_by_wheelbase(x, specs)
    )
    report["length_id_match"] = report["vehicle_from_length"] == report["actual_vehicle"]

    display = specs["display_name"].to_dict()
    report["actual_car"] = report["actual_vehicle"].map(display)
    report["predicted_car_knn"] = report["predicted_vehicle_knn"].map(display)
    report["predicted_car_insample"] = report["predicted_vehicle_insample"].map(display)
    report["predicted_car_from_length"] = report["vehicle_from_length"].map(display)

    col_order = [
        "clip_id",
        "actual_vehicle",
        "actual_car",
        "actual_length_m",
        "predicted_length_m",
        "length_abs_error_m",
        "predicted_car_from_length",
        "length_id_match",
        "predicted_vehicle_knn",
        "predicted_car_knn",
        "knn_match",
        "predicted_vehicle_insample",
        "predicted_car_insample",
        "insample_correct",
        "actual_wheelbase_m",
        "predicted_wheelbase_m",
        "wheelbase_abs_error_m",
        "vehicle_from_wheelbase",
        "speed_kmh",
        "split",
    ]
    return report[[c for c in col_order if c in report.columns]]


def _vehicle_summary_table(report: pd.DataFrame) -> pd.DataFrame:
    g = report.groupby("actual_vehicle", sort=True)
    return pd.DataFrame(
        {
            "n_clips": g.size(),
            "actual_length_m": g["actual_length_m"].first(),
            "mean_pred_length_m": g["predicted_length_m"].mean(),
            "length_mae_m": g["length_abs_error_m"].mean(),
            "classifier_acc": g["insample_correct"].mean(),
            "knn_acc": g["knn_match"].mean(),
            "length_nn_acc": g["length_id_match"].mean(),
        }
    ).round(3)


def _format_df_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    return view.to_string(index=True)


def _load_json_summary(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def write_result_summary(
    features_df: pd.DataFrame,
    output_dir: Path,
    report: pd.DataFrame | None = None,
    data_dir=None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_df = _attach_split(features_df, data_dir)
    if report is None:
        report = build_clip_report(features_df, data_dir=data_dir, output_dir=output_dir)

    report.to_csv(output_dir / "clip_predictions.csv", index=False)
    vehicle_summary = _vehicle_summary_table(report)
    vehicle_summary.to_csv(output_dir / "vehicle_summary.csv")

    len_summary = _load_json_summary(output_dir / "phase_a_summary_length_m.json")
    wb_summary = _load_json_summary(output_dir / "phase_a_summary_wheelbase_m.json")
    cls_folds, cls_preds = run_lovo_classification(features_df)

    overall_knn_acc = float(report["knn_match"].fillna(False).mean())
    insample_mask = report["insample_correct"].notna() & (report["split"] == "valid") if "split" in report.columns else report["insample_correct"].notna()
    overall_insample_acc = float(report.loc[insample_mask, "insample_correct"].mean()) if insample_mask.any() else float("nan")
    overall_len_nn_acc = float(report["length_id_match"].mean())
    mean_length_mae = float(report["length_abs_error_m"].mean())
    mean_wb_mae = float(report["wheelbase_abs_error_m"].mean())

    # Interpretation helpers
    def _verdict_lovo(ridge_mae: float | None, baseline: float | None, target_range: float) -> str:
        if ridge_mae is None or baseline is None:
            return "n/a"
        if ridge_mae < baseline * 0.95:
            return "BELOW baseline (generalises somewhat)"
        if ridge_mae <= baseline * 1.05:
            return "AT baseline (no clear length signal under LOVO)"
        return "ABOVE baseline (worse than guessing mean)"

    len_verdict = len_summary.get("verdict") if len_summary else "n/a"
    wb_verdict = wb_summary.get("verdict") if wb_summary else "n/a"
    best_len_model = len_summary.get("best_model", "n/a") if len_summary else "n/a"
    best_wb_model = wb_summary.get("best_model", "n/a") if wb_summary else "n/a"
    best_len_physics = len_summary.get("best_physics_model", best_len_model) if len_summary else "n/a"
    best_wb_physics = wb_summary.get("best_physics_model", best_wb_model) if wb_summary else "n/a"

    corr_len = correlation_report(features_df, "length_m").head(5)
    corr_wb = correlation_report(features_df, "wheelbase_m").head(5)

    wrong_cls = report[report["insample_correct"] == False].sort_values("clip_id")
    wrong_len = report[~report["length_id_match"]].sort_values("length_abs_error_m", ascending=False)

    compact = report[
        [
            "clip_id",
            "actual_car",
            "actual_length_m",
            "predicted_length_m",
            "predicted_car_from_length",
            "predicted_car_insample",
            "predicted_car_knn",
        ]
    ].copy()
    if "insample_correct" in report.columns:
        compact["insample_ok"] = report["insample_correct"].map(lambda x: "yes" if x is True else ("no" if x is False else "n/a"))

    len_tldr = f"  • LOVO length: {len_verdict} (best overall: {best_len_model})."
    if len_summary and best_len_physics != best_len_model:
        len_tldr += f" Best physics model: {best_len_physics} ({len_summary.get('best_physics_lovo_mae', float('nan')):.3f}m)."
    if len_summary and len_summary.get("sanity_baselines"):
        sb = len_summary["sanity_baselines"]
        len_tldr += (
            f" Sanity: speed={sb.get('speed_only', float('nan')):.3f}m, "
            f"power={sb.get('power_only', float('nan')):.3f}m "
            f"(power < mean ⇒ engine-identity leakage, not geometry)."
        )

    lines = [
        "=" * 72,
        "VS13 VEHICLE LENGTH — PHASE A RESULT SUMMARY",
        "=" * 72,
        "",
        "TL;DR",
        "-" * 72,
        f"  • Pooled feature↔length correlation is weak (top |ρ| ≈ {len_summary.get('top_spearman_r', 0):.2f} if available).",
        len_tldr,
        f"  • LOVO wheelbase ({best_wb_model}): {wb_verdict}.",
        f"  • LOVO kNN vehicle ID (12 training cars only): {overall_knn_acc:.1%} exact match (expected ≈0%).",
        f"  • In-sample vehicle ID (VS13 train→valid, all 13 in train): {overall_insample_acc:.1%} on valid clips.",
        f"  • Nearest-length vehicle guess: {overall_len_nn_acc:.1%} accuracy.",
        "",
        "  Interpretation: Strong pooled correlations often reflect ENGINE/vehicle identity,",
        "  not generalisable geometry. If classifier accuracy >> length LOVO gain, the spectrogram",
        "  encodes 'which car' more reliably than 'how long'.",
        "",
        "LENGTH REGRESSION (LOVO — best model per target)",
        "-" * 72,
    ]

    if len_summary:
        bunch = len_summary.get("bunching", {})
        lines.extend(
            [
                f"  Best model       : {len_summary.get('best_model', 'n/a')}",
                f"  Target range     : {len_summary['target_range_m']:.3f} m",
                f"  Baseline LOVO MAE: {len_summary['baseline_lovo_mae']:.3f} m  (always predict train mean)",
                f"  Best LOVO MAE    : {len_summary.get('best_lovo_mae', len_summary.get('ridge_lovo_mae_mean', float('nan'))):.3f} m",
                f"  Beats baseline   : {'yes' if len_summary.get('best_beats_baseline') else 'no'}",
                f"  Top Spearman     : {len_summary.get('top_feature_spearman', len_summary.get('top_feature', 'n/a'))} "
                f"(ρ={len_summary.get('top_spearman_r', 0):.3f})",
                f"  Top partial r    : {len_summary.get('top_feature_partial_r', 'n/a')} "
                f"(r={len_summary.get('top_partial_r', 0):.3f}, controls speed+power)",
                f"  Pred std ratio   : {bunch.get('pred_std_ratio', float('nan')):.1%}  "
                f"(1.0 = full spread; bunching if << 1)",
                f"  corr(speed,pred) : {bunch.get('corr_speed_pred', float('nan')):.3f}",
                f"  Verdict          : {len_verdict}",
                "",
            ]
        )
        models = len_summary.get("all_models", {})
        sanity = len_summary.get("sanity_baselines", {})
        if sanity:
            lines.append("  Sanity baselines (not geometry — confound checks):")
            for name, mae in sanity.items():
                lines.append(f"    {name:32s} {mae:.3f} m")
            lines.append("")
        if models:
            lines.append("  Physics / feature models (length):")
            for name, mae in sorted(models.items(), key=lambda x: x[1]):
                lines.append(f"    {name:32s} {mae:.3f} m")
            lines.append("")

    lines.extend(
        [
            "WHEELBASE REGRESSION (LOVO — best model)",
            "-" * 72,
        ]
    )
    if wb_summary:
        bunch_wb = wb_summary.get("bunching", {})
        lines.extend(
            [
                f"  Best model       : {wb_summary.get('best_model', 'n/a')}",
                f"  Target range     : {wb_summary['target_range_m']:.3f} m",
                f"  Baseline LOVO MAE: {wb_summary['baseline_lovo_mae']:.3f} m",
                f"  Best LOVO MAE    : {wb_summary.get('best_lovo_mae', wb_summary.get('ridge_lovo_mae_mean', float('nan'))):.3f} m",
                f"  Beats baseline   : {'yes' if wb_summary.get('best_beats_baseline') else 'no'}",
                f"  Top Spearman     : {wb_summary.get('top_feature_spearman', wb_summary.get('top_feature', 'n/a'))} "
                f"(ρ={wb_summary.get('top_spearman_r', 0):.3f})",
                f"  Pred std ratio   : {bunch_wb.get('pred_std_ratio', float('nan')):.1%}",
                f"  Verdict          : {wb_verdict}",
                "",
            ]
        )

    lines.extend(
        [
            "VEHICLE ID — LOVO kNN (held-out car not in library; exact match rarely possible)",
            "-" * 72,
            f"  LOVO kNN accuracy     : {overall_knn_acc:.1%}  ({report['knn_match'].fillna(False).sum():.0f}/{len(report)} clips)",
            f"  In-sample RF accuracy : {overall_insample_acc:.1%}  (valid split only, all 13 cars in train)",
            f"  Mean length MAE  : {mean_length_mae:.3f} m",
            f"  Mean wheelbase MAE: {mean_wb_mae:.3f} m",
            f"  Length→nearest-car accuracy: {overall_len_nn_acc:.1%}",
            "",
            _format_df_table(cls_folds.rename(columns={"held_out_vehicle": "vehicle"})),
            "",
            "PER-VEHICLE SUMMARY",
            "-" * 72,
            _format_df_table(vehicle_summary),
            "",
            "TOP FEATURES ↔ LENGTH (Spearman, pooled)",
            "-" * 72,
            _format_df_table(corr_len),
            "",
            "TOP FEATURES ↔ WHEELBASE (Spearman, pooled)",
            "-" * 72,
            _format_df_table(corr_wb),
            "",
            "CLIP TABLE (all 400) — see clip_predictions.csv for full export",
            "Columns: actual_car | pred_length | pred_car_from_length | pred_car_insample",
            "-" * 72,
            _format_df_table(
                compact.rename(
                    columns={
                        "actual_car": "actual",
                        "predicted_length_m": "pred_len_m",
                        "predicted_car_from_length": "pred_car(len)",
                        "predicted_car_insample": "pred_car(clf)",
                        "predicted_car_knn": "pred_car(knn)",
                    }
                ),
                max_rows=400,
            ),
            "",
            "WORST LENGTH ERRORS (top 10)",
            "-" * 72,
            _format_df_table(
                wrong_len[
                    ["clip_id", "actual_car", "actual_length_m", "predicted_length_m", "length_abs_error_m"]
                ].head(10)
            ),
            "",
            "MISCLASSIFIED CLIPS (first 20)",
            "-" * 72,
            _format_df_table(
                wrong_cls[
                    ["clip_id", "actual_car", "predicted_car_insample", "speed_kmh"]
                ].head(20)
            ),
            "",
            "OUTPUT FILES",
            "-" * 72,
            "  clip_predictions.csv   — full per-clip table",
            "  vehicle_summary.csv    — aggregated by actual vehicle",
            "  result_summary.txt     — this file",
            "",
            "=" * 72,
        ]
    )

    out_path = output_dir / "result_summary.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_report(features_df: pd.DataFrame, output_dir: Path, data_dir=None) -> pd.DataFrame:
    """Build clip report and write result_summary.txt."""
    output_dir = Path(output_dir)
    report = build_clip_report(features_df, data_dir=data_dir, output_dir=output_dir)
    write_result_summary(features_df, output_dir, report, data_dir=data_dir)
    return report
