"""Evaluation metrics, correlations, and LOVO cross-validation (Phase A)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

TargetCol = Literal["length_m", "wheelbase_m"]

META_COLS = {
    "clip_id",
    "vehicle",
    "speed_kmh",
    "speed_mps",
    "length_m",
    "wheelbase_m",
    "power_kw",
    "engine_type",
    "wav_path",
    "split",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def speed_normalized_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(df) if "_x_speed_m" in c]


def correlation_report(
    df: pd.DataFrame,
    target: TargetCol = "length_m",
    method: str = "spearman",
) -> pd.DataFrame:
    cols = feature_columns(df)
    rows = []
    for col in cols:
        x = df[col].astype(float)
        y = df[target].astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        if method == "pearson":
            r, p = stats.pearsonr(x[mask], y[mask])
        else:
            r, p = stats.spearmanr(x[mask], y[mask])
        rows.append({"feature": col, "r": r, "p_value": p, "abs_r": abs(r), "n": int(mask.sum())})
    out = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    return out


def partial_correlation(df: pd.DataFrame, feature: str, target: TargetCol, control: str) -> float:
    """Linear partial correlation controlling for one covariate."""
    sub = df[[feature, target, control]].astype(float).dropna()
    if len(sub) < 4:
        return float("nan")

    def _residual(y_col: str, x_col: str) -> np.ndarray:
        x = sub[x_col].values
        y = sub[y_col].values
        x = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return y - x @ beta

    r1 = _residual(feature, control)
    r2 = _residual(target, control)
    return float(stats.pearsonr(r1, r2)[0])


def lovo_splits(df: pd.DataFrame) -> list[tuple[str, pd.Index, pd.Index]]:
    vehicles = sorted(df["vehicle"].unique())
    splits = []
    for held_out in vehicles:
        test_idx = df.index[df["vehicle"] == held_out]
        train_idx = df.index[df["vehicle"] != held_out]
        splits.append((held_out, train_idx, test_idx))
    return splits


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_range: float) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "norm_mae": float(mae / target_range) if target_range > 0 else float("nan"),
    }


def run_lovo_regression(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: TargetCol = "length_m",
    model: str = "ridge",
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-vehicle-out regression. Returns (per-fold metrics, predictions)."""
    features = features or feature_columns(df)
    features = [f for f in features if f in df.columns]
    target_range = float(df[target].max() - df[target].min())

    fold_rows = []
    pred_rows = []

    for vehicle, train_idx, test_idx in lovo_splits(df):
        train = df.loc[train_idx]
        test = df.loc[test_idx]

        X_train = train[features].astype(float).values
        y_train = train[target].astype(float).values
        X_test = test[features].astype(float).values
        y_test = test[target].astype(float).values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model == "rf":
            reg = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
        else:
            reg = Ridge(alpha=alpha)
            reg.fit(X_train_s, y_train)
            y_pred = reg.predict(X_test_s)

        m = regression_metrics(y_test, y_pred, target_range)
        m["vehicle"] = vehicle
        m["model"] = model
        m["n_train"] = len(train)
        m["n_test"] = len(test)
        fold_rows.append(m)

        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "clip_id": df.loc[idx, "clip_id"],
                    "vehicle": vehicle,
                    "y_true": y_test[i],
                    "y_pred": y_pred[i],
                    "error": y_pred[i] - y_test[i],
                    "abs_error": abs(y_pred[i] - y_test[i]),
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


CONTROL_COLS = ["speed_kmh", "power_kw"]

PHYSICS_PROXIES = [
    "env_10db_width_x_speed_m",
    "env_3db_width_x_speed_m",
    "centroid_delta_t_x_speed_m",
    "reassigned_doppler_transition_width_x_speed_m",
]

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def physics_duration_columns(df: pd.DataFrame) -> list[str]:
    """Temporal width/lag features in seconds (not multiplied by speed)."""
    keys = ("width_s", "delta_t_s", "transition_width_s", "rise_s", "fall_s", "asymmetry")
    return [c for c in feature_columns(df) if any(k in c for k in keys) and "_x_speed" not in c]


def physics_speed_columns(df: pd.DataFrame) -> list[str]:
    """L ≈ v·Δt proxies from the experiment plan."""
    return [c for c in feature_columns(df) if c.endswith("_x_speed_m")]


def envelope_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(df) if c.startswith("env_")]


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Curated physics feature groups (§7.1–7.4)."""
    dur = physics_duration_columns(df)
    spd = physics_speed_columns(df)
    return {
        "envelope_duration": [c for c in dur if c.startswith("env_")],
        "envelope_speed": [c for c in spd if "env_" in c],
        "doppler": [c for c in feature_columns(df) if "doppler" in c],
        "xcorr_duration": [c for c in dur if "xcorr" in c or "centroid" in c],
        "xcorr_speed": [c for c in spd if "xcorr" in c or "centroid" in c],
        "physics_core": [c for c in PHYSICS_PROXIES if c in df.columns],
        "speed_normalized": spd,
        "all_features": feature_columns(df),
    }


def partial_correlation_multi(
    df: pd.DataFrame,
    feature: str,
    target: str,
    controls: list[str],
) -> float:
    """Linear partial correlation controlling for multiple covariates."""
    cols = [feature, target] + controls
    sub = df[cols].astype(float).dropna()
    if len(sub) < len(controls) + 3:
        return float("nan")

    def _residual(col: str, against: list[str]) -> np.ndarray:
        y = sub[col].values
        x = np.column_stack([np.ones(len(sub)), sub[against].values])
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return y - x @ beta

    r_feat = _residual(feature, controls)
    r_tgt = _residual(target, controls)
    if np.std(r_feat) < 1e-12 or np.std(r_tgt) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(r_feat, r_tgt)[0])


def partial_correlation_report(
    df: pd.DataFrame,
    target: TargetCol = "length_m",
    controls: list[str] | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    controls = controls or CONTROL_COLS
    rows = []
    for feat in feature_columns(df):
        pc = partial_correlation_multi(df, feat, target, controls)
        rows.append({"feature": feat, "partial_r": pc, "abs_partial_r": abs(pc) if np.isfinite(pc) else float("nan")})
    out = pd.DataFrame(rows).sort_values("abs_partial_r", ascending=False, na_position="last")
    return out.head(top_n) if top_n else out


def bunching_diagnostics(
    preds: pd.DataFrame,
    features_df: pd.DataFrame,
    target_col: TargetCol = "length_m",
    pred_col: str = "y_pred",
) -> dict[str, float]:
    """Detect regression-to-the-mean and speed leakage in LOVO predictions."""
    merged = preds.drop(columns=["vehicle"], errors="ignore").merge(
        features_df[["clip_id", "vehicle", "speed_kmh", target_col]],
        on="clip_id",
        how="left",
    )
    y_true = merged[target_col].astype(float)
    y_pred = merged[pred_col].astype(float)

    within_std = merged.groupby("vehicle")[pred_col].std().mean()
    actual_std = float(y_true.std())
    pred_std = float(y_pred.std())
    speed_corr = float(stats.pearsonr(merged["speed_kmh"], y_pred)[0]) if len(merged) > 2 else float("nan")

    return {
        "pred_std_m": pred_std,
        "actual_std_m": actual_std,
        "pred_std_ratio": pred_std / actual_std if actual_std > 0 else float("nan"),
        "within_vehicle_pred_std_mean_m": float(within_std),
        "corr_speed_pred": speed_corr,
        "mae_m": float(mean_absolute_error(y_true, y_pred)),
    }


def run_lovo_covariate_baseline(
    df: pd.DataFrame,
    covariates: list[str],
    target: TargetCol = "length_m",
) -> tuple[float, pd.DataFrame]:
    """LOVO MAE from affine model y ~ 1 + covariates (sanity-check baselines)."""
    covariates = [c for c in covariates if c in df.columns]
    pred_rows: list[dict] = []
    errors: list[float] = []

    for vehicle, train_idx, test_idx in lovo_splits(df):
        train = df.loc[train_idx]
        test = df.loc[test_idx]
        X_train = np.column_stack([np.ones(len(train)), train[covariates].astype(float).values])
        y_train = train[target].astype(float).values
        X_test = np.column_stack([np.ones(len(test)), test[covariates].astype(float).values])
        y_test = test[target].astype(float).values

        beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
        y_pred = X_test @ beta
        errors.extend(np.abs(y_test - y_pred))

        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "clip_id": df.loc[idx, "clip_id"],
                    "vehicle": vehicle,
                    "y_true": float(y_test[i]),
                    "y_pred": float(y_pred[i]),
                    "abs_error": float(abs(y_pred[i] - y_test[i])),
                }
            )

    return float(np.mean(errors)), pd.DataFrame(pred_rows)


def run_lovo_affine(
    df: pd.DataFrame,
    feature: str,
    target: TargetCol = "length_m",
) -> tuple[float, pd.DataFrame]:
    """LOVO affine calibration: target ≈ a + b·feature (physics proxy baseline)."""
    if feature not in df.columns:
        return float("nan"), pd.DataFrame()

    pred_rows: list[dict] = []
    errors: list[float] = []

    for vehicle, train_idx, test_idx in lovo_splits(df):
        train = df.loc[train_idx]
        test = df.loc[test_idx]
        x_tr = train[feature].astype(float).values
        y_tr = train[target].astype(float).values
        x_te = test[feature].astype(float).values
        y_te = test[target].astype(float).values

        X_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
        X_te = np.column_stack([np.ones(len(x_te)), x_te])
        beta, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        y_pred = X_te @ beta
        errors.extend(np.abs(y_te - y_pred))

        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "clip_id": df.loc[idx, "clip_id"],
                    "vehicle": vehicle,
                    "y_true": float(y_te[i]),
                    "y_pred": float(y_pred[i]),
                    "abs_error": float(abs(y_pred[i] - y_te[i])),
                }
            )

    return float(np.mean(errors)), pd.DataFrame(pred_rows)


def select_features_on_train(
    train: pd.DataFrame,
    target: TargetCol,
    controls: list[str] | None = None,
    k: int = 8,
) -> list[str]:
    """Pick top-|partial_r| features using training fold only."""
    controls = controls or CONTROL_COLS
    scored: list[tuple[str, float]] = []
    for feat in feature_columns(train):
        pc = partial_correlation_multi(train, feat, target, controls)
        if np.isfinite(pc):
            scored.append((feat, abs(pc)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scored[:k]]


def run_lovo_ridge_cv(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: TargetCol = "length_m",
    *,
    select_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """LOVO RidgeCV; optionally select top-k partial-r features per fold on train."""
    base_features = features or feature_columns(df)
    base_features = [f for f in base_features if f in df.columns]
    target_range = float(df[target].max() - df[target].min())

    fold_rows: list[dict] = []
    pred_rows: list[dict] = []

    for vehicle, train_idx, test_idx in lovo_splits(df):
        train = df.loc[train_idx]
        test = df.loc[test_idx]
        feats = select_features_on_train(train, target, k=select_k) if select_k else base_features
        feats = [f for f in feats if f in train.columns]
        if not feats:
            continue

        X_train = train[feats].astype(float).values
        y_train = train[target].astype(float).values
        X_test = test[feats].astype(float).values
        y_test = test[target].astype(float).values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg = RidgeCV(alphas=RIDGE_ALPHAS)
        reg.fit(X_train_s, y_train)
        y_pred = reg.predict(X_test_s)

        m = regression_metrics(y_test, y_pred, target_range)
        m["vehicle"] = vehicle
        m["model"] = "ridge_cv"
        m["n_features"] = len(feats)
        m["alpha"] = float(reg.alpha_)
        fold_rows.append(m)

        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "clip_id": df.loc[idx, "clip_id"],
                    "vehicle": vehicle,
                    "y_true": float(y_test[i]),
                    "y_pred": float(y_pred[i]),
                    "error": float(y_pred[i] - y_test[i]),
                    "abs_error": float(abs(y_pred[i] - y_test[i])),
                }
            )

    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)
    mae = float(folds["mae"].mean()) if len(folds) else float("nan")
    return folds, preds, mae


def _model_result(name: str, mae: float, preds: pd.DataFrame, extra: dict | None = None) -> dict:
    row = {"model": name, "lovo_mae": mae, "preds": preds}
    if extra:
        row.update(extra)
    return row


def baseline_mean_mae(df: pd.DataFrame, target: TargetCol = "length_m") -> float:
    """LOVO MAE when always predicting the training-set mean."""
    errors = []
    for _, train_idx, test_idx in lovo_splits(df):
        mean_val = df.loc[train_idx, target].mean()
        y_test = df.loc[test_idx, target].values
        errors.extend(np.abs(y_test - mean_val))
    return float(np.mean(errors))


def run_mean_baseline_predictions(
    df: pd.DataFrame,
    target: TargetCol = "length_m",
) -> tuple[float, pd.DataFrame]:
    """LOVO mean baseline with per-clip prediction rows."""
    pred_rows: list[dict] = []
    errors: list[float] = []

    for vehicle, train_idx, test_idx in lovo_splits(df):
        mean_val = float(df.loc[train_idx, target].mean())
        test = df.loc[test_idx]
        y_test = test[target].astype(float).values
        y_pred = np.full(len(y_test), mean_val)
        errors.extend(np.abs(y_test - y_pred))

        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "clip_id": df.loc[idx, "clip_id"],
                    "vehicle": vehicle,
                    "y_true": float(y_test[i]),
                    "y_pred": float(y_pred[i]),
                    "abs_error": float(abs(y_pred[i] - y_test[i])),
                }
            )

    return float(np.mean(errors)), pd.DataFrame(pred_rows)


def _verdict_lovo(mae: float, baseline: float) -> str:
    if mae < baseline * 0.95:
        return "BELOW baseline (generalises somewhat)"
    if mae <= baseline * 1.05:
        return "AT baseline (no clear signal under LOVO)"
    return "ABOVE baseline (worse than guessing mean)"


def run_phase_a(
    features_df: pd.DataFrame,
    output_dir: Path,
    target: TargetCol = "length_m",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corr = correlation_report(features_df, target=target, method="spearman")
    corr.to_csv(output_dir / f"correlation_{target}.csv", index=False)

    partial_df = partial_correlation_report(features_df, target=target, controls=CONTROL_COLS, top_n=0)
    partial_df.to_csv(output_dir / f"partial_corr_{target}.csv", index=False)

    groups = feature_groups(features_df)
    baseline = baseline_mean_mae(features_df, target)
    _, mean_preds = run_mean_baseline_predictions(features_df, target)

    candidates: list[dict] = [
        _model_result("mean_baseline", baseline, mean_preds),
    ]

    sanity: dict[str, float] = {}
    for cov_name, covs in [("speed_only", ["speed_kmh"]), ("power_only", ["power_kw"])]:
        mae, preds = run_lovo_covariate_baseline(features_df, covs, target)
        sanity[cov_name] = mae

    # Physics affine proxies: L ≈ a + b·(v·Δt)
    affine_rows = []
    for feat in PHYSICS_PROXIES:
        if feat not in features_df.columns:
            continue
        mae, preds = run_lovo_affine(features_df, feat, target)
        affine_rows.append({"feature": feat, "lovo_mae": mae})
        candidates.append(_model_result(f"affine:{feat}", mae, preds, {"feature": feat}))
    if affine_rows:
        pd.DataFrame(affine_rows).sort_values("lovo_mae").to_csv(
            output_dir / f"physics_affine_{target}.csv", index=False
        )

    # RidgeCV on curated feature groups + per-fold partial-r selection
    ridge_configs: list[tuple[str, list[str] | None, int | None]] = [
        ("ridge_cv_envelope_duration", groups["envelope_duration"], None),
        ("ridge_cv_envelope_speed", groups["envelope_speed"], None),
        ("ridge_cv_physics_core", groups["physics_core"], None),
        ("ridge_cv_speed_normalized", groups["speed_normalized"], None),
        ("ridge_cv_all_features", groups["all_features"], None),
        ("ridge_cv_top8_partial", None, 8),
    ]
    for name, feats, select_k in ridge_configs:
        if feats is not None and not feats:
            continue
        folds, preds, mae = run_lovo_ridge_cv(features_df, feats, target, select_k=select_k)
        if preds.empty:
            continue
        folds.to_csv(output_dir / f"lovo_{name}_{target}.csv", index=False)
        preds.to_csv(output_dir / f"lovo_{name}_preds_{target}.csv", index=False)
        candidates.append(_model_result(name, mae, preds))

    # Legacy RF on speed-normalised features (interpretability)
    if groups["speed_normalized"]:
        rf_folds, rf_preds = run_lovo_regression(
            features_df, groups["speed_normalized"], target, model="rf"
        )
        rf_folds.to_csv(output_dir / f"lovo_rf_speed_normalized_{target}.csv", index=False)
        rf_preds.to_csv(output_dir / f"lovo_rf_preds_speed_normalized_{target}.csv", index=False)
        candidates.append(
            _model_result("rf_speed_normalized", float(rf_folds["mae"].mean()), rf_preds)
        )

    best = min(candidates, key=lambda c: c["lovo_mae"])
    physics_candidates = [c for c in candidates if c["model"] != "mean_baseline"]
    best_physics = min(physics_candidates, key=lambda c: c["lovo_mae"]) if physics_candidates else best

    best_preds = best["preds"]
    report_preds = best_physics["preds"] if best["model"] == "mean_baseline" else best_preds
    bunch = bunching_diagnostics(best_preds, features_df, target)
    bunch_physics = bunching_diagnostics(report_preds, features_df, target)

    best_preds.to_csv(output_dir / f"best_lovo_preds_{target}.csv", index=False)
    report_preds.to_csv(output_dir / f"clip_model_preds_{target}.csv", index=False)

    summary = {
        "target": target,
        "n_clips": len(features_df),
        "n_vehicles": features_df["vehicle"].nunique(),
        "target_range_m": float(features_df[target].max() - features_df[target].min()),
        "baseline_lovo_mae": baseline,
        "best_model": best["model"],
        "best_lovo_mae": best["lovo_mae"],
        "best_beats_baseline": best["lovo_mae"] < baseline,
        "best_physics_model": best_physics["model"],
        "best_physics_lovo_mae": best_physics["lovo_mae"],
        "verdict": _verdict_lovo(best["lovo_mae"], baseline),
        "top_feature_spearman": corr.iloc[0]["feature"] if len(corr) else None,
        "top_spearman_r": float(corr.iloc[0]["r"]) if len(corr) else None,
        "top_feature_partial_r": partial_df.iloc[0]["feature"] if len(partial_df) else None,
        "top_partial_r": float(partial_df.iloc[0]["partial_r"]) if len(partial_df) else None,
        "bunching": bunch,
        "bunching_physics": bunch_physics,
        "sanity_baselines": sanity,
        "all_models": {c["model"]: c["lovo_mae"] for c in candidates},
        # Back-compat keys for report
        "ridge_lovo_mae_mean": best["lovo_mae"] if best["model"].startswith("ridge") else None,
        "rf_lovo_mae_mean": next(
            (c["lovo_mae"] for c in candidates if c["model"] == "rf_speed_normalized"), None
        ),
    }
    (output_dir / f"phase_a_summary_{target}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_predictions(
        report_preds,
        output_dir / f"scatter_best_{target}.png",
        target,
    )
    plot_feature_vs_length(
        features_df,
        partial_df.head(6)["feature"].tolist(),
        target,
        output_dir / f"top_features_{target}.png",
    )

    return {"correlation": corr, "partials": partial_df, "best": best, "summary": summary}


def plot_predictions(preds: pd.DataFrame, out_path: Path, target: TargetCol) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=preds, x="y_true", y="y_pred", hue="vehicle", ax=ax, legend=False, s=40, alpha=0.7)
    lims = [preds[["y_true", "y_pred"]].min().min(), preds[["y_true", "y_pred"]].max().max()]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"LOVO best model — {target}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_vs_length(
    df: pd.DataFrame,
    features: list[str],
    target: TargetCol,
    out_path: Path,
) -> None:
    n = len(features)
    if n == 0:
        return
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feat in zip(axes, features):
        sns.regplot(data=df, x=feat, y=target, scatter_kws={"s": 15, "alpha": 0.5}, ax=ax, line_kws={"color": "red"})
        ax.set_title(feat[:40], fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
