"""JASA week length ablations: L vs Wb (Ablation 1) → Δt proxy (Ablation 2) → final Wb→L rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from length_estimation.config import DEFAULT_OUTPUT_DIR
from length_estimation.src.evaluate import (
    bunching_diagnostics,
    lovo_splits,
    regression_metrics,
    run_lovo_affine,
)

ENV_PROXY = "env_10db_width_x_speed_m"
DOPPLER_PROXY = "reassigned_doppler_transition_width_x_speed_m"
REQUIRED_COLS = (ENV_PROXY, DOPPLER_PROXY, "length_m", "wheelbase_m", "vehicle", "clip_id")


def load_features(features_path: Path | None = None) -> pd.DataFrame:
    path = Path(features_path) if features_path else DEFAULT_OUTPUT_DIR / "features.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Features not found: {path}\nRun: python -m length_estimation.run features"
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def _mean_baseline_lovo(df: pd.DataFrame, target: str) -> tuple[float, pd.DataFrame]:
    pred_rows: list[dict] = []
    errors: list[float] = []
    for vehicle, train_idx, test_idx in lovo_splits(df):
        mu = float(df.loc[train_idx, target].astype(float).mean())
        y_te = df.loc[test_idx, target].astype(float).values
        errors.extend(np.abs(y_te - mu))
        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "clip_id": df.loc[idx, "clip_id"],
                    "vehicle": vehicle,
                    "y_true": float(y_te[i]),
                    "y_pred": mu,
                    "abs_error": float(abs(y_te[i] - mu)),
                }
            )
    return float(np.mean(errors)), pd.DataFrame(pred_rows)


def _metrics_from_preds(preds: pd.DataFrame, target_range: float) -> dict[str, float]:
    y_true = preds["y_true"].to_numpy(dtype=float)
    y_pred = preds["y_pred"].to_numpy(dtype=float)
    return regression_metrics(y_true, y_pred, target_range)


def _affine_full(df: pd.DataFrame, feature: str, target: str) -> dict[str, Any]:
    mae, preds = run_lovo_affine(df, feature, target)  # type: ignore[arg-type]
    tr = float(df[target].max() - df[target].min())
    m = _metrics_from_preds(preds, tr)
    bunch = bunching_diagnostics(preds, df, target_col=target)  # type: ignore[arg-type]
    return {
        "feature": feature,
        "target": target,
        "lovo_mae": float(mae),
        **m,
        "pred_std_ratio": bunch.get("pred_std_ratio"),
        "corr_speed_pred": bunch.get("corr_speed_pred"),
        "preds": preds,
    }


def ablation1(df: pd.DataFrame, *, out_dir: Path | None = None) -> dict[str, Any]:
    """Ablation 1: env−10 dB×v proxy → length_m vs wheelbase_m under strict LOVO."""
    out = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR / "jasa_week"
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for target in ("length_m", "wheelbase_m"):
        mean_mae, mean_preds = _mean_baseline_lovo(df, target)
        tr = float(df[target].max() - df[target].min())
        mean_m = _metrics_from_preds(mean_preds, tr)
        aff = _affine_full(df, ENV_PROXY, target)
        results[target] = {
            "mean_baseline": {"lovo_mae": mean_mae, **mean_m},
            "affine_env10db": {k: v for k, v in aff.items() if k != "preds"},
            "beats_mean": aff["lovo_mae"] + 1e-9 < mean_mae,
        }
        aff["preds"].to_csv(out / f"ablation1_{target}_preds.csv", index=False)

    wb_ok = results["wheelbase_m"]["beats_mean"]
    len_ok = results["length_m"]["beats_mean"]
    if wb_ok and not len_ok:
        winner = "wheelbase_m"
        rationale = "Wheelbase beats mean baseline; length does not → estimate axle spacing first."
    elif wb_ok and len_ok:
        wb_imp = (
            results["wheelbase_m"]["mean_baseline"]["lovo_mae"]
            - results["wheelbase_m"]["affine_env10db"]["lovo_mae"]
        )
        len_imp = (
            results["length_m"]["mean_baseline"]["lovo_mae"]
            - results["length_m"]["affine_env10db"]["lovo_mae"]
        )
        winner = "wheelbase_m" if wb_imp >= len_imp else "length_m"
        rationale = f"Both beat mean; choose larger absolute MAE drop ({winner})."
    elif len_ok:
        winner = "length_m"
        rationale = "Only length beats mean baseline."
    else:
        winner = "wheelbase_m"
        rationale = (
            "Neither beats mean; default to wheelbase for geometry narrative + report negative length result."
        )

    payload = {
        "length_m": results["length_m"],
        "wheelbase_m": results["wheelbase_m"],
        "decision": {"winning_target": winner, "rationale": rationale},
        "proxy": ENV_PROXY,
        "n_clips": int(len(df)),
        "n_vehicles": int(df["vehicle"].nunique()),
    }
    (out / "ablation1_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "ablation1_table.md").write_text(format_ablation1_table(payload), encoding="utf-8")
    return payload


def format_ablation1_table(a1: dict[str, Any]) -> str:
    """Markdown table for paper / weekly notes."""
    proxy = a1.get("proxy", ENV_PROXY)

    def row(target: str, model: str, key: str) -> str:
        block = a1[target][key]
        mae = block["lovo_mae"]
        r2 = block.get("r2", float("nan"))
        beats = "yes" if a1[target].get("beats_mean") and key == "affine_env10db" else "—"
        return f"| {target} | {model} | `{proxy}` | {mae:.3f} | {r2:.3f} | {beats} |"

    lines = [
        "# Ablation 1 — target: length vs wheelbase",
        "",
        f"Proxy: **`{proxy}`** (envelope −10 dB width × speed). Evaluation: LOVO affine `y ≈ a + b·proxy`.",
        "",
        "| Target | Model | Proxy | LOVO MAE (m) | R² | Beats mean? |",
        "|--------|-------|-------|-------------:|---:|:-----------:|",
        row("length_m", "mean baseline", "mean_baseline"),
        row("length_m", "affine env−10 dB×v", "affine_env10db"),
        row("wheelbase_m", "mean baseline", "mean_baseline"),
        row("wheelbase_m", "affine env−10 dB×v", "affine_env10db"),
        "",
        f"**Decision:** `{a1['decision']['winning_target']}` — {a1['decision']['rationale']}",
        "",
    ]
    return "\n".join(lines)


def ablation2(df: pd.DataFrame, target: str, *, out_dir: Path | None = None) -> dict[str, Any]:
    out = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR / "jasa_week"
    out.mkdir(parents=True, exist_ok=True)

    mean_mae, _ = _mean_baseline_lovo(df, target)
    arms: dict[str, dict] = {}
    for name, feat in (("envelope", ENV_PROXY), ("doppler", DOPPLER_PROXY)):
        aff = _affine_full(df, feat, target)
        arms[name] = {k: v for k, v in aff.items() if k != "preds"}
        aff["preds"].to_csv(out / f"ablation2_{target}_{name}_preds.csv", index=False)

    def key(name: str) -> tuple:
        a = arms[name]
        return (
            a["lovo_mae"],
            -float(a.get("pred_std_ratio") or 0.0),
            abs(float(a.get("corr_speed_pred") or 0.0)),
        )

    winner = min(arms.keys(), key=key)
    payload = {
        "target": target,
        "mean_baseline_mae": mean_mae,
        "arms": arms,
        "decision": {
            "winning_rule": winner,
            "delta_t_star_feature": arms[winner]["feature"],
            "lovo_mae": arms[winner]["lovo_mae"],
        },
    }
    (out / "ablation2_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def final_wb_to_l(df: pd.DataFrame, wb_feature: str, *, out_dir: Path | None = None) -> dict[str, Any]:
    """Fold-wise: calibrate Wb from feature, then L from predicted Wb (train vehicles only)."""
    out = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR / "jasa_week"
    out.mkdir(parents=True, exist_ok=True)

    pred_rows: list[dict] = []
    for vehicle, train_idx, test_idx in lovo_splits(df):
        train = df.loc[train_idx]
        test = df.loc[test_idx]

        x_tr = train[wb_feature].astype(float).values
        wb_tr = train["wheelbase_m"].astype(float).values
        X1 = np.column_stack([np.ones(len(x_tr)), x_tr])
        beta_wb, _, _, _ = np.linalg.lstsq(X1, wb_tr, rcond=None)

        x_te = test[wb_feature].astype(float).values
        wb_hat_te = np.column_stack([np.ones(len(x_te)), x_te]) @ beta_wb

        train_veh = train.groupby("vehicle", as_index=False)[["wheelbase_m", "length_m"]].mean()
        X2 = np.column_stack(
            [np.ones(len(train_veh)), train_veh["wheelbase_m"].astype(float).values]
        )
        beta_l, _, _, _ = np.linalg.lstsq(
            X2, train_veh["length_m"].astype(float).values, rcond=None
        )

        L_hat_te = np.column_stack([np.ones(len(wb_hat_te)), wb_hat_te]) @ beta_l
        L_te = test["length_m"].astype(float).values
        wb_te = test["wheelbase_m"].astype(float).values

        for i, idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "clip_id": df.loc[idx, "clip_id"],
                    "vehicle": vehicle,
                    "wb_true": float(wb_te[i]),
                    "wb_pred": float(wb_hat_te[i]),
                    "L_true": float(L_te[i]),
                    "L_pred": float(L_hat_te[i]),
                    "wb_abs_error": float(abs(wb_hat_te[i] - wb_te[i])),
                    "L_abs_error": float(abs(L_hat_te[i] - L_te[i])),
                }
            )

    preds = pd.DataFrame(pred_rows)
    wb_range = float(df["wheelbase_m"].max() - df["wheelbase_m"].min())
    L_range = float(df["length_m"].max() - df["length_m"].min())
    wb_m = regression_metrics(preds["wb_true"].values, preds["wb_pred"].values, wb_range)
    L_m = regression_metrics(preds["L_true"].values, preds["L_pred"].values, L_range)

    per_veh = (
        preds.groupby("vehicle")
        .agg(
            n=("clip_id", "count"),
            wb_mae=("wb_abs_error", "mean"),
            L_mae=("L_abs_error", "mean"),
            wb_true=("wb_true", "mean"),
            L_true=("L_true", "mean"),
            wb_pred=("wb_pred", "mean"),
            L_pred=("L_pred", "mean"),
        )
        .reset_index()
    )

    preds.to_csv(out / "final_rule_preds.csv", index=False)
    per_veh.to_csv(out / "final_rule_per_vehicle.csv", index=False)

    return {
        "wb_feature": wb_feature,
        "wheelbase": wb_m,
        "length": L_m,
        "preds": preds,
        "per_vehicle": per_veh,
    }


def run_ablation1(
    features_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    df = load_features(features_path)
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / "jasa_week"
    result = ablation1(df, out_dir=out)
    print("=== Ablation 1 ===")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out / 'ablation1_summary.json'}")
    print(f"Wrote {out / 'ablation1_table.md'}")
    return result


def run_full_pipeline(
    features_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    df = load_features(features_path)
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR / "jasa_week"

    a1 = ablation1(df, out_dir=out)
    print("=== Ablation 1 ===")
    print(json.dumps(a1, indent=2))

    target = a1["decision"]["winning_target"]
    a2 = ablation2(df, target, out_dir=out)
    print("=== Ablation 2 ===")
    print(json.dumps(a2, indent=2))

    feat_star = a2["decision"]["delta_t_star_feature"]

    if target == "wheelbase_m":
        final = final_wb_to_l(df, feat_star, out_dir=out)
        final_summary = {
            "formula": (
                f"Wb_hat = a0 + b0 * {feat_star}; "
                "L_hat = a + b * Wb_hat (fold-wise vehicle catalog map)"
            ),
            "wb_feature": feat_star,
            "wheelbase_lovo": final["wheelbase"],
            "length_lovo": final["length"],
            "ablation1_winner": target,
            "ablation2_winner": a2["decision"]["winning_rule"],
            "cnn_ceiling_length_mae": 0.0973,
            "mean_baseline_length_mae": a1["length_m"]["mean_baseline"]["lovo_mae"],
            "mean_baseline_wheelbase_mae": a1["wheelbase_m"]["mean_baseline"]["lovo_mae"],
            "ablation1_length_affine_mae": a1["length_m"]["affine_env10db"]["lovo_mae"],
            "ablation2_loser": (
                "doppler" if a2["decision"]["winning_rule"] == "envelope" else "envelope"
            ),
            "ablation2_loser_mae": a2["arms"][
                "doppler" if a2["decision"]["winning_rule"] == "envelope" else "envelope"
            ]["lovo_mae"],
        }
    else:
        aff = _affine_full(df, feat_star, "length_m")
        aff["preds"].to_csv(out / "final_rule_preds.csv", index=False)
        final_summary = {
            "formula": f"L_hat = a + b * {feat_star}",
            "length_lovo": {k: aff[k] for k in ("mae", "rmse", "r2", "norm_mae", "lovo_mae")},
            "ablation1_winner": target,
            "ablation2_winner": a2["decision"]["winning_rule"],
            "cnn_ceiling_length_mae": 0.0973,
        }

    (out / "final_rule_summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    print("=== Final ===")
    print(json.dumps(final_summary, indent=2))
    print(f"Artifacts under {out}")
