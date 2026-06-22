"""Evaluation for length CNN — run automatically after training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from length_estimation.config import DEFAULT_OUTPUT_DIR, PhaseBConfig
from length_estimation.src.evaluate import bunching_diagnostics, lovo_splits, regression_metrics
from length_estimation.src.phase_b.metrics import enrich_with_vehicle_id, summarize_predictions
from length_estimation.src.phase_b.dataset import collate_batch, make_dataset
from length_estimation.src.phase_b.train import load_checkpoint, resolve_device
from length_estimation.src.preprocess import ClipRecord, load_clips, resolve_data_dir


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Phase B requires PyTorch.") from exc
    return torch, DataLoader


def predict_records(model, records: list[ClipRecord], cfg: PhaseBConfig, device: str) -> pd.DataFrame:
    torch, DataLoader = _require_torch()
    device = resolve_device(device)
    model.eval()

    ds = make_dataset(records, cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)

    rows: list[dict] = []
    with torch.no_grad():
        for x, y, speed, metas in loader:
            x = x.to(device)
            speed_t = speed.to(device) if speed is not None else None
            pred = model(x, speed_t)
            for i, meta in enumerate(metas):
                rows.append(
                    {
                        "clip_id": meta["clip_id"],
                        "vehicle": meta["vehicle"],
                        "speed_kmh": meta["speed_kmh"],
                        "split": meta["split"],
                        "y_true": float(y[i].item()),
                        "y_pred": float(pred[i].item()),
                        "abs_error": float(abs(pred[i].item() - y[i].item())),
                    }
                )
    return pd.DataFrame(rows)


def _vehicle_summary(preds: pd.DataFrame) -> pd.DataFrame:
    g = preds.groupby("vehicle", sort=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "actual_length_m": g["y_true"].first(),
            "mean_pred_m": g["y_pred"].mean(),
            "mae_m": g["abs_error"].mean(),
            "pred_std_m": g["y_pred"].std(),
        }
    ).round(4)


def _write_eval_report(
    out_dir: Path,
    preds: pd.DataFrame,
    metrics: dict,
    cfg: PhaseBConfig,
    checkpoint: Path,
    *,
    phase_a_baseline_mae: float | None = None,
    ckpt_meta: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    veh = _vehicle_summary(preds)
    feat_df = preds[["clip_id", "vehicle", "speed_kmh"]].copy()
    feat_df["length_m"] = preds["y_true"].values
    bunch = bunching_diagnostics(
        preds[["clip_id", "vehicle", "y_true", "y_pred"]],
        feat_df,
        "length_m",
        "y_pred",
    )

    cls = summarize_predictions(preds)
    enriched = enrich_with_vehicle_id(preds)
    enriched.to_csv(out_dir / "eval_predictions.csv", index=False)
    veh.to_csv(out_dir / "eval_vehicle_summary.csv")
    (out_dir / "eval_metrics.json").write_text(
        json.dumps({**metrics, "bunching": bunch, "classification": cls}, indent=2),
        encoding="utf-8",
    )

    beats_a = phase_a_baseline_mae is not None and metrics["mae"] < phase_a_baseline_mae
    best_epoch = (ckpt_meta or {}).get("epoch")
    train_summary_path = checkpoint.parent / "train_summary.json"
    if best_epoch is None and train_summary_path.exists():
        best_epoch = json.loads(train_summary_path.read_text(encoding="utf-8")).get("best_epoch")

    lines = [
        "=" * 72,
        "PHASE B — LENGTH CNN EVALUATION",
        "=" * 72,
        "",
        f"  Checkpoint : {checkpoint}",
        f"  Spec type  : {cfg.spec_type}",
        f"  Speed aux  : {cfg.include_speed}",
    ]
    if best_epoch is not None:
        lines.append(f"  Best epoch : {best_epoch}  (weights in best.pt)")
    lines.extend(
        [
            "",
            "OVERALL (all clips in this eval split)",
            "-" * 72,
            f"  N clips              : {cls['n_clips']}",
            f"  MAE (length)         : {cls['mae_m']:.4f} m",
            f"  RMSE (length)        : {cls['rmse_m']:.4f} m",
            f"  R²                   : {metrics.get('r2', float('nan')):.4f}",
            f"  Vehicle ID accuracy  : {cls['vehicle_id_accuracy']:.1%}  "
            f"({cls['vehicle_id_correct']}/{cls['vehicle_id_total']} clips)",
            "    (nearest catalog length to predicted length → vehicle name)",
            "",
            "DIAGNOSTICS",
            "-" * 72,
            f"  Pred std ratio       : {bunch['pred_std_ratio']:.1%}",
            f"  corr(speed,pred)     : {bunch['corr_speed_pred']:.3f}",
        ]
    )
    if phase_a_baseline_mae is not None:
        lines.append(f"  Phase A LOVO baseline: {phase_a_baseline_mae:.4f} m")
        lines.append(f"  Beats Phase A baseline: {'YES' if beats_a else 'NO'}")
    lines.extend(
        [
            "",
            "PER-VEHICLE (length MAE)",
            "-" * 72,
            veh.to_string(),
            "",
            "OUTPUT FILES",
            "  eval_predictions.csv   (includes predicted_vehicle, vehicle_correct)",
            "  eval_vehicle_summary.csv",
            "  eval_metrics.json",
            "  eval_summary.txt",
            "=" * 72,
        ]
    )
    (out_dir / "eval_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_eval_split(
    checkpoint: Path,
    data_dir=None,
    output_dir: Path | None = None,
    device: str = "auto",
    split: str = "valid",
) -> pd.DataFrame:
    """Evaluate checkpoint on VS13 valid (or train) split."""
    model, cfg, ckpt = load_checkpoint(checkpoint, resolve_device(device))
    records = load_clips(resolve_data_dir(data_dir))
    eval_records = [r for r in records if r.split == split]
    if not eval_records:
        raise RuntimeError(f"No clips with split='{split}'")

    preds = predict_records(model, eval_records, cfg, device)
    target_range = float(preds["y_true"].max() - preds["y_true"].min())
    m = regression_metrics(preds["y_true"].values, preds["y_pred"].values, target_range)
    m["n_clips"] = len(preds)
    m["eval_split"] = split
    m["checkpoint"] = str(checkpoint)

    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR) / "phase_b" / checkpoint.parent.name
    phase_a_mae = _load_phase_a_baseline()
    _write_eval_report(
        out_dir, preds, m, cfg, checkpoint,
        phase_a_baseline_mae=phase_a_mae,
        ckpt_meta=ckpt,
    )
    cls = summarize_predictions(preds)
    print(
        f"Eval ({split}): MAE={m['mae']:.4f}m  "
        f"vehicle-ID acc={cls['vehicle_id_accuracy']:.1%}  "
        f"-> {out_dir}/eval_summary.txt"
    )
    return preds


def run_eval_lovo(
    run_dir: Path,
    data_dir=None,
    output_dir: Path | None = None,
    device: str = "auto",
) -> pd.DataFrame:
    """Evaluate all LOVO fold checkpoints in a run directory."""
    run_dir = Path(run_dir)
    data_dir = resolve_data_dir(data_dir)
    records = load_clips(data_dir)
    by_id = {r.clip_id: r for r in records}
    df = pd.DataFrame([{"clip_id": r.clip_id, "vehicle": r.vehicle, "length_m": r.length_m} for r in records])
    target_range = float(df["length_m"].max() - df["length_m"].min())

    pred_rows: list[dict] = []
    fold_maes: list[float] = []

    for vehicle, _train_idx, test_idx in lovo_splits(df):
        ckpt = run_dir / f"fold_{vehicle}.pt"
        if not ckpt.exists():
            print(f"WARN missing checkpoint: {ckpt}")
            continue
        model, cfg, _ = load_checkpoint(ckpt, resolve_device(device))
        test_records = [by_id[cid] for cid in df.loc[test_idx, "clip_id"]]
        fold_preds = predict_records(model, test_records, cfg, device)
        fold_preds["held_out_vehicle"] = vehicle
        pred_rows.append(fold_preds)
        fold_maes.append(float(fold_preds["abs_error"].mean()))

    preds = pd.concat(pred_rows, ignore_index=True)
    m = regression_metrics(preds["y_true"].values, preds["y_pred"].values, target_range)
    m["n_clips"] = len(preds)
    m["lovo_fold_mae_mean"] = float(np.mean(fold_maes))
    m["eval_mode"] = "lovo"
    m["run_dir"] = str(run_dir)

    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR) / "phase_b" / run_dir.name
    phase_a_mae = _load_phase_a_baseline()
    _write_eval_report(out_dir, preds, m, cfg, run_dir, phase_a_baseline_mae=phase_a_mae)
    cls = summarize_predictions(preds)
    print(
        f"Eval (LOVO): MAE={m['mae']:.4f}m  "
        f"vehicle-ID acc={cls['vehicle_id_accuracy']:.1%}  "
        f"fold_mean={m['lovo_fold_mae_mean']:.4f}m"
    )
    return preds


def _load_phase_a_baseline() -> float | None:
    path = DEFAULT_OUTPUT_DIR / "phase_a" / "phase_a_summary_length_m.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("baseline_lovo_mae", data.get("best_lovo_mae")))
    return None


def run_eval(
    checkpoint: Path | None = None,
    run_dir: Path | None = None,
    data_dir=None,
    output_dir: Path | None = None,
    device: str = "auto",
    split: str = "valid",
) -> pd.DataFrame:
    """
    Entry point called after training.
    - checkpoint=...  -> split eval (default valid)
    - run_dir=...     -> LOVO eval over fold_*.pt
    """
    if run_dir is not None:
        return run_eval_lovo(run_dir, data_dir, output_dir, device)
    if checkpoint is None:
        raise ValueError("Provide checkpoint (split mode) or run_dir (LOVO mode)")
    return run_eval_split(checkpoint, data_dir, output_dir, device, split=split)
