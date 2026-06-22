"""Inference: predict vehicle length from a pass-by wav clip."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from length_estimation.config import DEFAULT_OUTPUT_DIR
from length_estimation.src.phase_b.dataset import load_length_spec, normalise_speed_kmh
from length_estimation.src.phase_b.metrics import enrich_with_vehicle_id, summarize_predictions
from length_estimation.src.phase_b.train import _require_torch, load_checkpoint, resolve_device
from length_estimation.src.preprocess import ClipRecord, _parse_annotation, load_clips


@dataclass
class LengthPrediction:
    length_m: float
    speed_kmh: float
    clip_id: str | None = None
    vehicle: str | None = None
    actual_length_m: float | None = None
    predicted_vehicle: str | None = None
    vehicle_correct: bool | None = None


def predictions_to_dataframe(preds: list[LengthPrediction]) -> pd.DataFrame:
    rows = []
    for p in preds:
        rows.append(
            {
                "clip_id": p.clip_id,
                "vehicle": p.vehicle,
                "speed_kmh": p.speed_kmh,
                "y_true": p.actual_length_m,
                "y_pred": p.length_m,
                "abs_error": abs(p.length_m - p.actual_length_m)
                if p.actual_length_m is not None
                else None,
                "predicted_vehicle": p.predicted_vehicle,
                "vehicle_correct": p.vehicle_correct,
            }
        )
    return pd.DataFrame(rows)


def _attach_classification(pred: LengthPrediction, pred_col: str = "y_pred") -> LengthPrediction:
    if pred.vehicle is None or pred.actual_length_m is None:
        return pred
    row = pd.DataFrame(
        [{"vehicle": pred.vehicle, "y_true": pred.actual_length_m, "y_pred": pred.length_m}]
    )
    enriched = enrich_with_vehicle_id(row, pred_col=pred_col)
    return LengthPrediction(
        length_m=pred.length_m,
        speed_kmh=pred.speed_kmh,
        clip_id=pred.clip_id,
        vehicle=pred.vehicle,
        actual_length_m=pred.actual_length_m,
        predicted_vehicle=str(enriched.iloc[0]["predicted_vehicle"]),
        vehicle_correct=bool(enriched.iloc[0]["vehicle_correct"]),
    )


def predict_wav(
    wav_path: Path,
    checkpoint: Path,
    *,
    speed_kmh: float,
    cpa_time_s: float,
    device: str = "auto",
) -> LengthPrediction:
    """Predict length from a single wav (CPA time and speed required)."""
    torch, _, _ = _require_torch()

    model, cfg, _ = load_checkpoint(checkpoint, resolve_device(device))
    device = resolve_device(device)

    record = ClipRecord(
        clip_id=Path(wav_path).stem,
        vehicle="unknown",
        speed_kmh=speed_kmh,
        cpa_time_s=cpa_time_s,
        wav_path=Path(wav_path),
        length_m=0.0,
        wheelbase_m=0.0,
        power_kw=0.0,
        engine_type="unknown",
    )
    spec = load_length_spec(record, cfg)
    x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)
    speed = (
        torch.tensor([normalise_speed_kmh(speed_kmh, cfg)], dtype=torch.float32, device=device)
        if cfg.include_speed
        else None
    )

    model.eval()
    with torch.no_grad():
        pred = float(model(x, speed).item())

    return LengthPrediction(length_m=pred, speed_kmh=speed_kmh, clip_id=record.clip_id)


def predict_clip(
    record: ClipRecord,
    checkpoint: Path,
    device: str = "auto",
) -> LengthPrediction:
    """Predict length for a ClipRecord."""
    torch, _, _ = _require_torch()

    model, cfg, _ = load_checkpoint(checkpoint, resolve_device(device))
    device = resolve_device(device)
    spec = load_length_spec(record, cfg)
    x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)
    speed = (
        torch.tensor([normalise_speed_kmh(record.speed_kmh, cfg)], dtype=torch.float32, device=device)
        if cfg.include_speed
        else None
    )

    model.eval()
    with torch.no_grad():
        pred = float(model(x, speed).item())

    out = LengthPrediction(
        length_m=pred,
        speed_kmh=record.speed_kmh,
        clip_id=record.clip_id,
        vehicle=record.vehicle,
        actual_length_m=record.length_m,
    )
    return _attach_classification(out)


def predict_from_sidecar(wav_path: Path, checkpoint: Path, device: str = "auto") -> LengthPrediction:
    """Load speed + CPA from adjacent .txt annotation."""
    txt = wav_path.with_suffix(".txt")
    if not txt.exists():
        raise FileNotFoundError(f"Annotation not found: {txt}")
    speed_kmh, cpa_time_s = _parse_annotation(txt)
    return predict_wav(wav_path, checkpoint, speed_kmh=speed_kmh, cpa_time_s=cpa_time_s, device=device)


def predict_dataset(
    checkpoint: Path,
    data_dir=None,
    device: str = "auto",
) -> list[LengthPrediction]:
    """Run inference on all clips under data_dir."""
    records = load_clips(data_dir)
    return [predict_clip(r, checkpoint, device) for r in records]


def run_inference_report(
    checkpoint: Path,
    data_dir=None,
    device: str = "auto",
    output_path: Path | None = None,
    split: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Predict all clips (optionally filtered by split), write CSV + print overall MAE & vehicle-ID accuracy.
    """
    records = load_clips(data_dir)
    if split:
        records = [r for r in records if r.split == split]
    preds = [predict_clip(r, checkpoint, device) for r in records]
    df = predictions_to_dataframe(preds)
    df = enrich_with_vehicle_id(df)

    metrics = summarize_predictions(df)
    out = output_path or (
        DEFAULT_OUTPUT_DIR / "phase_b" / checkpoint.parent.name / f"inference_{split or 'all'}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary_path = out.with_suffix(".txt")
    summary_path.write_text(
        "\n".join(
            [
                "INFERENCE SUMMARY",
                f"  checkpoint : {checkpoint}",
                f"  split      : {split or 'all'}",
                f"  n clips    : {metrics['n_clips']}",
                f"  MAE        : {metrics['mae_m']:.4f} m",
                f"  RMSE       : {metrics['rmse_m']:.4f} m",
                f"  vehicle ID accuracy : {metrics['vehicle_id_accuracy']:.1%} "
                f"({metrics['vehicle_id_correct']}/{metrics['vehicle_id_total']})",
                f"  csv        : {out}",
            ]
        ),
        encoding="utf-8",
    )
    return df, metrics
