#!/usr/bin/env python3
"""Direction inference on VS13 (length_estimation) grouped by vehicle type.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Uses ``cnn.inference``; must preserve identical predictions for baseline checkpoint weights.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from idmt_experiments.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_OUTPUT_DIR,
    DIRECTION_LABELS,
    DirectionConfig,
    SR_NATIVE,
    resolve_checkpoint_file,
)
from idmt_experiments.cnn.inference import predict_wav_mono
from idmt_experiments.cnn.train import load_checkpoint, resolve_device
from idmt_experiments.src.features import load_mono
from idmt_experiments.transfer.eval import predict_wav_fusion, predict_wav_transfer
from idmt_experiments.test_clips import (
    _spec_png_path,
    _validate_checkpoint,
    resolve_prediction,
    save_constant_q_spectrogram,
)

DEFAULT_VS13_DIR = (
    Path(__file__).resolve().parents[3] / "length_estimation" / "data" / "vs13"
)
DEFAULT_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "cnn" / "direction" / "vs13_direction" / "predictions.csv"
)
DEFAULT_CHECKPOINT = resolve_checkpoint_file(
    DirectionConfig(task="direction"),
    "mel_3class",
)
DEFAULT_FUSION_LEFT = (
    DEFAULT_CHECKPOINT_DIR / "transfer" / "direction" / "deep_mel_2class_left_100ep" / "best.pt"
)
DEFAULT_FUSION_RIGHT = (
    DEFAULT_CHECKPOINT_DIR / "transfer" / "direction" / "deep_mel_2class_right_100ep" / "best.pt"
)
DEFAULT_FUSION_METRICS = (
    DEFAULT_OUTPUT_DIR / "fusion" / "direction" / "fusion_2class_100ep" / "eval_metrics.json"
)
DEFAULT_FUSION_VS13_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "fusion" / "direction" / "vs13_direction" / "predictions.csv"
)
_SKIP_STEMS = frozenset({"Train_valid_split"})

ROW_CLIP = "clip"
ROW_GROUP = "group"
ROW_OVERALL = "overall"


@dataclass(frozen=True)
class Vs13Clip:
    vehicle: str
    clip_id: str
    wav_path: Path
    speed_kmh: int | None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run mel direction inference on VS13 audio (vehicle folders under data/vs13). "
            "Writes per-clip predictions plus per-vehicle group summaries and overall L2R/R2L rates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m idmt_experiments.infer_vs13\n"
            "  python -m idmt_experiments.infer_vs13 "
            "--data-dir D:/Antigravity/DopplerLab/length_estimation/data/vs13\n"
            "  python -m idmt_experiments.infer_vs13 --no-spectrograms\n"
            "  python -m idmt_experiments.infer_vs13 --no-direction-only  # allow no_vehicle\n"
        ),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_VS13_DIR,
        help=f"VS13 root with one folder per vehicle (default: {DEFAULT_VS13_DIR})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: fusion vs13_direction or cnn vs13_direction)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Mel/transfer checkpoint (CNN or Phase B). Not used with --fusion.",
    )
    p.add_argument(
        "--fusion",
        action="store_true",
        help="Phase C late fusion (default left+right deep_mel_2class_100ep, w_L from eval_metrics).",
    )
    p.add_argument(
        "--left-checkpoint",
        type=Path,
        default=DEFAULT_FUSION_LEFT,
        help="Left mono checkpoint for --fusion",
    )
    p.add_argument(
        "--right-checkpoint",
        type=Path,
        default=DEFAULT_FUSION_RIGHT,
        help="Right mono checkpoint for --fusion",
    )
    p.add_argument(
        "--fusion-weight",
        type=float,
        default=None,
        help="Left fusion weight w_L (default: read fusion.eval_metrics.json or 0.05)",
    )
    p.add_argument(
        "--fusion-run-name",
        default="fusion_2class_100ep",
        help="Model label written to CSV/JSON when using --fusion",
    )
    p.add_argument(
        "--transfer",
        action="store_true",
        help="Use Phase B transfer inference (single --checkpoint, not --fusion).",
    )
    p.add_argument(
        "--spectrogram-dir",
        type=Path,
        default=None,
        help="Folder for constant-Q PNGs (default: <output-dir>/spectrograms)",
    )
    p.add_argument("--no-spectrograms", action="store_true", help="Skip constant-Q PNG export")
    p.add_argument(
        "--direction-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pick L2R vs R2L only, ignoring no_vehicle (default: on; VS13 has no background clips)."
        ),
    )
    p.add_argument(
        "--flip-direction",
        action="store_true",
        help="Swap L2R <-> R2L after prediction.",
    )
    p.add_argument("--device", default="auto")
    return p


def _parse_speed_kmh(stem: str, vehicle: str) -> int | None:
    prefix = f"{vehicle}_"
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix) :]
    return int(tail) if tail.isdigit() else None


def discover_vs13_clips(data_dir: Path) -> list[Vs13Clip]:
    """Scan ``{data_dir}/{VehicleName}/{VehicleName}_{speed}.wav``."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"VS13 data directory not found: {data_dir}")

    clips: list[Vs13Clip] = []
    for vdir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        vehicle = vdir.name
        for wav_path in sorted(vdir.glob("*.wav")):
            if wav_path.stem in _SKIP_STEMS:
                continue
            clips.append(
                Vs13Clip(
                    vehicle=vehicle,
                    clip_id=wav_path.stem,
                    wav_path=wav_path.resolve(),
                    speed_kmh=_parse_speed_kmh(wav_path.stem, vehicle),
                )
            )
    if not clips:
        raise FileNotFoundError(f"No .wav clips found under {data_dir}")
    return clips


def _direction_stats(predictions: list[str]) -> dict[str, int | float | str]:
    n = len(predictions)
    n_l2r = sum(1 for p in predictions if p == "L2R")
    n_r2l = sum(1 for p in predictions if p == "R2L")
    n_nv = sum(1 for p in predictions if p == "no_vehicle")
    majority = Counter(predictions).most_common(1)[0][0] if predictions else ""
    return {
        "n_clips": n,
        "n_L2R": n_l2r,
        "n_R2L": n_r2l,
        "n_no_vehicle": n_nv,
        "pct_L2R": (n_l2r / n) if n else 0.0,
        "pct_R2L": (n_r2l / n) if n else 0.0,
        "pct_no_vehicle": (n_nv / n) if n else 0.0,
        "majority_prediction": majority,
    }


def _summary_row(
    *,
    row_type: str,
    car_group: str,
    clip_name: str,
    stats: dict[str, int | float | str],
    model_name: str,
    checkpoint: Path,
) -> dict[str, str | float | int]:
    return {
        "row_type": row_type,
        "car_group": car_group,
        "clip_name": clip_name,
        "clip_path": "",
        "speed_kmh": "",
        "input_type": "",
        "duration_s": "",
        "n_mel_frames": "",
        "prediction": stats["majority_prediction"],
        "n_clips": stats["n_clips"],
        "n_L2R": stats["n_L2R"],
        "n_R2L": stats["n_R2L"],
        "n_no_vehicle": stats["n_no_vehicle"],
        "pct_L2R": round(float(stats["pct_L2R"]), 4),
        "pct_R2L": round(float(stats["pct_R2L"]), 4),
        "pct_no_vehicle": round(float(stats["pct_no_vehicle"]), 4),
        "majority_prediction": stats["majority_prediction"],
        "model": model_name,
        "checkpoint": str(checkpoint),
        "spectrogram_png": "",
        "prob_L2R": "",
        "prob_R2L": "",
        "prob_no_vehicle": "",
    }


def write_vs13_predictions_csv(
    output: Path,
    clip_rows: list[dict],
    *,
    checkpoint: Path,
    group_summaries: list[dict],
    overall_stats: dict[str, int | float | str],
    direction_only: bool = True,
    model_name: str | None = None,
    extra_summary: dict | None = None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_name = model_name or checkpoint.parent.name

    fieldnames = [
        "row_type",
        "car_group",
        "clip_name",
        "clip_path",
        "speed_kmh",
        "input_type",
        "duration_s",
        "n_mel_frames",
        "prediction",
        "n_clips",
        "n_L2R",
        "n_R2L",
        "n_no_vehicle",
        "pct_L2R",
        "pct_R2L",
        "pct_no_vehicle",
        "majority_prediction",
        "model",
        "checkpoint",
        "spectrogram_png",
        "prob_L2R",
        "prob_R2L",
        "prob_no_vehicle",
    ]

    ordered_rows: list[dict] = []
    group_by_name = {g["car_group"]: g for g in group_summaries}
    current_group: str | None = None
    for row in clip_rows:
        group = row["car_group"]
        if current_group is not None and group != current_group:
            ordered_rows.append(group_by_name[current_group])
        ordered_rows.append(row)
        current_group = group
    if current_group is not None and current_group in group_by_name:
        ordered_rows.append(group_by_name[current_group])

    overall_row = _summary_row(
        row_type=ROW_OVERALL,
        car_group="__ALL__",
        clip_name="__OVERALL__",
        stats=overall_stats,
        model_name=model_name,
        checkpoint=checkpoint,
    )
    ordered_rows.append(overall_row)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered_rows)

    summary_path = output.with_name("predictions_summary.json")
    summary_payload = {
                "layout": "vs13",
                "model": model_name,
                "checkpoint": str(checkpoint),
                "n_vehicles": len(group_summaries),
                "n_clips": overall_stats["n_clips"],
                "n_L2R": overall_stats["n_L2R"],
                "n_R2L": overall_stats["n_R2L"],
                "n_no_vehicle": overall_stats["n_no_vehicle"],
                "pct_L2R": overall_stats["pct_L2R"],
                "pct_R2L": overall_stats["pct_R2L"],
                "pct_no_vehicle": overall_stats["pct_no_vehicle"],
                "majority_prediction": overall_stats["majority_prediction"],
                "direction_only": direction_only,
                "per_vehicle": [
                    {
                        "car_group": g["car_group"],
                        "n_clips": g["n_clips"],
                        "n_L2R": g["n_L2R"],
                        "n_R2L": g["n_R2L"],
                        "n_no_vehicle": g["n_no_vehicle"],
                        "pct_L2R": g["pct_L2R"],
                        "pct_R2L": g["pct_R2L"],
                        "majority_prediction": g["majority_prediction"],
                    }
                    for g in group_summaries
                ],
            }
    if extra_summary:
        summary_payload.update(extra_summary)
    summary_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    return output


def _ensure_checkpoint_exists(checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def _load_fusion_weight(metrics_path: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    if metrics_path.is_file():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        fusion = data.get("fusion") or {}
        if "w_left" in fusion:
            return float(fusion["w_left"])
    return 0.05


def run_vs13(
    data_dir: Path,
    checkpoint: Path | None,
    output: Path,
    *,
    spectrogram_dir: Path | None = None,
    write_spectrograms: bool = True,
    device: str = "auto",
    direction_only: bool = False,
    flip_direction: bool = False,
    fusion: bool = False,
    left_checkpoint: Path | None = None,
    right_checkpoint: Path | None = None,
    fusion_weight: float | None = None,
    fusion_run_name: str = "fusion_2class_100ep",
    transfer: bool = False,
) -> Path:
    data_dir = data_dir.resolve()
    output = output.resolve()

    w_left: float | None = None
    model_label: str
    checkpoint_label: Path
    extra_summary: dict | None = None
    left_ckpt: Path | None = None
    right_ckpt: Path | None = None

    if fusion:
        left_ckpt = Path(left_checkpoint).resolve()
        right_ckpt = Path(right_checkpoint).resolve()
        _ensure_checkpoint_exists(left_ckpt)
        _ensure_checkpoint_exists(right_ckpt)
        w_left = _load_fusion_weight(DEFAULT_FUSION_METRICS, fusion_weight)
        model_label = fusion_run_name
        checkpoint_label = right_ckpt
        from idmt_experiments.transfer.eval import load_transfer_checkpoint

        _, cfg, _, _ = load_transfer_checkpoint(left_ckpt, "cpu")
        extra_summary = {
            "fusion": True,
            "w_left": w_left,
            "w_right": 1.0 - w_left,
            "left_checkpoint": str(left_ckpt),
            "right_checkpoint": str(right_ckpt),
        }
    elif checkpoint is not None:
        checkpoint = checkpoint.resolve()
        _ensure_checkpoint_exists(checkpoint) if transfer else _validate_checkpoint(checkpoint)
        model_label = checkpoint.parent.name
        checkpoint_label = checkpoint
        if transfer:
            from idmt_experiments.transfer.eval import load_transfer_checkpoint

            _, cfg, _, _ = load_transfer_checkpoint(checkpoint, "cpu")
        else:
            _, cfg, _, _ = load_checkpoint(checkpoint, resolve_device("cpu"))
    else:
        raise ValueError("Provide --checkpoint or use --fusion")

    label_cols = list(DIRECTION_LABELS if cfg.n_classes == 3 else DIRECTION_LABELS[:2])

    clips = discover_vs13_clips(data_dir)
    spec_dir = spectrogram_dir
    if write_spectrograms:
        spec_dir = (spec_dir or output.parent / "spectrograms").resolve()
        spec_dir.mkdir(parents=True, exist_ok=True)

    clip_rows: list[dict] = []
    by_vehicle: dict[str, list[str]] = {}

    for clip in tqdm(clips, desc="predict", unit="clip"):
        y, sr = load_mono(clip.wav_path, sr=SR_NATIVE, input_type="audio")
        if fusion:
            assert w_left is not None and left_ckpt and right_ckpt
            result = predict_wav_fusion(
                clip.wav_path,
                left_ckpt,
                right_ckpt,
                w_left=w_left,
                device=device,
            )
        elif transfer:
            result = predict_wav_transfer(clip.wav_path, checkpoint, device=device)
        else:
            result = predict_wav_mono(
                clip.wav_path, checkpoint, device=device, y=y, sr=sr
            )
        probs = result["probabilities"]
        prediction = resolve_prediction(
            probs,
            direction_only=direction_only,
            flip_direction=flip_direction,
        )

        spec_rel = ""
        if write_spectrograms and spec_dir is not None:
            spec_path = _spec_png_path(clip.wav_path, data_dir, spec_dir)
            save_constant_q_spectrogram(y, sr, spec_path)
            spec_rel = str(spec_path.relative_to(spec_dir))

        rel_path = clip.wav_path.relative_to(data_dir).as_posix()
        ckpt_str = str(checkpoint_label)
        if fusion and left_ckpt and right_ckpt:
            ckpt_str = f"L:{left_ckpt} R:{right_ckpt}"
        row: dict[str, str | float | int] = {
            "row_type": ROW_CLIP,
            "car_group": clip.vehicle,
            "clip_name": clip.clip_id,
            "clip_path": rel_path,
            "speed_kmh": clip.speed_kmh if clip.speed_kmh is not None else "",
            "input_type": "audio",
            "duration_s": round(float(result.get("duration_s", 0.0)), 3),
            "n_mel_frames": int(result.get("n_mel_frames", 0)),
            "prediction": prediction,
            "n_clips": "",
            "n_L2R": "",
            "n_R2L": "",
            "n_no_vehicle": "",
            "pct_L2R": "",
            "pct_R2L": "",
            "pct_no_vehicle": "",
            "majority_prediction": "",
            "model": model_label,
            "checkpoint": ckpt_str,
            "spectrogram_png": spec_rel,
        }
        for label in label_cols:
            row[f"prob_{label}"] = round(probs.get(label, 0.0), 4)
        clip_rows.append(row)
        by_vehicle.setdefault(clip.vehicle, []).append(prediction)

    group_summaries: list[dict] = []
    for vehicle in sorted(by_vehicle):
        stats = _direction_stats(by_vehicle[vehicle])
        group_summaries.append(
            _summary_row(
                row_type=ROW_GROUP,
                car_group=vehicle,
                clip_name=f"__GROUP__{vehicle}",
                stats=stats,
                model_name=model_label,
                checkpoint=checkpoint_label,
            )
        )

    all_predictions = [p for preds in by_vehicle.values() for p in preds]
    overall_stats = _direction_stats(all_predictions)

    return write_vs13_predictions_csv(
        output,
        clip_rows,
        checkpoint=checkpoint_label,
        group_summaries=group_summaries,
        overall_stats=overall_stats,
        direction_only=direction_only,
        model_name=model_label,
        extra_summary=extra_summary if fusion else None,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.fusion and args.transfer:
        raise SystemExit("Use either --fusion or --transfer, not both.")
    if args.output is None:
        args.output = DEFAULT_FUSION_VS13_OUTPUT if args.fusion else DEFAULT_OUTPUT
    if not args.fusion and args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINT
    out = run_vs13(
        args.data_dir,
        args.checkpoint,
        args.output,
        spectrogram_dir=args.spectrogram_dir,
        write_spectrograms=not args.no_spectrograms,
        device=args.device,
        direction_only=args.direction_only,
        flip_direction=args.flip_direction,
        fusion=args.fusion,
        left_checkpoint=args.left_checkpoint,
        right_checkpoint=args.right_checkpoint,
        fusion_weight=args.fusion_weight,
        fusion_run_name=args.fusion_run_name,
        transfer=args.transfer,
    )
    summary = json.loads(out.with_name("predictions_summary.json").read_text(encoding="utf-8"))
    print(f"Wrote predictions -> {out}")
    dir_only = summary.get("direction_only", True)
    line = (
        f"Overall: {summary['n_clips']} clips, "
        f"L2R {summary['pct_L2R']:.1%} ({summary['n_L2R']}), "
        f"R2L {summary['pct_R2L']:.1%} ({summary['n_R2L']})"
    )
    if dir_only:
        line += " [direction-only; no_vehicle disabled for VS13]"
    else:
        line += (
            f", no_vehicle {summary['pct_no_vehicle']:.1%} ({summary['n_no_vehicle']})"
        )
    print(line)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
