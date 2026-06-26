#!/usr/bin/env python3
"""Direction inference on VS13 (length_estimation) grouped by vehicle type."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, DIRECTION_LABELS, SR_NATIVE
from idmt_experiments.src.direction.inference import predict_wav_mono
from idmt_experiments.src.direction.train import load_checkpoint, resolve_device
from idmt_experiments.src.features import load_mono
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
    Path(__file__).resolve().parents[3]
    / "length_estimation"
    / "outputs"
    / "vs13_direction"
    / "predictions.csv"
)
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "direction" / "mel_3class" / "best.pt"
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
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Mel direction checkpoint (default: {DEFAULT_CHECKPOINT})",
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
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_name = checkpoint.parent.name

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
    summary_path.write_text(
        json.dumps(
            {
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def run_vs13(
    data_dir: Path,
    checkpoint: Path,
    output: Path,
    *,
    spectrogram_dir: Path | None = None,
    write_spectrograms: bool = True,
    device: str = "auto",
    direction_only: bool = False,
    flip_direction: bool = False,
) -> Path:
    data_dir = data_dir.resolve()
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    _validate_checkpoint(checkpoint)

    _, cfg, _, _ = load_checkpoint(checkpoint, resolve_device("cpu"))
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
            "model": checkpoint.parent.name,
            "checkpoint": str(checkpoint),
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
                model_name=checkpoint.parent.name,
                checkpoint=checkpoint,
            )
        )

    all_predictions = [p for preds in by_vehicle.values() for p in preds]
    overall_stats = _direction_stats(all_predictions)

    return write_vs13_predictions_csv(
        output,
        clip_rows,
        checkpoint=checkpoint,
        group_summaries=group_summaries,
        overall_stats=overall_stats,
        direction_only=direction_only,
    )


def main() -> None:
    args = build_parser().parse_args()
    out = run_vs13(
        args.data_dir,
        args.checkpoint,
        args.output,
        spectrogram_dir=args.spectrogram_dir,
        write_spectrograms=not args.no_spectrograms,
        device=args.device,
        direction_only=args.direction_only,
        flip_direction=args.flip_direction,
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
