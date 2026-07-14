#!/usr/bin/env python3
"""Batch direction predictions on local monoaural test clips (mel model only).

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Default checkpoint is ``mel_3class``. Inference must match ``cnn.inference`` for baseline
weights; do not change prediction path without re-benchmarking.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, DIRECTION_LABELS, PACKAGE_ROOT, SR_NATIVE, resolve_checkpoint_file, DirectionConfig
from idmt_experiments.cnn.inference import predict_wav_mono
from idmt_experiments.cnn.train import load_checkpoint, resolve_device
from idmt_experiments.src.features import load_mono

DEFAULT_DATA_DIR = PACKAGE_ROOT / "test" / "data"
DEFAULT_OUTPUT = PACKAGE_ROOT / "test" / "outputs" / "predictions.csv"
DEFAULT_SPEC_DIR = PACKAGE_ROOT / "test" / "outputs"
DEFAULT_CHECKPOINT = resolve_checkpoint_file(DirectionConfig(task="direction"), "mel_3class")
DEFAULT_METADATA_NAME = "metadata.txt"

# metadata.txt: filename;vehicle_count;left|right
# left  = vehicle travels toward the left  -> R2L
# right = vehicle travels toward the right -> L2R
METADATA_TO_LABEL = {"left": "R2L", "right": "L2R"}

INPUT_TYPE_AUDIO = "audio"
INPUT_TYPE_VIDEO = "video"
DEFAULT_EXT_BY_INPUT = {INPUT_TYPE_AUDIO: "wav", INPUT_TYPE_VIDEO: "avi"}
VIDEO_EXT_CHOICES = ("avi", "mp4", "mkv", "mov", "webm", "wmv", "m4v")

# Constant-Q spectrogram defaults (visualization only; full clip, no trim)
CQT_HOP_LENGTH = 512
CQT_FMIN_HZ = librosa.note_to_hz("C1")
CQT_N_BINS = 84


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict L2R / R2L on monoaural clips in test/data (mel checkpoint only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m idmt_experiments.test_clips\n"
            "  python -m idmt_experiments.test_clips --input-type video\n"
            "  python -m idmt_experiments.test_clips --input-type video --ext mp4\n"
        ),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Folder of clips (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--input-type",
        choices=[INPUT_TYPE_AUDIO, INPUT_TYPE_VIDEO],
        default=INPUT_TYPE_AUDIO,
        help=(
            "audio: direct audio files (default extension .wav). "
            "video: extract audio from video files via ffmpeg (default extension .avi; "
            "also mp4, mkv, mov, ... with --ext)."
        ),
    )
    p.add_argument(
        "--ext",
        default=None,
        help=(
            "File extension without dot. Defaults to wav for --input-type audio "
            "and avi for --input-type video."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Mel direction checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--spectrogram-dir",
        type=Path,
        default=DEFAULT_SPEC_DIR,
        help=f"Folder for constant-Q PNGs, same stem as clip (default: {DEFAULT_SPEC_DIR})",
    )
    p.add_argument("--no-spectrograms", action="store_true", help="Skip constant-Q PNG export")
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=f"Optional labels file in data-dir (default: <data-dir>/{DEFAULT_METADATA_NAME})",
    )
    p.add_argument(
        "--score-only",
        action="store_true",
        help="Merge metadata into existing --output CSV and compute accuracy (no inference)",
    )
    p.add_argument(
        "--direction-only",
        action="store_true",
        help=(
            "Pick L2R vs R2L from the two direction logits only (ignore no_vehicle). "
            "Often helps on pass-by clips where the model abstains too often."
        ),
    )
    p.add_argument(
        "--flip-direction",
        action="store_true",
        help="Swap L2R <-> R2L after prediction (use if model direction sign is inverted).",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--recursive", action="store_true", help="Search subfolders under data-dir")
    return p


def _normalize_ext(ext: str) -> str:
    ext = ext.strip().lower().lstrip(".")
    if not ext:
        raise ValueError("extension must be non-empty (e.g. wav or avi)")
    return ext


def resolve_input_type_and_ext(input_type: str, ext: str | None) -> tuple[str, str]:
    """Resolve CLI input kind and file extension."""
    if input_type not in (INPUT_TYPE_AUDIO, INPUT_TYPE_VIDEO):
        raise ValueError(f"input_type must be audio or video, got {input_type!r}")

    if ext is None:
        return input_type, DEFAULT_EXT_BY_INPUT[input_type]

    ext_norm = _normalize_ext(ext)
    # Back-compat: --ext avi without --input-type video
    if input_type == INPUT_TYPE_AUDIO and ext_norm in VIDEO_EXT_CHOICES:
        input_type = INPUT_TYPE_VIDEO
    return input_type, ext_norm


def _collect_clips(data_dir: Path, *, ext: str, recursive: bool) -> list[Path]:
    pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
    return sorted(data_dir.glob(pattern))


def _spec_png_path(clip: Path, data_dir: Path, spec_dir: Path) -> Path:
    rel = clip.relative_to(data_dir)
    # flat name: subdirs joined with _ so foo/bar.avi -> foo_bar.png
    png_name = rel.with_suffix(".png").as_posix().replace("/", "_")
    return spec_dir / png_name


def save_constant_q_spectrogram(
    y: np.ndarray,
    sr: int,
    out_path: Path,
    *,
    hop_length: int = CQT_HOP_LENGTH,
    fmin: float = CQT_FMIN_HZ,
    n_bins: int = CQT_N_BINS,
) -> Path:
    """Save log-magnitude constant-Q spectrogram for the full waveform (no trim)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        cqt_db,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        x_axis="time",
        y_axis="cqt_hz",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(out_path.stem)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _validate_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Train or copy mel_3class/best.pt locally (weights are gitignored)."
        )
    _, cfg, _, _ = load_checkpoint(checkpoint, resolve_device("cpu"))
    if cfg.feature_type != "mel":
        raise ValueError(
            f"Expected mel (monoaural) checkpoint, got feature_type={cfg.feature_type!r}. "
            "Use checkpoints/cnn/direction/mel_3class/best.pt, not a stereo CC model."
        )


def _default_metadata_path(data_dir: Path) -> Path:
    return data_dir / DEFAULT_METADATA_NAME


def load_metadata(metadata_path: Path) -> dict[str, dict[str, str | int]]:
    """Parse metadata.txt: ``clip.avi;vehicle_count;left|right`` per line."""
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        return {}
    out: dict[str, dict[str, str | int]] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 3:
            continue
        fname, count_raw, direction = parts[0], parts[1], parts[2].lower()
        if direction not in METADATA_TO_LABEL:
            continue
        try:
            vehicle_count = int(count_raw)
        except ValueError:
            vehicle_count = -1
        out[fname] = {
            "label_raw": direction,
            "label_true": METADATA_TO_LABEL[direction],
            "vehicle_count": vehicle_count,
        }
    return out


def _flip_label(label: str) -> str:
    if label == "L2R":
        return "R2L"
    if label == "R2L":
        return "L2R"
    return label


def resolve_prediction(
    probs: dict[str, float],
    *,
    direction_only: bool = False,
    flip_direction: bool = False,
) -> str:
    """Map class probabilities to a direction label."""
    if direction_only:
        p_l2r = float(probs.get("L2R", 0.0))
        p_r2l = float(probs.get("R2L", 0.0))
        pred = "L2R" if p_l2r >= p_r2l else "R2L"
    else:
        pred = max(
            ("L2R", float(probs.get("L2R", 0.0))),
            ("R2L", float(probs.get("R2L", 0.0))),
            ("no_vehicle", float(probs.get("no_vehicle", 0.0))),
            key=lambda item: item[1],
        )[0]
    if flip_direction:
        pred = _flip_label(pred)
    return pred


def _row_probs(row: dict) -> dict[str, float]:
    return {
        label: float(row.get(f"prob_{label}", 0.0))
        for label in DIRECTION_LABELS
        if f"prob_{label}" in row
    }


def apply_inference_options(
    rows: list[dict],
    *,
    direction_only: bool = False,
    flip_direction: bool = False,
) -> None:
    """Recompute prediction column from stored probabilities."""
    if not direction_only and not flip_direction:
        return
    for row in rows:
        if row.get("clip_name") == "__OVERALL__":
            continue
        probs = _row_probs(row)
        if probs:
            row["prediction"] = resolve_prediction(
                probs,
                direction_only=direction_only,
                flip_direction=flip_direction,
            )


def _accuracy_subset(rows: list[dict], predicate) -> dict[str, int | float]:
    scored = correct = 0
    for row in rows:
        if row.get("clip_name") == "__OVERALL__":
            continue
        label_true = row.get("label_true", "")
        if not label_true or not predicate(row):
            continue
        scored += 1
        correct += int(row.get("prediction") == label_true)
    accuracy = (correct / scored) if scored else 0.0
    return {"n_scored": scored, "n_correct": correct, "accuracy": accuracy}


def apply_metadata_scores(rows: list[dict]) -> tuple[list[dict], dict]:
    """Add label_true / correct columns and compute summary metrics."""
    scored = 0
    correct = 0
    for row in rows:
        if row.get("clip_name") == "__OVERALL__":
            continue
        meta = row.get("_meta") or {}
        label_true = meta.get("label_true", row.get("label_true", ""))
        label_raw = meta.get("label_raw", row.get("label_raw", ""))
        vehicle_count = meta.get("vehicle_count", row.get("vehicle_count", ""))
        row["label_raw"] = label_raw
        row["label_true"] = label_true
        row["vehicle_count"] = vehicle_count
        if label_true:
            is_correct = row.get("prediction") == label_true
            row["correct"] = int(is_correct)
            scored += 1
            correct += int(is_correct)
        else:
            row["correct"] = ""
    accuracy = (correct / scored) if scored else 0.0
    summary = {
        "n_scored": scored,
        "n_correct": correct,
        "accuracy": accuracy,
        "accuracy_single_vehicle": _accuracy_subset(
            rows, lambda r: str(r.get("vehicle_count", "")) == "1"
        ),
        "accuracy_multi_vehicle": _accuracy_subset(
            rows, lambda r: str(r.get("vehicle_count", "")) not in ("", "1")
        ),
        "accuracy_direction_predicted": _accuracy_subset(
            rows, lambda r: r.get("prediction") in ("L2R", "R2L")
        ),
    }
    return rows, summary


def _prob_columns(rows: list[dict]) -> list[str]:
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k.startswith("prob_") and k not in cols:
                cols.append(k)
    return sorted(cols)


def write_predictions_csv(
    output: Path,
    rows: list[dict],
    *,
    checkpoint: Path,
    metadata: dict[str, dict[str, str]] | None = None,
    summary: dict | None = None,
    direction_only: bool = False,
    flip_direction: bool = False,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    clip_rows = [r for r in rows if r.get("clip_name") != "__OVERALL__"]
    for row in clip_rows:
        meta = (metadata or {}).get(str(row.get("clip_name", "")), {})
        if not meta:
            meta = row.pop("_meta", None) or {}
        else:
            row.pop("_meta", None)
        if meta:
            row["label_raw"] = meta.get("label_raw", "")
            row["label_true"] = meta.get("label_true", "")
            row["vehicle_count"] = meta.get("vehicle_count", "")

    clip_rows, computed = apply_metadata_scores(clip_rows)
    summary = summary or computed
    summary.setdefault("direction_only", direction_only)
    summary.setdefault("flip_direction", flip_direction)

    model_name = checkpoint.parent.name
    for row in clip_rows:
        row["model"] = model_name
        row["checkpoint"] = str(checkpoint)

    prob_cols = _prob_columns(clip_rows)
    fieldnames = [
        "clip_name",
        "clip_path",
        "input_type",
        "duration_s",
        "n_mel_frames",
        "vehicle_count",
        "label_raw",
        "label_true",
        "prediction",
        "correct",
        "model",
        "checkpoint",
        "spectrogram_png",
    ] + prob_cols

    overall_row = {
        "clip_name": "__OVERALL__",
        "clip_path": "",
        "input_type": "",
        "duration_s": "",
        "n_mel_frames": "",
        "label_raw": "",
        "label_true": f"{summary['n_correct']}/{summary['n_scored']}",
        "prediction": f"accuracy={summary['accuracy']:.4f}",
        "correct": summary["accuracy"],
        "model": model_name,
        "checkpoint": str(checkpoint),
        "spectrogram_png": "",
    }

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clip_rows)
        if summary.get("n_scored", 0) > 0:
            writer.writerow(overall_row)

    summary_path = output.with_name("predictions_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "feature_type": "mel",
                "n_classes": 3,
                "n_scored": summary["n_scored"],
                "n_correct": summary["n_correct"],
                "accuracy": summary["accuracy"],
                "accuracy_single_vehicle": summary.get("accuracy_single_vehicle"),
                "accuracy_multi_vehicle": summary.get("accuracy_multi_vehicle"),
                "accuracy_direction_predicted": summary.get("accuracy_direction_predicted"),
                "metadata_mapping": METADATA_TO_LABEL,
                "direction_only": summary.get("direction_only", False),
                "flip_direction": summary.get("flip_direction", False),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def score_existing_csv(
    input_csv: Path,
    output: Path,
    metadata_path: Path,
    checkpoint: Path,
    *,
    direction_only: bool = False,
    flip_direction: bool = False,
) -> Path:
    input_csv = Path(input_csv)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {input_csv}")
    metadata = load_metadata(metadata_path)
    if not metadata:
        raise FileNotFoundError(f"No metadata loaded from {metadata_path}")

    with input_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clip_rows = [r for r in rows if r.get("clip_name") != "__OVERALL__"]
    for row in clip_rows:
        row["_meta"] = metadata.get(row["clip_name"], {})
    apply_inference_options(
        clip_rows,
        direction_only=direction_only,
        flip_direction=flip_direction,
    )
    return write_predictions_csv(
        output,
        clip_rows,
        checkpoint=checkpoint,
        metadata=metadata,
        direction_only=direction_only,
        flip_direction=flip_direction,
    )


def run_batch(
    data_dir: Path,
    checkpoint: Path,
    output: Path,
    *,
    input_type: str = INPUT_TYPE_AUDIO,
    ext: str = "wav",
    spectrogram_dir: Path | None = DEFAULT_SPEC_DIR,
    write_spectrograms: bool = True,
    device: str = "auto",
    recursive: bool = False,
    metadata_path: Path | None = None,
    direction_only: bool = False,
    flip_direction: bool = False,
) -> Path:
    data_dir = data_dir.resolve()
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    input_type, ext = resolve_input_type_and_ext(input_type, ext)
    ext = _normalize_ext(ext)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    _validate_checkpoint(checkpoint)
    _, cfg, _, _ = load_checkpoint(checkpoint, resolve_device("cpu"))
    label_cols = list(DIRECTION_LABELS if cfg.n_classes == 3 else DIRECTION_LABELS[:2])

    clips = _collect_clips(data_dir, ext=ext, recursive=recursive)
    if not clips:
        raise FileNotFoundError(
            f"No .{ext} files in {data_dir} (input-type={input_type})"
        )

    meta_path = metadata_path or _default_metadata_path(data_dir)
    metadata = load_metadata(meta_path)

    spec_dir = spectrogram_dir.resolve() if spectrogram_dir is not None else None
    if write_spectrograms and spec_dir is not None:
        spec_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float]] = []

    for clip in tqdm(clips, desc="predict", unit="clip"):
        y, sr = load_mono(clip, sr=SR_NATIVE, input_type=input_type)
        result = predict_wav_mono(clip, checkpoint, device=device, y=y, sr=sr)
        probs = result["probabilities"]
        prediction = resolve_prediction(
            probs,
            direction_only=direction_only,
            flip_direction=flip_direction,
        )

        spec_rel = ""
        if write_spectrograms and spec_dir is not None:
            spec_path = _spec_png_path(clip, data_dir, spec_dir)
            save_constant_q_spectrogram(y, sr, spec_path)
            spec_rel = str(spec_path.relative_to(spec_dir))

        row: dict[str, str | float] = {
            "clip_name": clip.name,
            "clip_path": str(clip.relative_to(data_dir)),
            "input_type": input_type,
            "duration_s": round(float(result.get("duration_s", 0.0)), 3),
            "n_mel_frames": int(result.get("n_mel_frames", 0)),
            "prediction": prediction,
            "spectrogram_png": spec_rel,
            "_meta": metadata.get(clip.name, {}),
        }
        for label in label_cols:
            row[f"prob_{label}"] = round(probs.get(label, 0.0), 4)
        rows.append(row)

    return write_predictions_csv(
        output,
        rows,
        checkpoint=checkpoint,
        metadata=metadata,
        direction_only=direction_only,
        flip_direction=flip_direction,
    )


def main() -> None:
    args = build_parser().parse_args()
    input_type, ext = resolve_input_type_and_ext(args.input_type, args.ext)
    metadata_path = args.metadata or _default_metadata_path(args.data_dir.resolve())

    if args.score_only:
        in_csv = args.output
        if not in_csv.is_file():
            legacy = PACKAGE_ROOT / "test" / "predictions.csv"
            in_csv = legacy if legacy.is_file() else in_csv
        out = score_existing_csv(
            in_csv,
            args.output.resolve(),
            metadata_path.resolve(),
            args.checkpoint.resolve(),
            direction_only=args.direction_only,
            flip_direction=args.flip_direction,
        )
        summary = json.loads(out.with_name("predictions_summary.json").read_text(encoding="utf-8"))
        print(f"Scored predictions -> {out}")
        print(f"Accuracy: {summary['accuracy']:.2%} ({summary['n_correct']}/{summary['n_scored']})")
        print(json.dumps(summary, indent=2))
        return

    out = run_batch(
        args.data_dir,
        args.checkpoint,
        args.output,
        input_type=input_type,
        ext=ext,
        spectrogram_dir=args.spectrogram_dir,
        write_spectrograms=not args.no_spectrograms,
        device=args.device,
        recursive=args.recursive,
        metadata_path=metadata_path,
        direction_only=args.direction_only,
        flip_direction=args.flip_direction,
    )
    n = len(_collect_clips(args.data_dir.resolve(), ext=ext, recursive=args.recursive))
    print(f"Wrote {n} predictions -> {out}")
    summary_path = out.with_name("predictions_summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("n_scored", 0) > 0:
            print(
                f"Accuracy: {summary['accuracy']:.2%} "
                f"({summary['n_correct']}/{summary['n_scored']})"
            )
    if not args.no_spectrograms:
        print(f"Constant-Q PNGs -> {args.spectrogram_dir.resolve()}")
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "model": args.checkpoint.resolve().parent.name,
                "data_dir": str(args.data_dir.resolve()),
                "input_type": input_type,
                "ext": ext,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
