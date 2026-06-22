#!/usr/bin/env python3
"""
Inference: predict vehicle length (m) from VS13 pass-by wav clips.

See length_estimation/README.md for checkpoint choice (LOVO fold vs split best.pt).

Examples
--------
# Single clip (uses adjacent .txt for speed + CPA time)
python -m length_estimation.infer --checkpoint checkpoints/length_cnn/my_run/best.pt --wav data/vs13/Mazda3/Mazda3_50.wav

# All clips — prints overall MAE + vehicle-ID classification accuracy
python -m length_estimation.infer --checkpoint checkpoints/length_cnn/my_run/best.pt --all

# Valid split only
python -m length_estimation.infer --checkpoint checkpoints/length_cnn/my_run/best.pt --all --split valid

# Re-run split eval (prefer length_estimation.eval for LOVO + split)
python -m length_estimation.eval --mode split --run-name my_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from length_estimation.src.phase_b.eval import run_eval
from length_estimation.src.phase_b.inference import predict_from_sidecar, run_inference_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VS13 length CNN inference / eval-only")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--wav", type=Path, default=None, help="Single wav file (needs .txt sidecar)")
    p.add_argument("--all", action="store_true", help="Predict clips + overall MAE & vehicle-ID accuracy")
    p.add_argument("--split", choices=["train", "valid", "all"], default="all")
    p.add_argument("--eval-only", action="store_true", help="Run eval script only (valid split)")
    p.add_argument("--eval-split", choices=["valid", "train"], default="valid")
    p.add_argument("--output", type=Path, default=None, help="CSV path for --all")
    p.add_argument("--device", default="auto")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ckpt = args.checkpoint
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    if args.eval_only:
        run_eval(checkpoint=ckpt, data_dir=args.data_dir, device=args.device, split=args.eval_split)
        return

    if args.wav:
        pred = predict_from_sidecar(args.wav, ckpt, device=args.device)
        print(f"clip      : {pred.clip_id}")
        print(f"speed     : {pred.speed_kmh:.1f} km/h")
        print(f"length    : {pred.length_m:.3f} m")
        return

    if args.all:
        split = None if args.split == "all" else args.split
        _df, metrics = run_inference_report(
            ckpt,
            data_dir=args.data_dir,
            device=args.device,
            output_path=args.output,
            split=split,
        )
        print(
            f"Overall MAE={metrics['mae_m']:.4f} m  |  "
            f"Vehicle-ID accuracy={metrics['vehicle_id_accuracy']:.1%} "
            f"({metrics['vehicle_id_correct']}/{metrics['vehicle_id_total']} clips)"
        )
        return

    print("Provide --wav, --all, or --eval-only")
    sys.exit(1)


if __name__ == "__main__":
    main()
