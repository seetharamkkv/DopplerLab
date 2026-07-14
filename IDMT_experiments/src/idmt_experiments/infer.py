#!/usr/bin/env python3
"""Inference on a single IDMT-style wav clip.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Thin CLI over ``cnn.inference``; must preserve identical predictions for baseline weights.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idmt_experiments.cnn.inference import predict_wav


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict pass-by direction from wav")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--swap-channels", action="store_true", help="Swap L/R before inference")
    p.add_argument("--json", action="store_true", help="Print JSON only")
    return p


def main() -> None:
    args = build_parser().parse_args()
    result = predict_wav(
        args.wav,
        args.checkpoint,
        device=args.device,
        swap_channels=args.swap_channels,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"File      : {result['wav_path']}")
        print(f"Predicted : {result['pred_label']}")
        for k, v in result["probabilities"].items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
