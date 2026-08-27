#!/usr/bin/env python3
"""One-command demo: audio → rotatable trajectory orbit product (Phase 5).

Audio / STFT only — no metadata required.

Examples:
  python scripts/demo_orbit_product.py \\
      --wav data/tier1_smoke/audio_clips/sample_0000002/*.wav \\
      --out outputs/orbit_product

  python scripts/demo_orbit_product.py \\
      --phase1 data/tier1_smoke/audio_clips/sample_0000002 \\
      --method flexible --out outputs/orbit_product
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traj_reconstruction import load_phase1_sample
from traj_reconstruction.product import export_orbit_product, predict_orbit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1", type=Path, default=None)
    p.add_argument("--wav", type=Path, default=None)
    p.add_argument("--method", choices=("flexible", "parametric", "mlp"), default="flexible")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/orbit_product"))
    p.add_argument("--scale-ambiguous", action="store_true")
    args = p.parse_args()

    wav = args.wav
    stft = None
    if args.phase1 is not None:
        sample = load_phase1_sample(args.phase1)
        stft = sample.stft_db
        wav = sample.wav_path
        # Demo still does not feed GT into predict_orbit.

    if wav is None and stft is None:
        raise SystemExit("provide --phase1 or --wav")

    print("Building orbit product from audio only (SIMULATED demo OK)…")
    if wav is not None:
        product = predict_orbit(
            wav_path=wav,
            method=args.method,
            mlp_checkpoint=args.checkpoint,
            scale_ambiguous=args.scale_ambiguous,
        )
    else:
        product = predict_orbit(
            stft_db=stft,
            method=args.method,
            mlp_checkpoint=args.checkpoint,
            scale_ambiguous=args.scale_ambiguous,
        )

    paths = export_orbit_product(product, args.out)
    summary = {
        "method": product.method,
        "confidence": product.confidence,
        "mirror_ambiguous": product.mirror_ambiguous,
        "scale_ambiguous": product.scale_ambiguous,
        "heading_absolute": product.heading_absolute,
        "n_frames": int(len(product.frame_times)),
        "disclaimer": product.disclaimer,
        "artifacts": {k: str(v) for k, v in paths.items()},
    }
    print(json.dumps(summary, indent=2))
    print(f"Open interactive viewer: {paths['html']}")


if __name__ == "__main__":
    main()
