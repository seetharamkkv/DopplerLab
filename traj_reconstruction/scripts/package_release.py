#!/usr/bin/env python3
"""Assemble a release folder for the simulated orbit product (Phase 6)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from traj_reconstruction import __version__
from traj_reconstruction.contract import DATA_SCOPE, INFERENCE_ALLOWED_RELPATHS, INFERENCE_FORBIDDEN_RELPATHS
from traj_reconstruction.leakage import run_leakage_audit


RELEASE_README = """# traj_reconstruction release (simulated)

Audio-only recovery of an **observer-centered trajectory orbit**
(full rotational family about the microphone). Absolute world heading is not identified.

## Input
- Monaural WAV, and/or
- `spectrograms/stft.npy`

## Output
- `orbit_product.html` — rotate / mirror the recovered path about the observer
- `orbit_product.json` — polyline + polar `(r, theta_rel)` + ambiguity flags
- `orbit_product.png` — static figure

## Ambiguity flags
- `mirror_ambiguous=true` (monaural default)
- `heading_absolute=false`
- `scale_ambiguous` when amplitude calibration is unknown

## Non-claims
- No absolute map heading
- No metadata (vehicle/site/CPA/speed labels) at inference
- Real roadside transfer is **deferred** unless a GPS appendix is attached

## Demo
```bash
python -m pip install -e .
python scripts/demo_orbit_product.py --wav /path/to/clip.wav --out release_demo
```

## Leakage
See `NO_LEAKAGE.md` and `leakage_audit.json` in this folder.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/release"))
    p.add_argument("--batch", type=Path, default=Path("data/tier1_smoke"))
    p.add_argument("--product-dir", type=Path, default=Path("outputs/orbit_product"))
    args = p.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    (out / "NO_LEAKAGE.md").write_text(
        (Path(__file__).resolve().parents[1] / "docs" / "NO_LEAKAGE.md").read_text()
    )
    (out / "README.md").write_text(RELEASE_README)
    report = run_leakage_audit(args.batch if args.batch.exists() else None)
    (out / "leakage_audit.json").write_text(json.dumps(report.to_dict(), indent=2))

    manifest = {
        "package": "traj-reconstruction",
        "version": __version__,
        "data_scope": DATA_SCOPE,
        "inference_allowed": list(INFERENCE_ALLOWED_RELPATHS),
        "inference_forbidden_count": len(INFERENCE_FORBIDDEN_RELPATHS),
        "schema": "DopplerSim Phase 1 / traj_reconstruction tier1 freehand",
        "real_transfer": "deferred",
        "leakage_ok": report.ok,
        "product_ux": "Phase 5 HTML orbit viewer",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.product_dir.exists():
        dest = out / "orbit_product"
        shutil.copytree(args.product_dir, dest, dirs_exist_ok=True)

    # Lightweight pointer to source modules (not a full wheel copy).
    (out / "ENTRYPOINTS.txt").write_text(
        "\n".join(
            [
                "scripts/demo_orbit_product.py",
                "scripts/audit_leakage.py",
                "scripts/infer_orbit.py",
                "scripts/run_tiered_validation.py",
            ]
        )
        + "\n"
    )
    print(json.dumps({"out": str(out), "leakage_ok": report.ok, "manifest": manifest}, indent=2))
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
