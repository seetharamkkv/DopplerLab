#!/usr/bin/env python3
"""Run Phase 6 leakage audit and optional WAV-only CI smoke.

Examples:
  python scripts/audit_leakage.py --batch data/tier1_smoke --out outputs/leakage
  python scripts/audit_leakage.py --batch data/tier1_smoke --out outputs/leakage --wav-smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traj_reconstruction.leakage import run_leakage_audit, wav_only_smoke


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/leakage"))
    p.add_argument("--wav-smoke", action="store_true")
    p.add_argument("--wav", type=Path, default=None, help="Explicit WAV file for smoke")
    p.add_argument(
        "--method",
        choices=("parametric", "flexible", "mlp"),
        default="parametric",
        help="Inference method for wav-only smoke (default parametric for speed)",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = run_leakage_audit(args.batch)
    (args.out / "leakage_audit.json").write_text(json.dumps(report.to_dict(), indent=2))

    md = ["# Leakage audit report", "", f"**ok:** `{report.ok}`", ""]
    for f in report.findings:
        md.append(f"- **{f.severity}** `{f.code}`: {f.message}")
    md.append("")
    (args.out / "leakage_audit.md").write_text("\n".join(md))
    print(json.dumps({"ok": report.ok, "n_findings": len(report.findings)}, indent=2))

    if args.wav_smoke:
        if args.wav is not None:
            wav = args.wav
        elif args.batch is not None:
            wavs = sorted(Path(args.batch).glob("audio_clips/sample_*/*.wav"))
            if not wavs:
                raise SystemExit("no wav found under batch")
            wav = wavs[0]
        else:
            raise SystemExit("--wav-smoke requires --wav or --batch")
        smoke = wav_only_smoke(wav, args.out, method=args.method)
        (args.out / "wav_only_smoke.json").write_text(json.dumps(smoke, indent=2))
        print(json.dumps(smoke, indent=2))

    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
