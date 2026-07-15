#!/usr/bin/env python3
"""
Phase A CLI for VS13 vehicle length estimation (features + physics baselines).

For Phase B (CNN), use the dedicated entry points instead:
  python -m length_estimation.train
  python -m length_estimation.eval
  python -m length_estimation.infer

See length_estimation/README.md for the full walkthrough.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from length_estimation.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_OUTPUT_DIR,
    PhaseBConfig,
)
from length_estimation.src.evaluate import run_phase_a
from length_estimation.src.features_physics import extract_clip_features
from length_estimation.src.report import run_report
from length_estimation.src.preprocess import (
    load_clips,
    resolve_data_dir,
    save_manifest,
)
from length_estimation.jasa_week import run_ablation1, run_full_pipeline


def _get_records(args: argparse.Namespace):
    data_dir = resolve_data_dir(getattr(args, "data_dir", None))
    write_manifest = getattr(args, "write_manifest", False)
    records = load_clips(data_dir, write_manifest=write_manifest)
    if not records:
        print(f"No clips found under {data_dir}")
        print("Expected: {VehicleName}/{VehicleName}_{speed}.wav + .txt  (annotation: 'speed_kmh cpa_time_s')")
        sys.exit(1)
    return records, data_dir


def cmd_index(args: argparse.Namespace) -> None:
    """Optional — writes manifest.csv for inspection. Other commands scan disk directly."""
    records, data_dir = _get_records(args)
    out = Path(args.output_dir) / "manifest.csv"
    df = save_manifest(records, out)
    print(f"Data root: {data_dir}")
    print(f"Indexed {len(df)} clips from {df['vehicle'].nunique()} vehicles -> {out}")


def cmd_features(args: argparse.Namespace) -> None:
    records, data_dir = _get_records(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.csv"

    print(f"Data root: {data_dir} ({len(records)} clips)")

    rows = []
    for rec in tqdm(records, desc="Extracting features"):
        try:
            rows.append(
                extract_clip_features(
                    rec,
                    include_reassigned=not args.no_reassigned,
                    include_ssq=not args.no_ssq,
                )
            )
        except Exception as exc:
            print(f"WARN {rec.clip_id}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(features_path, index=False)
    print(f"Saved {len(df)} feature rows -> {features_path}")


def cmd_phase_a(args: argparse.Namespace) -> None:
    features_path = Path(args.features) if args.features else Path(args.output_dir) / "features.csv"
    if not features_path.exists():
        print(f"Features not found: {features_path}")
        print("Run: python -m length_estimation.run features")
        sys.exit(1)

    df = pd.read_csv(features_path)
    out = Path(args.output_dir) / "phase_a"
    targets = ["length_m", "wheelbase_m"] if args.target == "both" else [args.target]

    for target in targets:
        print(f"Phase A — target={target}")
        run_phase_a(df, out, target=target)
        print(f"  -> {out}/phase_a_summary_{target}.json")

    print("Generating clip report + result_summary.txt ...")
    run_report(df, out, data_dir=getattr(args, "data_dir", None))
    print(f"  -> {out}/result_summary.txt")
    print(f"  -> {out}/clip_predictions.csv")


def cmd_report(args: argparse.Namespace) -> None:
    features_path = Path(args.features) if args.features else Path(args.output_dir) / "features.csv"
    if not features_path.exists():
        print(f"Features not found: {features_path}")
        sys.exit(1)
    df = pd.read_csv(features_path)
    out = Path(args.output_dir) / "phase_a"
    print("Building report (LOVO length + vehicle classifier)...")
    run_report(df, out, data_dir=getattr(args, "data_dir", None))
    print(f"  -> {out}/result_summary.txt")
    print(f"  -> {out}/clip_predictions.csv")
    print(f"  -> {out}/vehicle_summary.csv")


def cmd_jasa_ablation1(args: argparse.Namespace) -> None:
    """JASA week Ablation 1: env−10 dB×v → length_m vs wheelbase_m (LOVO)."""
    features = Path(args.features) if args.features else Path(args.output_dir) / "features.csv"
    out = Path(args.jasa_output_dir) if args.jasa_output_dir else Path(args.output_dir) / "jasa_week"
    run_ablation1(features_path=features, output_dir=out)


def cmd_jasa_week(args: argparse.Namespace) -> None:
    """JASA week full pipeline: Ablation 1 → 2 → final Wb→L rule."""
    features = Path(args.features) if args.features else Path(args.output_dir) / "features.csv"
    out = Path(args.jasa_output_dir) if args.jasa_output_dir else Path(args.output_dir) / "jasa_week"
    run_full_pipeline(features_path=features, output_dir=out)


def cmd_phase_b(args: argparse.Namespace) -> None:
    """Deprecated wrapper — prefer: python -m length_estimation.train"""
    from length_estimation.config import PhaseBConfig
    from length_estimation.src.phase_b.eval import run_eval
    from length_estimation.src.phase_b.train import train_lovo, train_split

    cfg = PhaseBConfig(
        spec_type=args.spec_type,
        target="length_m",
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        include_speed=args.include_speed,
    )
    print("Note: prefer `python -m length_estimation.train` (auto-runs eval)")
    if getattr(args, "mode", "split") == "lovo":
        run_dir = train_lovo(
            data_dir=getattr(args, "data_dir", None),
            run_name=args.run_name,
            cfg=cfg,
            device=args.device if args.device != "auto" else "auto",
        )
        run_eval(run_dir=run_dir, data_dir=getattr(args, "data_dir", None), device=args.device)
    else:
        ckpt = train_split(
            data_dir=getattr(args, "data_dir", None),
            run_name=args.run_name,
            cfg=cfg,
            device=args.device if args.device != "auto" else "auto",
        )
        run_eval(checkpoint=ckpt, data_dir=getattr(args, "data_dir", None), device=args.device)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VS13 vehicle length estimation",
        epilog=(
            "Phase A only. For CNN train/eval/infer see length_estimation/README.md.\n"
            "Note: 'index' is optional. 'features' scans data/vs13 automatically."
        ),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="VS13 root (default: length_estimation/data/vs13)",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)

    sub = p.add_subparsers(dest="command", required=True)

    idx = sub.add_parser(
        "index",
        help="[Optional] Write manifest.csv — other commands do not need this",
    )
    idx.set_defaults(func=cmd_index, write_manifest=False)

    feat = sub.add_parser("features", help="Phase A: extract hand-crafted features")
    feat.add_argument("--no-reassigned", action="store_true")
    feat.add_argument("--no-ssq", action="store_true")
    feat.add_argument(
        "--write-manifest",
        action="store_true",
        help="Also refresh outputs/manifest.csv while extracting features",
    )
    feat.set_defaults(func=cmd_features)

    pa = sub.add_parser(
        "phase-a",
        help="Phase A: correlation, physics baselines, LOVO RidgeCV, best-model report",
    )
    pa.add_argument("--features", type=Path, default=None)
    pa.add_argument("--target", choices=["length_m", "wheelbase_m", "both"], default="both")
    pa.set_defaults(func=cmd_phase_a)

    rpt = sub.add_parser("report", help="Per-clip table + result_summary.txt (needs features.csv)")
    rpt.add_argument("--features", type=Path, default=None)
    rpt.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    rpt.set_defaults(func=cmd_report)

    j1 = sub.add_parser(
        "jasa-ablation1",
        help="JASA week Ablation 1: LOVO length_m vs wheelbase_m (env−10 dB×v proxy)",
    )
    j1.add_argument("--features", type=Path, default=None)
    j1.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    j1.add_argument(
        "--jasa-output-dir",
        type=Path,
        default=None,
        help="Override artifact dir (default: <output-dir>/jasa_week/)",
    )
    j1.set_defaults(func=cmd_jasa_ablation1)

    jw = sub.add_parser(
        "jasa-week",
        help="JASA week full pipeline: Ablation 1 → 2 → final Wb→L LOVO rule",
    )
    jw.add_argument("--features", type=Path, default=None)
    jw.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    jw.add_argument("--jasa-output-dir", type=Path, default=None)
    jw.set_defaults(func=cmd_jasa_week)

    pb = sub.add_parser(
        "phase-b",
        help="[Deprecated] Use: python -m length_estimation.train",
    )
    pb.add_argument("--mode", choices=["split", "lovo"], default="split")
    pb.add_argument("--target", choices=["length_m"], default="length_m")
    pb.add_argument("--spec-type", choices=["mel", "ssq"], default="mel")
    pb.add_argument("--batch-size", type=int, default=16)
    pb.add_argument("--epochs", type=int, default=40)
    pb.add_argument("--lr", type=float, default=1e-3)
    pb.add_argument("--include-speed", action="store_true")
    pb.add_argument("--device", default="auto")
    pb.add_argument("--run-name", type=str, default=None)
    pb.set_defaults(func=cmd_phase_b)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
