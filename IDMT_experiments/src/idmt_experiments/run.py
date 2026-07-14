#!/usr/bin/env python3
"""
IDMT day-1 utilities: index, split audit, classical baseline, physics plots.

Examples
--------
python -m idmt_experiments.run index
python -m idmt_experiments.run audit
python -m idmt_experiments.run classical --feature-type cc
python -m idmt_experiments.run plot-physics --out outputs/shared/figures/cc_direction_proof.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from idmt_experiments.config import DEFAULT_DATA_DIR, DEFAULT_SHARED_OUTPUT_DIR
from idmt_experiments.src.baselines import run_classical_baseline
from idmt_experiments.src.features import compute_cc_stack, compute_log_mel, load_stereo
from idmt_experiments.src.preprocess import discover_all_clips, filter_for_task, save_manifest
from idmt_experiments.src.splits import build_eusipco_split, default_split_meta_path, persist_split_meta
from idmt_experiments.src.weather_audit import audit_weather_dataset


def cmd_index(args) -> None:
    records = discover_all_clips(args.data_dir, mic_filter=args.mic, channel_filter=args.channel)
    out = save_manifest(records, args.output_dir / "manifest.csv")
    df_stats = {
        "n_clips": len(records),
        "n_events": len({r.event_id for r in records}),
        "n_background": sum(r.is_background for r in records),
        "locations": sorted({r.location for r in records}),
        "travel_directions": sorted({r.travel_direction for r in records if not r.is_background}),
    }
    (args.output_dir / "manifest_stats.json").write_text(json.dumps(df_stats, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({df_stats['n_clips']} clips, {df_stats['n_events']} unique events)")


def cmd_audit(args) -> None:
    train, val, test, meta = build_eusipco_split(
        args.data_dir,
        mic_filter=args.mic,
        channel_filter=args.channel,
    )
    persist_split_meta(meta, default_split_meta_path("eusipco", args.output_dir))
    audit = meta["audit"]
    print("EUSIPCO split leakage audit")
    print(f"  ok                 : {audit['ok']}")
    print(f"  train events       : {audit['n_train_events']}")
    print(f"  valid events       : {audit['n_valid_events']}")
    print(f"  test events        : {audit['n_test_events']}")
    print(f"  train clips        : {len(train)}")
    print(f"  valid clips        : {len(val)}")
    print(f"  test clips         : {len(test)}")
    if not audit["ok"]:
        print("  FAIL — overlaps:", audit)
        raise SystemExit(1)
    print("  PASS — no shared events across train / valid / test")


def cmd_audit_weather(args) -> None:
    report = audit_weather_dataset(
        args.data_dir,
        mic_filter=args.mic,
        channel_filter=args.channel,
    )
    out_path = args.output_dir / "splits" / "weather_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Weather dataset audit")
    print(f"  vehicle clips     : {report['n_vehicle_clips']}")
    print(f"  location confound : {report['location_confounded']}")
    print(f"  wet recorded at   : {report['sites_with_wet_recordings']}")
    print(f"  recommended split : {report['recommended_split']} @ {report['recommended_site']}")
    print("")
    print("  Location x weather (full dataset):")
    for loc, counts in report["location_weather_counts"].items():
        d, w = counts.get("D", 0), counts.get("W", 0)
        print(f"    {loc}: D={d} W={w}")
    print("")
    pooled = report["pooled_split"]["baselines"]
    site = report["site_split"]["baselines"]
    print("  Pooled test baselines (weather_stratified — inflated):")
    for k, v in pooled.items():
        print(f"    {k}: {v:.4f}")
    print("")
    print("  Site-only test baselines (weather_site — honest):")
    for k, v in site.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Wrote {out_path}")


def cmd_classical(args) -> None:
    print(
        "Starting classical baseline (loads ~5k train + ~2.7k test clips — expect 10–20 min on CPU)...",
        flush=True,
    )
    result = run_classical_baseline(
        args.data_dir,
        task=args.task,
        feature_type=args.feature_type,
        n_classes=args.n_classes,
        split_name=args.mode,
        output_dir=args.output_dir,
    )
    m = result["metrics"]
    print(f"Classical LR ({result['task']}, {args.feature_type}, {result['n_classes']}-class, {result['split_name']})")
    print(f"  test accuracy : {m['accuracy']:.4f}")
    print(f"  macro F1      : {m['macro_f1']:.4f}")
    if result.get("channel_swap"):
        cs = result["channel_swap"]
        if cs.get("flip_consistency") is not None:
            print(f"  swap consistency: {cs['flip_consistency']:.4f}")


def cmd_plot_physics(args) -> None:
    records = discover_all_clips(args.data_dir, mic_filter="SE", channel_filter="CH34")
    vehicles = [r for r in records if not r.is_background and r.vehicle == "C"]
    l2r = next(r for r in vehicles if r.travel_direction == "L2R")
    r2l = next(r for r in vehicles if r.travel_direction == "R2L")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, rec, title in zip(
        axes.flat,
        [l2r, r2l, l2r, r2l],
        ["L2R mel", "R2L mel", "L2R cross-correlation", "R2L cross-correlation"],
    ):
        y, sr = load_stereo(rec.wav_path)
        if "mel" in title.lower():
            spec = compute_log_mel(y, sr)
            ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        else:
            cc = compute_cc_stack(y, sr)
            ax.imshow(cc.T, aspect="auto", origin="lower", cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("time block")
    fig.suptitle("IDMT direction physics check — mel + stereo CC")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IDMT experiments utilities")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_SHARED_OUTPUT_DIR)
    p.add_argument("--mic", default="SE")
    p.add_argument("--channel", default="CH34")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="Build manifest CSV + stats")
    sub.add_parser("audit", help="Verify EUSIPCO split has no event leakage")
    sub.add_parser("audit-weather", help="Weather confound audit + naive baselines")

    c = sub.add_parser("classical", help="Logistic regression baseline (fast, no GPU)")
    c.add_argument("--task", choices=["direction", "vehicle", "weather"], default="direction")
    c.add_argument("--feature-type", choices=["mel", "cc", "stereo_mel"], default="cc")
    c.add_argument("--n-classes", type=int, default=None)
    c.add_argument(
        "--mode",
        choices=["eusipco", "weather_holdout", "weather_site", "weather_pooled"],
        default="eusipco",
        help="Split for classical baseline",
    )

    pl = sub.add_parser("plot-physics", help="Save mel + CC figure for L2R vs R2L")
    pl.add_argument("--out", type=Path, default=DEFAULT_SHARED_OUTPUT_DIR / "figures" / "cc_direction_proof.png")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "audit":
        cmd_audit(args)
    elif args.command == "audit-weather":
        cmd_audit_weather(args)
    elif args.command == "classical":
        cmd_classical(args)
    elif args.command == "plot-physics":
        cmd_plot_physics(args)


if __name__ == "__main__":
    main()
