#!/usr/bin/env python3
"""Eval / audit entry points for physics direction models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, PhysicsConfig, resolve_physics_run_dir
from idmt_experiments.physics.audit import run_feature_audit
from idmt_experiments.physics.eval import run_eval
from idmt_experiments.src.splits import build_split


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Physics direction eval / audit")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("eval", help="Evaluate a trained physics run")
    e.add_argument("--run-name", required=True)
    e.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    e.add_argument("--data-dir", type=Path, default=None)
    e.add_argument("--output-dir", type=Path, default=None)
    e.add_argument("--split", choices=["test", "valid"], default="test")
    e.add_argument("--interventions", action="store_true")

    a = sub.add_parser("audit", help="Univariate feature separability on train split")
    a.add_argument("--data-dir", type=Path, default=None)
    a.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "physics" / "direction")
    a.add_argument("--mono-source", choices=["left", "right"], default="left")
    a.add_argument(
        "--feature-set",
        choices=["kinematic_v1", "kinematic_v2", "kinematic_v3"],
        default="kinematic_v2",
    )
    a.add_argument("--split", default="eusipco")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "eval":
        run_dir = Path(args.checkpoint_dir) / "physics" / "direction" / args.run_name
        if not (run_dir / "run_config.json").exists():
            run_dir = resolve_physics_run_dir(PhysicsConfig(), args.run_name, root=args.checkpoint_dir)
        run_eval(
            run_dir=run_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            split=args.split,
            run_intervention_tests=args.interventions,
        )
    elif args.command == "audit":
        cfg = PhysicsConfig(mono_source=args.mono_source, split_name=args.split, feature_set=args.feature_set)
        train_records, _, _, _ = build_split(
            cfg.split_name,
            args.data_dir,
            mic_filter=cfg.mic_filter,
            channel_filter=cfg.channel_filter,
            val_fraction=cfg.val_fraction,
            seed=cfg.split_seed,
        )
        suffix = f"{cfg.mono_source}_{cfg.feature_set}"
        out = args.output_dir / f"feature_audit_{suffix}.json"
        flip_out = args.output_dir / f"feature_flip_audit_{suffix}.json"
        report = run_feature_audit(
            train_records, cfg, out_path=out, flip_out_path=flip_out
        )
        top = report.get("ranked_by_effect_size", [])[:5]
        print(f"Feature audit written to {out}")
        print(f"  Top features by |Cohen's d|: {top}")
        if flip_out.exists():
            flip = json.loads(flip_out.read_text(encoding="utf-8"))
            anticorr = flip.get("ranked_by_anticorrelation", [])[:5]
            sep_flip = flip.get("separation_flips_with_effect", [])[:8]
            print(f"Time-reverse flip audit written to {flip_out}")
            print(f"  Most anticorrelated (fwd vs rev): {anticorr}")
            print(f"  Separation sign flips (|d|>0.1): {sep_flip}")


if __name__ == "__main__":
    main()
