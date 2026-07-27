"""CLI entry points for split generation and leakage verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_SPLITS_DIR, DEFAULT_REAL_ROOT
from .leakage import leakage_report
from .splits import (
    build_lovo_folds,
    build_scene_split,
    build_speed_stratified_split,
    load_split,
    save_split,
)


def _add_common_split_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--real_root", type=Path, default=DEFAULT_REAL_ROOT)
    p.add_argument("--synth_root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.85)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON or directory (LOVO writes a folder)",
    )


def make_split_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Build a leakage-aware pass-by split. Default is scene-level "
            "(vehicles may overlap). Opt into LOVO or speed-gap spreading "
            "via --protocol / --min-speed-gap, or use the dedicated scripts."
        )
    )
    p.add_argument(
        "--protocol",
        choices=("scene", "lovo", "speed_stratified"),
        default="scene",
    )
    _add_common_split_args(p)
    p.add_argument(
        "--min-speed-gap",
        type=int,
        default=1,
        help=(
            "Scene protocol only. 1=default (exact vehicle|speed keys). "
            "2=also ban same-vehicle +/-1 km/h near-twins (spread mode)."
        ),
    )
    args = p.parse_args(argv)
    return _run_make_split(args)


def make_lovo_main(argv: list[str] | None = None) -> int:
    """Dedicated LOVO entry (unknown-vehicle folds)."""
    p = argparse.ArgumentParser(
        description=(
            "Leave-one-vehicle-out folds. No train clip shares the held-out car. "
            "Use this when claiming unknown-vehicle performance."
        )
    )
    _add_common_split_args(p)
    args = p.parse_args(argv)
    args.protocol = "lovo"
    args.min_speed_gap = 1
    return _run_make_split(args)


def make_spread_scene_main(argv: list[str] | None = None) -> int:
    """Scene split with opt-in near-twin spreading (default gap=2)."""
    p = argparse.ArgumentParser(
        description=(
            "Scene-level split that also enforces a minimum same-vehicle "
            "speed gap between train and test (default 2 km/h → no +/-1 twins). "
            "Still not LOVO — cars can appear in both sets."
        )
    )
    _add_common_split_args(p)
    p.add_argument(
        "--min-speed-gap",
        type=int,
        default=2,
        help="Minimum |speed_train-speed_test| for the same vehicle (default: 2).",
    )
    args = p.parse_args(argv)
    args.protocol = "scene"
    return _run_make_split(args)


def _run_make_split(args: argparse.Namespace) -> int:
    out = args.out
    protocol = args.protocol

    if protocol == "scene":
        gap = int(getattr(args, "min_speed_gap", 1) or 1)
        split = build_scene_split(
            args.real_root,
            args.synth_root,
            train_frac=args.train_frac,
            seed=args.seed,
            val_frac=args.val_frac,
            min_speed_gap=gap,
        )
        suffix = f"_gap{gap}" if gap > 1 else ""
        out = out or (
            DEFAULT_SPLITS_DIR / f"scene_split_s{args.seed}{suffix}.json"
        )
        save_split(out, split)
        report = leakage_report(split)
        report.raise_if_failed()
        c = split.to_dict()["counts"]
        print(f"Wrote {out}")
        print(
            f"protocol=scene min_speed_gap={gap} "
            f"real train/test={c['n_real_train']}/{c['n_real_test']} "
            f"fit/val={c['n_fit']}/{c['n_val']} "
            f"synth train/test={c['n_synth_train']}/{c['n_synth_test']}"
        )
        demoted = (split.meta or {}).get("n_demoted_for_speed_gap", 0)
        if demoted:
            print(f"Demoted {demoted} test clips into train to satisfy speed gap")
        for w in report.warnings:
            print(f"WARN: {w}")
        return 0

    if protocol == "speed_stratified":
        split = build_speed_stratified_split(
            args.real_root, train_frac=min(args.train_frac, 0.7), seed=args.seed
        )
        out = out or (
            DEFAULT_SPLITS_DIR / f"speed_stratified_s{args.seed}.json"
        )
        save_split(out, split)
        leakage_report(split).raise_if_failed()
        print(f"Wrote {out}  train={len(split.train_uids)} test={len(split.test_uids)}")
        return 0

    # lovo — write one JSON per fold + an index
    folds = build_lovo_folds(args.real_root)
    out_dir = out or (DEFAULT_SPLITS_DIR / "lovo")
    out_dir = Path(out_dir)
    if out_dir.suffix == ".json":
        out_dir = out_dir.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for fold in folds:
        leakage_report(fold).raise_if_failed()
        path = out_dir / f"fold_{fold.fold_id:02d}_{fold.held_out_vehicle}.json"
        save_split(path, fold)
        index.append(
            {
                "fold_id": fold.fold_id,
                "held_out_vehicle": fold.held_out_vehicle,
                "path": str(path),
                "n_train": len(fold.train_uids),
                "n_test": len(fold.test_uids),
            }
        )
        print(f"  fold {fold.fold_id}: hold out {fold.held_out_vehicle} -> {path.name}")
    idx_path = out_dir / "index.json"
    idx_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(folds)} LOVO folds under {out_dir}")
    print("Claim: unknown vehicle (no car overlap between train and test).")
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fail loud if a split leaks.")
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--stats-uids", type=Path, default=None, help="JSON list of stats uids")
    p.add_argument("--eval-uids", type=Path, default=None)
    p.add_argument("--seresnet-train-uids", type=Path, default=None)
    p.add_argument("--dit-train-uids", type=Path, default=None)
    args = p.parse_args(argv)

    def _load_list(path: Path | None) -> list[str] | None:
        if path is None:
            return None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        for key in (
            "train_uids",
            "uids",
            "fit_uids",
            "s_train_uids",
            "eval_uids",
            "paths",
        ):
            if key in data:
                return data[key]
        raise ValueError(f"No uid list in {path}")

    split = load_split(args.split)
    report = leakage_report(
        split,
        stats_uids=_load_list(args.stats_uids),
        eval_uids=_load_list(args.eval_uids),
        seresnet_train_uids=_load_list(args.seresnet_train_uids),
        dit_train_uids=_load_list(args.dit_train_uids),
    )
    for w in report.warnings:
        print(f"WARN: {w}")
    if not report.ok:
        print("LEAKAGE CHECK FAILED:")
        for e in report.errors:
            print(f"  - {e}")
        return 1
    print("LEAKAGE CHECK OK")
    gap = (split.meta or {}).get("min_speed_gap", 1)
    print(
        f"  protocol={split.protocol} min_speed_gap={gap} "
        f"train={len(split.train_uids)} test={len(split.test_uids)} "
        f"test_scenes={len(split.test_scene_keys)}"
    )
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if cmd in ("make-split", "make_split"):
        raise SystemExit(make_split_main(rest))
    if cmd in ("make-lovo", "lovo"):
        raise SystemExit(make_lovo_main(rest))
    if cmd in ("make-spread", "spread"):
        raise SystemExit(make_spread_scene_main(rest))
    if cmd in ("verify", "verify-leakage"):
        raise SystemExit(verify_main(rest))
    print(
        "Usage: python -m passby_data.cli make-split|make-lovo|make-spread|verify ...",
        file=sys.stderr,
    )
    raise SystemExit(2)
