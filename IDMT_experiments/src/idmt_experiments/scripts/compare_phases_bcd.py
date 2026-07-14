#!/usr/bin/env python3
"""Build comparison table for Phases B/C/D vs reference models (2-class direction)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR
from idmt_experiments.cnn.eval import run_eval as cnn_run_eval
from idmt_experiments.fusion.eval import run_fusion_eval
from idmt_experiments.hybrid.eval import run_eval as hybrid_run_eval
from idmt_experiments.transfer.eval import run_eval as transfer_run_eval


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{100.0 * x:.1f}%"


def _vehicle_metrics(metrics: dict | None) -> tuple[float | None, float | None, dict | None]:
    """Balanced acc, flip agreement, and per-class recall for vehicle-only L2R/R2L."""
    if not metrics:
        return None, None, None
    vo = metrics.get("vehicle_only") or {}
    bal = vo.get("vehicle_balanced_accuracy")
    recall = vo.get("vehicle_per_class_recall")
    flip = vo.get("vehicle_flip_agreement")
    if bal is None:
        pr = metrics.get("per_class_recall") or {}
        if "L2R" in pr and "R2L" in pr:
            recall = {"L2R": pr["L2R"], "R2L": pr["R2L"]}
            bal = (pr["L2R"] + pr["R2L"]) / 2.0
    if flip is None:
        cs = metrics.get("channel_swap") or {}
        flip = cs.get("flip_consistency") or metrics.get("flip_agreement")
    return bal, flip, recall


def _recall(metrics: dict | None, label: str, vehicle_recall: dict | None = None) -> str:
    if vehicle_recall and label in vehicle_recall:
        return _fmt_pct(vehicle_recall[label])
    if not metrics:
        return "—"
    return _fmt_pct(metrics.get("per_class_recall", {}).get(label))


def build_table(rows: list[dict]) -> str:
    header = (
        "| Phase | Run | Bal. acc | L2R recall | R2L recall | Flip agree. | Notes |\n"
        "|-------|-----|----------|------------|------------|-------------|-------|"
    )
    lines = [header]
    for r in rows:
        vm = r.get("vehicle_recall")
        metrics = r.get("metrics")
        lines.append(
            f"| {r['phase']} | `{r['run']}` | {_fmt_pct(r.get('bal_acc'))} | "
            f"{_recall(metrics, 'L2R', vm)} | {_recall(metrics, 'R2L', vm)} | "
            f"{_fmt_pct(r.get('flip'))} | {r.get('notes', '')} |"
        )
    return "\n".join(lines)


def build_doc(baseline_rows: list[dict], phase_rows: list[dict]) -> str:
    intro = (
        "# 2-class direction comparison (L2R vs R2L)\n\n"
        "**Metric:** balanced accuracy on **test** split, **vehicle clips only** (no `no_vehicle`).\n\n"
    )
    targets = (
        "## Targets (from plan)\n\n"
        "| Phase | Target |\n"
        "|-------|--------|\n"
        "| B | >= 80.5% |\n"
        "| C | >= 81.0% |\n"
        "| D | accuracy + flip consistency |\n"
    )
    return (
        intro
        + "## Baselines\n\n"
        + build_table(baseline_rows)
        + "\n\n## Phases B–D (100 epochs)\n\n"
        + build_table(phase_rows)
        + "\n\n"
        + targets
    )


def _row_from_metrics(
    spec: dict,
    metrics: dict | None,
    *,
    bal_acc=None,
    flip=None,
    vehicle_recall=None,
) -> dict:
    if spec.get("kind") == "cnn_3class_vehicle" and metrics:
        bal_acc, flip, vehicle_recall = _vehicle_metrics(metrics)
    return {
        "phase": spec["phase"],
        "run": spec["run"],
        "metrics": metrics,
        "bal_acc": bal_acc if bal_acc is not None else (metrics.get("balanced_accuracy") if metrics else None),
        "flip": flip,
        "vehicle_recall": vehicle_recall,
        "notes": spec.get("notes", ""),
    }


def _collect_row(spec: dict, *, ckpt_root: Path, out_root: Path, refresh: bool) -> dict:
    kind = spec["kind"]
    run = spec["run"]

    if kind == "fusion":
        metrics_path = out_root / "fusion/direction" / run / "eval_metrics.json"
        if refresh or not metrics_path.exists():
            if spec["left"].exists() and spec["right"].exists():
                run_fusion_eval(spec["left"], spec["right"], run_name=run, device=spec.get("device", "auto"))
        metrics = _load_metrics(metrics_path)
        return _row_from_metrics(
            spec,
            metrics,
            bal_acc=metrics.get("balanced_accuracy") if metrics else None,
            flip=metrics.get("channel_swap_agreement"),
        )

    if kind == "fusion_baseline":
        metrics_path = out_root / "fusion/direction" / run / "eval_metrics.json"
        metrics = _load_metrics(metrics_path)
        return _row_from_metrics(
            spec,
            metrics,
            bal_acc=metrics.get("balanced_accuracy") if metrics else None,
            flip=metrics.get("channel_swap_agreement"),
        )

    if kind == "metrics_only":
        metrics_path = spec.get("metrics_path") or (
            out_root / spec.get("metrics_subdir", "") / run / "eval_metrics.json"
        )
        metrics = _load_metrics(Path(metrics_path))
        bal, flip, recall = _vehicle_metrics(metrics)
        if bal is None and metrics:
            bal = metrics.get("balanced_accuracy")
            flip = flip or metrics.get("flip_agreement") or metrics.get("flip_consistency")
            cs = metrics.get("channel_swap") or {}
            flip = flip or cs.get("flip_consistency") or metrics.get("channel_swap_agreement")
        return _row_from_metrics(spec, metrics, bal_acc=bal, flip=flip, vehicle_recall=recall)

    if kind == "cnn_3class_vehicle":
        out_sub = "cnn/direction"
        metrics_path = out_root / out_sub / run / "eval_metrics.json"
        ckpt = spec.get("ckpt")
        if refresh and ckpt and ckpt.exists():
            cnn_run_eval(checkpoint=ckpt, device=spec.get("device", "auto"))
        metrics = _load_metrics(metrics_path)
        return _row_from_metrics(spec, metrics)

    if kind == "physics":
        out_sub = "physics/direction"
        metrics_path = out_root / out_sub / run / "eval_metrics.json"
        run_dir = spec.get("run_dir")
        if refresh or not metrics_path.exists():
            if run_dir and run_dir.exists():
                from idmt_experiments.physics.eval import run_eval as physics_run_eval

                physics_run_eval(run_dir=run_dir)
        metrics = _load_metrics(metrics_path)
        return _row_from_metrics(
            spec,
            metrics,
            flip=metrics.get("flip_agreement") if metrics else None,
        )

    if kind == "hybrid":
        out_sub = "hybrid/direction"
        metrics_path = out_root / out_sub / run / "eval_metrics.json"
        ckpt = spec.get("ckpt")
        if refresh or not metrics_path.exists():
            if ckpt and ckpt.exists():
                hybrid_run_eval(checkpoint=ckpt, device=spec.get("device", "auto"))
        metrics = _load_metrics(metrics_path)
        bal, flip, recall = _vehicle_metrics(metrics)
        return _row_from_metrics(spec, metrics, bal_acc=bal, flip=flip, vehicle_recall=recall)

    # transfer / film
    out_sub = "transfer/direction" if kind == "transfer" else "hybrid/direction"
    metrics_path = out_root / out_sub / run / "eval_metrics.json"
    ckpt = spec.get("ckpt")
    if refresh or not metrics_path.exists():
        if ckpt and ckpt.exists():
            if kind == "transfer":
                transfer_run_eval(ckpt, output_subdir=out_sub, device=spec.get("device", "auto"))
            else:
                hybrid_run_eval(checkpoint=ckpt, device=spec.get("device", "auto"))
    metrics = _load_metrics(metrics_path)
    flip = None
    if metrics:
        flip = metrics.get("flip_agreement") or metrics.get("flip_consistency")
        cs = metrics.get("channel_swap") or {}
        flip = flip or cs.get("flip_consistency")
    return _row_from_metrics(spec, metrics, flip=flip)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare 2-class direction runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--refresh", action="store_true", help="Re-run eval for all checkpoints")
    args = p.parse_args()

    ckpt_root = Path(DEFAULT_CHECKPOINT_DIR)
    out_root = Path(DEFAULT_OUTPUT_DIR)

    baseline_specs = [
        {
            "phase": "ref",
            "run": "mel_3class_left",
            "ckpt": ckpt_root / "cnn/direction/mel_3class_left/best.pt",
            "kind": "cnn_3class_vehicle",
            "notes": "CNN baseline — mono left, 40 ep + preempt",
        },
        {
            "phase": "ref",
            "run": "mel_3class_right",
            "ckpt": ckpt_root / "cnn/direction/mel_3class_right/best.pt",
            "kind": "cnn_3class_vehicle",
            "notes": "CNN baseline — mono right",
        },
        {
            "phase": "ref",
            "run": "mel_3class",
            "ckpt": ckpt_root / "cnn/direction/mel_3class/best.pt",
            "kind": "cnn_3class_vehicle",
            "notes": "CNN baseline — stereo mean downmix (L+R)/2",
        },
        {
            "phase": "ref",
            "run": "mel_3class_left_ep60",
            "ckpt": ckpt_root / "cnn/direction/mel_3class_left_ep60/best.pt",
            "kind": "cnn_3class_vehicle",
            "notes": "CNN — 60 ep, no preempt (intervention baseline)",
        },
        {
            "phase": "ref",
            "run": "mel_3class_left_ep200",
            "ckpt": ckpt_root / "cnn/direction/mel_3class_left_ep200/best.pt",
            "kind": "cnn_3class_vehicle",
            "notes": "CNN — 200 ep, no preempt",
        },
        {
            "phase": "ref",
            "run": "mel_3class_left_aug_v1",
            "ckpt": ckpt_root / "cnn/direction/mel_3class_left_aug_v1/best.pt",
            "kind": "cnn_3class_vehicle",
            "notes": "Phase A — SpecAugment + focal + balanced sampler",
        },
        {
            "phase": "ref",
            "run": "physics_mlp_full_left",
            "run_dir": ckpt_root / "physics/direction/physics_mlp_full_left",
            "kind": "physics",
            "notes": "Physics MLP — kinematic_full features, mono left",
        },
        {
            "phase": "ref",
            "run": "hybrid_mel_left_v3_ep60",
            "ckpt": ckpt_root / "hybrid/direction/hybrid_mel_left_v3_ep60/best.pt",
            "kind": "hybrid",
            "notes": "Archived hybrid PINN — mel + physics late fusion, 60 ep",
        },
        {
            "phase": "ref",
            "run": "fusion_cnn_baseline_2class",
            "kind": "fusion_baseline",
            "notes": "CNN L+R late fusion (mel_3class left+right, w_L=0.15)",
        },
    ]

    phase_specs = [
        {
            "phase": "B",
            "run": "deep_mel_2class_left_100ep",
            "ckpt": ckpt_root / "transfer/direction/deep_mel_2class_left_100ep/best.pt",
            "kind": "transfer",
            "notes": "Deep residual mel CNN, mono left",
        },
        {
            "phase": "B",
            "run": "deep_mel_2class_right_100ep",
            "ckpt": ckpt_root / "transfer/direction/deep_mel_2class_right_100ep/best.pt",
            "kind": "transfer",
            "notes": "Deep residual mel CNN, mono right",
        },
        {
            "phase": "C",
            "run": "fusion_2class_100ep",
            "kind": "fusion",
            "left": ckpt_root / "transfer/direction/deep_mel_2class_left_100ep/best.pt",
            "right": ckpt_root / "transfer/direction/deep_mel_2class_right_100ep/best.pt",
            "notes": "Late fusion deep mel L+R (weight fit on valid)",
        },
        {
            "phase": "D",
            "run": "film_2class_left_100ep",
            "kind": "metrics_only",
            "metrics_path": out_root / "hybrid/direction/film_2class_left_100ep/eval_metrics.json",
            "notes": "FiLM + flip-consistency loss, mono left (archived metrics; weights removed)",
        },
    ]

    device = args.device
    for spec in baseline_specs + phase_specs:
        spec["device"] = device

    baseline_rows = [_collect_row(s, ckpt_root=ckpt_root, out_root=out_root, refresh=args.refresh) for s in baseline_specs]
    phase_rows = [_collect_row(s, ckpt_root=ckpt_root, out_root=out_root, refresh=args.refresh) for s in phase_specs]

    doc = build_doc(baseline_rows, phase_rows)
    out_path = out_root / "phases_bcd_comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    print(doc)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
