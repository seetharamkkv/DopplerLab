"""Inference with the feasible-turn projection.

    position_ctc/.venv/bin/python position_ctc/scripts/infer_position.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.data import discover_clips, load_arrays, split_by_vehicle  # noqa: E402
from position_ctc.features import DATA_HOP, DATA_SR  # noqa: E402
from position_ctc.kinematics import project_path, violation_fraction  # noqa: E402
from position_ctc.metrics import aggregate, align_path, range_m, score_path  # noqa: E402
from position_ctc.paths import resolve_audio_clips  # noqa: E402

DT = DATA_HOP / float(DATA_SR)
CKPT = ROOT / "position_ctc" / "outputs" / "ablate" / "range_polar_ft.pt"
OUT = ROOT / "position_ctc" / "outputs" / "ablate" / "inference"
GT_COLOR = "#1f4e79"
PRED_COLOR = "#d35400"
PROJ_COLOR = "#0e7c66"


def _load_ablate():
    path = ROOT / "position_ctc" / "scripts" / "ablate_position.py"
    spec = importlib.util.spec_from_file_location("ablate_position", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick(scored: list[dict]) -> list[dict]:
    chosen = []
    by_vehicle: dict[str, list[dict]] = {}
    for row in scored:
        by_vehicle.setdefault(row["vehicle"], []).append(row)
    for vehicle in sorted(by_vehicle):
        rows = sorted(by_vehicle[vehicle], key=lambda item: item["raw_score"]["orbit_rms_m"])
        for rank, row in (("best", rows[0]), ("median", rows[len(rows) // 2]), ("worst", rows[-1])):
            chosen.append({**row, "rank": rank})
    return chosen


def _draw(path: Path, sample: dict) -> None:
    gt = sample["xy"]
    pred = sample["pred"]
    projected = sample["projected"]
    aligned, reflected, _ = align_path(projected, gt)
    time = np.arange(len(gt)) * DT
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True)

    def _path(ax, other, color, label):
        ax.plot(gt[:, 0], gt[:, 1], color=GT_COLOR, lw=2.0, label="true path")
        ax.plot(pred[:, 0], pred[:, 1], color=PRED_COLOR, lw=1.2, ls=":", label="network")
        ax.plot(other[:, 0], other[:, 1], color=color, lw=1.8, ls="--", label=label)
        ax.scatter([0], [0], c="black", marker="x", s=46, zorder=3, label="mic")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8, loc="best")

    _path(axes[0], projected, PROJ_COLOR, "feasible projection")
    axes[0].set_title("Canonical frame")
    align_label = "projected, reflected" if reflected else "projected, aligned"
    _path(axes[1], aligned, PROJ_COLOR, align_label)
    axes[1].set_title("Best rotation about the mic")
    axes[2].plot(time, range_m(gt), color=GT_COLOR, lw=2.0, label="true range")
    axes[2].plot(time, range_m(pred), color=PRED_COLOR, lw=1.2, ls=":", label="network")
    axes[2].plot(time, range_m(projected), color=PROJ_COLOR, lw=1.8, ls="--", label="projected")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("range (m)")
    axes[2].set_title("Distance from the mic")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _markdown(chosen: list[dict], raw: dict, projected: dict) -> str:
    lines = [
        "# Inference with the turn limit",
        "",
        "Held-out inference from `range_polar_ft.pt`. The network path is drawn dotted.",
        "The dashed path is that output after the yaw-rate projection. Test vehicles are",
        "Kia Sportage and Renault Captur.",
        "",
        f"All {int(raw['n'])} test clips: network orbit {raw['orbit_rms_m_mean']:.2f} m,",
        f"range {raw['range_mae_m_mean']:.2f} m. After projection: orbit {projected['orbit_rms_m_mean']:.2f} m,",
        f"range {projected['range_mae_m_mean']:.2f} m, closest point {projected['cpa_abs_m_mean']:.2f} m.",
        "",
        "| Clip | Vehicle | Case | Network orbit | Projected orbit | Projected range | Projected CPA |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for sample in chosen:
        raw_score = sample["raw_score"]
        proj_score = sample["proj_score"]
        lines.append(
            f"| `{sample['clip_id']}` | {sample['vehicle']} | {sample['rank']} | "
            f"{raw_score['orbit_rms_m']:.2f} m | {proj_score['orbit_rms_m']:.2f} m | "
            f"{proj_score['range_mae_m']:.2f} m | {proj_score['cpa_abs_m']:.2f} m |"
        )
    lines.append("")
    for sample in chosen:
        score = sample["proj_score"]
        lines.extend(
            [
                f"## {sample['vehicle']}, {sample['rank']}",
                "",
                f"`{sample['clip_id']}` — projected orbit {score['orbit_rms_m']:.2f} m, "
                f"range MAE {score['range_mae_m']:.2f} m, "
                f"closest-point error {score['cpa_abs_m']:.2f} m. "
                f"Yaw-limit fraction {sample['raw_violation']:.2f} before projection and "
                f"{sample['proj_violation']:.2f} after.",
                "",
                f"![{sample['vehicle']} {sample['rank']}](inference/{sample['clip_id']}.png)",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    ablate = _load_ablate()
    clips = discover_clips(resolve_audio_clips())
    splits = split_by_vehicle(clips, seed=0)
    device = ablate._device()
    model = ablate.PositionModel("range_polar").to(device)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    scored = []
    raw_scores = []
    proj_scores = []
    for index in splits["test"]:
        clip = clips[index]
        feat, state_frames = load_arrays(clip)
        row = {
            "clip_id": clip.clip_id,
            "vehicle": clip.vehicle,
            "feat": feat.astype(np.float32),
            "xy": np.ascontiguousarray(state_frames[:, [0, 2]], dtype=np.float32),
        }
        pred = ablate._predict_rows(model, [row], device)[0]
        projected = project_path(pred, DT)
        raw_score = score_path(pred, row["xy"])
        proj_score = score_path(projected, row["xy"])
        raw_scores.append(raw_score)
        proj_scores.append(proj_score)
        scored.append(
            {
                **row,
                "pred": pred,
                "projected": projected,
                "raw_score": raw_score,
                "proj_score": proj_score,
                "raw_violation": violation_fraction(pred, DT),
                "proj_violation": violation_fraction(projected, DT),
            }
        )
    raw = aggregate(raw_scores)
    projected_metrics = aggregate(proj_scores)
    print(
        f"network orbit={raw['orbit_rms_m_mean']:.2f}m range={raw['range_mae_m_mean']:.2f}m",
        flush=True,
    )
    print(
        f"projected orbit={projected_metrics['orbit_rms_m_mean']:.2f}m "
        f"range={projected_metrics['range_mae_m_mean']:.2f}m "
        f"cpa={projected_metrics['cpa_abs_m_mean']:.2f}m",
        flush=True,
    )
    chosen = _pick(scored)
    OUT.mkdir(parents=True, exist_ok=True)
    for sample in chosen:
        _draw(OUT / f"{sample['clip_id']}.png", sample)
    report = ROOT / "position_ctc" / "outputs" / "ablate" / "inference.md"
    report.write_text(_markdown(chosen, raw, projected_metrics), encoding="utf-8")
    print(f"wrote {report}", flush=True)


if __name__ == "__main__":
    main()
