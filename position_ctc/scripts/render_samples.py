"""Write a markdown gallery of held-out position predictions.

    position_ctc/.venv/bin/python position_ctc/scripts/render_samples.py
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
from position_ctc.metrics import align_path, range_m, score_path  # noqa: E402
from position_ctc.paths import resolve_audio_clips  # noqa: E402

DT = DATA_HOP / float(DATA_SR)
CKPT = ROOT / "position_ctc" / "outputs" / "ablate" / "range_polar_ft.pt"
OUT = ROOT / "position_ctc" / "outputs" / "ablate" / "samples"
GT_COLOR = "#1f4e79"
PRED_COLOR = "#d35400"
ALIGN_COLOR = "#0e7c66"


def _load_ablate():
    path = ROOT / "position_ctc" / "scripts" / "ablate_position.py"
    spec = importlib.util.spec_from_file_location("ablate_position", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predict(model, batch_fn, row, device):
    feat, gt, lengths = batch_fn([row], None, 320)
    with torch.no_grad():
        pred, _ = model(feat.to(device), gt.to(device), lengths.to(device))
    n = row["xy"].shape[0]
    return pred[0, :n].cpu().numpy()


def _pick(scored: list[dict]) -> list[dict]:
    chosen = []
    by_vehicle: dict[str, list[dict]] = {}
    for row in scored:
        by_vehicle.setdefault(row["vehicle"], []).append(row)
    for vehicle in sorted(by_vehicle):
        rows = sorted(by_vehicle[vehicle], key=lambda item: item["score"]["orbit_rms_m"])
        slots = {
            "best": rows[0],
            "median": rows[len(rows) // 2],
            "worst": rows[-1],
        }
        for rank, row in slots.items():
            chosen.append({**row, "rank": rank})
    return chosen


def _draw(path: Path, sample: dict) -> None:
    gt = sample["xy"]
    pred = sample["pred"]
    aligned, reflected, _ = align_path(pred, gt)
    time = np.arange(len(gt)) * DT
    gt_r = range_m(gt)
    pred_r = range_m(pred)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True)

    def _path(ax, other, color, label):
        ax.plot(gt[:, 0], gt[:, 1], color=GT_COLOR, lw=2.0, label="true path")
        ax.plot(other[:, 0], other[:, 1], color=color, lw=1.8, ls="--", label=label)
        ax.scatter([0], [0], c="black", marker="x", s=46, zorder=3, label="mic")
        ax.scatter(gt[int(np.argmin(gt_r)), 0], gt[int(np.argmin(gt_r)), 1], c=GT_COLOR, s=28, zorder=4)
        ax.scatter(other[int(np.argmin(range_m(other))), 0], other[int(np.argmin(range_m(other))), 1], c=color, s=28, zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8, loc="best")

    _path(axes[0], pred, PRED_COLOR, "prediction")
    axes[0].set_title("Canonical frame")
    align_label = "aligned, reflected" if reflected else "aligned"
    _path(axes[1], aligned, ALIGN_COLOR, align_label)
    axes[1].set_title("Best rotation about the mic")

    axes[2].plot(time, gt_r, color=GT_COLOR, lw=2.0, label="true range")
    axes[2].plot(time, pred_r, color=PRED_COLOR, lw=1.8, ls="--", label="predicted range")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("range (m)")
    axes[2].set_title("Distance from the mic")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _markdown(chosen: list[dict]) -> str:
    lines = [
        "# Sample position predictions",
        "",
        "Held-out clips from the fine-tuned range-and-direction model",
        "(`range_polar_ft.pt`). Test vehicles are Kia Sportage and Renault Captur;",
        "neither was seen in training.",
        "",
        "The left panel is the network output in the canonical training frame",
        "(closest approach on the positive y-axis). The middle panel rotates,",
        "and if needed reflects, that output about the microphone so the path",
        "lines up with the true trajectory. Dots mark the closest point.",
        "Range does not change under that alignment.",
        "",
        "Orbit RMS is the position error after that alignment. Range MAE is the",
        "average absolute gap between the two range curves. CPA is the absolute",
        "error of the closest-point distance.",
        "",
        "| Clip | Vehicle | Case | Orbit RMS | Range MAE | CPA error |",
        "|---|---|---|---:|---:|---:|",
    ]
    for sample in chosen:
        score = sample["score"]
        lines.append(
            f"| `{sample['clip_id']}` | {sample['vehicle']} | {sample['rank']} | "
            f"{score['orbit_rms_m']:.2f} m | {score['range_mae_m']:.2f} m | {score['cpa_abs_m']:.2f} m |"
        )
    lines.append("")
    for sample in chosen:
        score = sample["score"]
        image = f"samples/{sample['clip_id']}.png"
        lines.extend(
            [
                f"## {sample['vehicle']}, {sample['rank']}",
                "",
                f"`{sample['clip_id']}` — orbit {score['orbit_rms_m']:.2f} m, "
                f"range MAE {score['range_mae_m']:.2f} m, "
                f"closest-point error {score['cpa_abs_m']:.2f} m.",
                "",
                f"![{sample['vehicle']} {sample['rank']} path]({image})",
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
    for index in splits["test"]:
        clip = clips[index]
        feat, state_frames = load_arrays(clip)
        row = {
            "clip_id": clip.clip_id,
            "vehicle": clip.vehicle,
            "feat": feat.astype(np.float32),
            "xy": np.ascontiguousarray(state_frames[:, [0, 2]], dtype=np.float32),
        }
        pred = _predict(model, ablate._batch, row, device)
        row["pred"] = pred
        row["score"] = score_path(pred, row["xy"])
        scored.append(row)
        print(f"{clip.clip_id} {clip.vehicle} orbit={row['score']['orbit_rms_m']:.2f}", flush=True)

    chosen = _pick(scored)
    OUT.mkdir(parents=True, exist_ok=True)
    for sample in chosen:
        _draw(OUT / f"{sample['clip_id']}.png", sample)
    report = ROOT / "position_ctc" / "outputs" / "ablate" / "sample_predictions.md"
    report.write_text(_markdown(chosen), encoding="utf-8")
    print(f"wrote {report}", flush=True)


if __name__ == "__main__":
    main()
