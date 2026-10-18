"""Four-graph and heatmap gallery for the trajectory86_fixed test split.

    position_ctc/.venv/bin/python position_ctc/scripts/plot_test_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "scripts"))

from trajectory_framewise import (  # noqa: E402
    ClipStore,
    MotionFramewise,
    infer,
    save_example,
    scores,
)

RUN = ROOT / "position_ctc" / "outputs" / "trajectory86_fixed"
OUT = RUN / "test_cases"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    device = _device()
    config = json.loads((RUN / "config.json").read_text())
    normalization = json.loads((RUN / "normalization.json").read_text())
    records = json.loads((RUN / "records.json").read_text())
    splits = json.loads((RUN / "splits.json").read_text())
    checkpoint = torch.load(RUN / "best.pt", map_location=device, weights_only=True)
    model = MotionFramewise(config).to(device)
    model.load_state_dict(checkpoint["model"])
    store = ClipStore(records, config, RUN / "cache")
    centers = model.centers.cpu().numpy()
    scored = []
    for clip_id in splits["test"]:
        clip = store.get(clip_id)
        prediction = infer(model, clip, normalization, device, config)
        row = {"clip_id": clip_id, **scores(clip, prediction, config["bin_width"])}
        row["directory"] = save_example(clip_id, clip, prediction, centers, OUT)
        row["_prediction"] = prediction
        scored.append(row)
        print(f"{clip_id}  rmse={row['trajectory_rmse']:.2f} m  vy={row['vy_mae']:.2f}", flush=True)

    ranked = sorted(scored, key=lambda item: item["trajectory_rmse"])
    by_vy = sorted(scored, key=lambda item: item["vy_mae"])
    picks = {
        "best trajectory RMSE": ranked[0],
        "median trajectory RMSE": ranked[len(ranked) // 2],
        "worst trajectory RMSE": ranked[-1],
        "highest vy MAE": by_vy[-1],
    }
    seen = set()
    chosen = []
    for label, item in picks.items():
        if item["clip_id"] in seen:
            continue
        seen.add(item["clip_id"])
        chosen.append((label, item))

    rel = lambda path: Path(path).resolve().relative_to(OUT).as_posix()
    lines = [
        "# Test cases: four graphs and probability heatmaps",
        "",
        f"Checkpoint `trajectory86_fixed/best.pt`, epoch {int(checkpoint['epoch'])}.",
        "These are the **test** split clips (`splits.json`), not the validation gallery.",
        "The four graphs are x(t), y(t), vx(t), vy(t).",
        "The heatmaps are the dense-head bin probabilities before Viterbi.",
        "",
        f"All {len(scored)} test clips: mean trajectory RMSE "
        f"{float(np.mean([r['trajectory_rmse'] for r in scored])):.2f} m, "
        f"mean vy MAE {float(np.mean([r['vy_mae'] for r in scored])):.2f} m/s.",
        "",
        "| Case | Clip | Trajectory RMSE | vx MAE | vy MAE | Endpoint |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, item in chosen:
        lines.append(
            f"| {label} | `{item['clip_id']}` | {item['trajectory_rmse']:.2f} m | "
            f"{item['vx_mae']:.2f} m/s | {item['vy_mae']:.2f} m/s | {item['endpoint_error']:.2f} m |"
        )
    lines.append("")
    for label, item in chosen:
        folder = Path(item["directory"]).name
        lines.extend(
            [
                f"## {label}",
                "",
                f"`{item['clip_id']}` — trajectory RMSE {item['trajectory_rmse']:.2f} m, "
                f"vy MAE {item['vy_mae']:.2f} m/s.",
                "",
                f"![four graphs]({folder}/four_graphs.png)",
                "",
                f"![heatmaps]({folder}/probability_heatmaps.png)",
                "",
            ]
        )
    (OUT / "TEST_CASES.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'TEST_CASES.md'}", flush=True)


if __name__ == "__main__":
    main()
