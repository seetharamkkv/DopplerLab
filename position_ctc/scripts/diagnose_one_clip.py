"""Fit one clip with neighborhood cross-entropy and argmax decoding.

Same frequency CNN, BiLSTM, 2 m/s bins, and 86 Hz frames as trajectory_framewise.
CTC weight is zero, dropout is absent, and weight decay is zero. This checks
whether the encoder can track repeated turns on a clip it is allowed to memorize.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "scripts"))

from trajectory_framewise import (  # noqa: E402
    ClipStore,
    MotionFramewise,
    infer,
    save_example,
    scores,
    to_device,
)

RUN = ROOT / "position_ctc" / "outputs" / "trajectory86"
OUT = RUN / "one_clip" / "sample_0000014"
CLIP_ID = "audio_clips/sample_0000014"
EPOCHS = 300
LR = 1e-3
LOG_EVERY = 10


def _groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    named = list(model.named_parameters())
    return {
        "cnn": [p for n, p in named if n.startswith("cnn.")],
        "lstm": [p for n, p in named if n.startswith("lstm.")],
        "vel_x": [p for n, p in named if n.startswith("velocity.0.")],
        "vel_y": [p for n, p in named if n.startswith("velocity.1.")],
        "anchor": [p for n, p in named if n.startswith("anchor.")],
    }


def _norm(parameters: list[nn.Parameter], grad: bool) -> float:
    total = 0.0
    for parameter in parameters:
        value = parameter.grad if grad else parameter.detach()
        if value is None:
            continue
        total += float(value.detach().float().pow(2).sum())
    return total ** 0.5


def _neighborhood(dense: torch.Tensor, truth: torch.Tensor, centers: torch.Tensor, sigma: float):
    distance = truth[..., None] - centers
    targets = torch.softmax(-0.5 * (distance / sigma).square(), dim=-1)
    per_frame = -(targets * dense.log_softmax(-1)).sum(-1)
    return per_frame[:, 0].mean(), per_frame[:, 1].mean()


@torch.no_grad()
def _frame_report(dense: torch.Tensor, truth: torch.Tensor, centers: torch.Tensor) -> dict[str, float]:
    probs = dense.softmax(-1)
    pred = probs.argmax(-1)
    true_bins = (truth[..., None] - centers).abs().argmin(-1)
    pred_v = centers[pred]
    report = {
        "vx_mae": float((pred_v[:, 0] - truth[:, 0]).abs().mean()),
        "vy_mae": float((pred_v[:, 1] - truth[:, 1]).abs().mean()),
        "vx_bin_acc": float((pred[:, 0] == true_bins[:, 0]).float().mean()),
        "vy_bin_acc": float((pred[:, 1] == true_bins[:, 1]).float().mean()),
        "vx_prob_std": float(probs[:, 0].std(dim=0).mean()),
        "vy_prob_std": float(probs[:, 1].std(dim=0).mean()),
    }
    vy_bins = true_bins[:, 1]
    zero = int((centers - 0).abs().argmin())
    for name, mask in (
        ("pos", vy_bins > zero),
        ("neg", vy_bins < zero),
        ("zero", vy_bins == zero),
    ):
        report[f"vy_{name}_frames"] = int(mask.sum())
        if int(mask.sum()) == 0:
            report[f"vy_{name}_prob"] = float("nan")
            report[f"vy_{name}_acc"] = float("nan")
            continue
        gathered = probs[:, 1][mask, vy_bins[mask]]
        report[f"vy_{name}_prob"] = float(gathered.mean())
        report[f"vy_{name}_acc"] = float((pred[:, 1][mask] == vy_bins[mask]).float().mean())
    return report


def _snapshot(groups: dict[str, list[nn.Parameter]]) -> dict[str, list[torch.Tensor]]:
    return {name: [p.detach().clone() for p in params] for name, params in groups.items()}


def _update_norm(groups, before) -> dict[str, float]:
    out = {}
    for name, params in groups.items():
        total = 0.0
        for parameter, old in zip(params, before[name]):
            total += float((parameter.detach() - old).float().pow(2).sum())
        out[name] = total ** 0.5
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((RUN / "config.json").read_text())
    normalization = json.loads((RUN / "normalization.json").read_text())
    records = json.loads((RUN / "records.json").read_text())
    device = torch.device("cpu")
    store = ClipStore(records, config, RUN / "cache")
    clip = store.get(CLIP_ID)
    features, selected, times, p0, truth = to_device(clip, normalization, device)
    centers_ref = torch.arange(-config["vmax"], config["vmax"] + config["bin_width"] / 2, config["bin_width"])

    held = MotionFramewise(config).to(device)
    saved = torch.load(RUN / "best.pt", map_location=device, weights_only=True)
    held.load_state_dict(saved["model"])
    held.eval()
    comparison = {}
    for name, decoder in (("argmax", "greedy"), ("viterbi", "viterbi")):
        cfg = dict(config, decoder=decoder)
        prediction = infer(held, clip, normalization, device, cfg)
        comparison[name] = scores(clip, prediction, config["bin_width"])
    del held

    torch.manual_seed(0)
    model = MotionFramewise(config).to(device)
    groups = _groups(model)
    for name, params in groups.items():
        if not params:
            raise RuntimeError(f"no parameters in {name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    sigma = float(config["sigma"])
    history = []
    grad_report = None
    update_report = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        before = _snapshot(groups) if epoch == 1 else None
        dense, _ctc = model(features, selected, p0)
        loss_x, loss_y = _neighborhood(dense, truth, model.centers, sigma)
        loss = loss_x + loss_y
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite neighborhood loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if epoch == 1:
            grad_report = {name: _norm(params, grad=True) for name, params in groups.items()}
            if min(grad_report[name] for name in ("cnn", "lstm", "vel_x", "vel_y")) <= 0:
                raise RuntimeError(f"a required gradient is zero: {grad_report}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 1:
            update_report = _update_norm(groups, before)
            if min(update_report[name] for name in ("cnn", "lstm", "vel_x", "vel_y")) <= 0:
                raise RuntimeError(f"a required parameter did not change: {update_report}")
        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                dense, _ctc = model(features, selected, p0)
                report = _frame_report(dense, truth, model.centers)
            row = {
                "epoch": epoch,
                "loss_x": float(loss_x.detach()),
                "loss_y": float(loss_y.detach()),
                "loss": float(loss.detach()),
                **report,
            }
            history.append(row)
            print(
                f"epoch {epoch:03d}  loss x {row['loss_x']:.3f} y {row['loss_y']:.3f}  "
                f"mae vx {row['vx_mae']:.3f} vy {row['vy_mae']:.3f}  "
                f"vy p pos {row['vy_pos_prob']:.3f} neg {row['vy_neg_prob']:.3f} "
                f"zero {row['vy_zero_prob']:.3f}  "
                f"acc pos {row['vy_pos_acc']:.3f} neg {row['vy_neg_acc']:.3f}",
                flush=True,
            )
            tracked = (
                row["vx_mae"] < 0.75
                and row["vy_mae"] < 0.75
                and row["vy_pos_acc"] >= 0.8
                and row["vy_neg_acc"] >= 0.8
            )
            if tracked and epoch >= 30:
                break

    model.eval()
    fitted = {}
    for name, decoder in (("argmax", "greedy"), ("viterbi", "viterbi")):
        cfg = dict(config, decoder=decoder, ctc_weight=0.0, vy_balance=0.0)
        prediction = infer(model, clip, normalization, device, cfg)
        fitted[name] = scores(clip, prediction, config["bin_width"])
        if name == "argmax":
            save_example(CLIP_ID, clip, prediction, model.centers.cpu().numpy(), OUT)
    with (OUT / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "clip_id": CLIP_ID,
        "objective": "neighborhood cross-entropy on vx and vy, CTC weight 0, weight decay 0, no dropout",
        "decode_while_fitting": "argmax",
        "lr": LR,
        "sigma_mps": sigma,
        "bin_width_mps": config["bin_width"],
        "epochs_ran": history[-1]["epoch"],
        "checkpoint_argmax_vs_viterbi": comparison,
        "gradients_epoch_1": grad_report,
        "parameter_update_epoch_1": update_report,
        "fitted_argmax_vs_viterbi": fitted,
        "final": history[-1],
        "centers_match": bool(torch.equal(model.centers.cpu(), centers_ref)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
