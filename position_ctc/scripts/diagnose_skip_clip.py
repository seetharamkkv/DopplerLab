"""Single-clip fit with frequency pooling of 64, layer norm, and a CNN skip.

86 Hz frames, 2 m/s bins, separate velocity heads, anchored integration.
CTC stays off. Dropout and weight decay stay off. Argmax is the fit metric.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "scripts"))

from trajectory_framewise import (  # noqa: E402
    ClipStore,
    integrate_frames,
    save_example,
    to_device,
)

RUN = ROOT / "position_ctc" / "outputs" / "trajectory86"
OUT = RUN / "one_clip_skip64" / "sample_0000014"
CLIP_ID = "audio_clips/sample_0000014"
EPOCHS = 300
LR = 1e-3
CLIP_NORM = 5.0
INTERIOR = (1.5, 8.0)
POOL = 64


class SkipFramewise(nn.Module):
    """Frequency CNN, layer norm, BiLSTM, and a learned skip from the CNN."""

    def __init__(self, config: dict, freq_pool: int = POOL):
        super().__init__()
        hidden = int(config["hidden"])
        width, vmax = float(config["bin_width"]), float(config["vmax"])
        centers = torch.arange(-vmax, vmax + width / 2, width)
        self.register_buffer("centers", centers)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(freq_pool),
            nn.Flatten(),
            nn.Linear(32 * freq_pool, hidden),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(hidden)
        self.anchor = nn.Linear(2, 16)
        self.lstm = nn.LSTM(hidden + 16, hidden, batch_first=True, bidirectional=True)
        self.skip = nn.Linear(hidden, hidden * 2, bias=False)
        self.velocity = nn.ModuleList([nn.Linear(hidden * 2, len(centers)) for _ in range(2)])

    def forward(self, features: torch.Tensor, selected: torch.Tensor, p0: torch.Tensor):
        if features.ndim != 2:
            raise ValueError("features must be [T, F]")
        frames = self.cnn(features.unsqueeze(1))
        if frames.shape[0] != features.shape[0]:
            raise RuntimeError(f"CNN changed time from {features.shape[0]} to {frames.shape[0]}")
        feat = self.norm(frames)
        cond = torch.tanh(self.anchor(p0 / 100.0)).expand(feat.shape[0], -1)
        recurrent, _ = self.lstm(torch.cat([feat, cond], dim=-1).unsqueeze(0))
        recurrent = recurrent[0]
        hidden = recurrent + self.skip(feat)
        supervised = hidden[selected]
        dense = torch.stack([head(supervised) for head in self.velocity], dim=1)
        return dense, feat, recurrent, hidden


def _groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    named = list(model.named_parameters())
    keys = {
        "cnn": "cnn.",
        "norm": "norm.",
        "lstm": "lstm.",
        "skip": "skip.",
        "vel_x": "velocity.0.",
        "vel_y": "velocity.1.",
        "anchor": "anchor.",
    }
    groups = {name: [p for n, p in named if n.startswith(prefix)] for name, prefix in keys.items()}
    missing = [name for name, params in groups.items() if not params]
    if missing:
        raise RuntimeError(f"missing parameters: {missing}")
    return groups


def _grad_norm(params: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in params:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum())
    return total ** 0.5


def _delta_norm(params: list[nn.Parameter], saved: list[torch.Tensor]) -> float:
    total = 0.0
    for parameter, old in zip(params, saved):
        total += float((parameter.detach() - old).float().pow(2).sum())
    return total ** 0.5


def _neighborhood(dense: torch.Tensor, truth: torch.Tensor, centers: torch.Tensor, sigma: float):
    distance = truth[..., None] - centers
    targets = torch.softmax(-0.5 * (distance / sigma).square(), dim=-1)
    ce = -(targets * dense.log_softmax(-1)).sum(-1)
    entropy = -(targets * targets.clamp_min(1e-12).log()).sum(-1)
    return ce, entropy


def _temporal(values: torch.Tensor) -> dict[str, float]:
    """Mean channel std over time, and mean step size. A constant trace is zero."""
    series = values.detach().float()
    step = (series[1:] - series[:-1]).norm(dim=-1).mean()
    return {"std": float(series.std(dim=0).mean()), "step": float(step)}


@torch.no_grad()
def _axis_groups(dense: torch.Tensor, truth: torch.Tensor, centers: torch.Tensor, times: torch.Tensor):
    probs = dense.softmax(-1)
    pred = probs.argmax(-1)
    true_bins = (truth[..., None] - centers).abs().argmin(-1)
    pred_v = centers[pred]
    zero = int((centers - 0).abs().argmin())
    interior = (times >= INTERIOR[0]) & (times <= INTERIOR[1])
    report = {
        "vx_mae": float((pred_v[:, 0] - truth[:, 0]).abs().mean()),
        "vy_mae": float((pred_v[:, 1] - truth[:, 1]).abs().mean()),
    }
    vy_bins = true_bins[:, 1]
    for scope, mask_scope in (("all", torch.ones_like(interior)), ("interior", interior)):
        for name, mask_sign in (
            ("pos", vy_bins > zero),
            ("neg", vy_bins < zero),
            ("zero", vy_bins == zero),
        ):
            mask = mask_scope & mask_sign
            key = f"vy_{scope}_{name}"
            report[f"{key}_frames"] = int(mask.sum())
            if int(mask.sum()) == 0:
                report[f"{key}_acc"] = float("nan")
                report[f"{key}_prob"] = float("nan")
                continue
            report[f"{key}_acc"] = float((pred[:, 1][mask] == vy_bins[mask]).float().mean())
            report[f"{key}_prob"] = float(probs[:, 1][mask, vy_bins[mask]].mean())
    return report


def _predict(model, features, selected, times, p0, truth):
    model.eval()
    with torch.no_grad():
        dense, feat, recurrent, hidden = model(features, selected, p0)
        log_probs = dense.log_softmax(-1)
        indices = log_probs.argmax(-1)
        velocity = model.centers[indices]
        position = integrate_frames(velocity, times, p0)
        oracle_velocity = model.centers[(truth[..., None] - model.centers).abs().argmin(-1)]
        oracle_position = integrate_frames(oracle_velocity, times, p0)
        label_position = integrate_frames(truth, times, p0)
    prediction = {
        "probabilities": log_probs.exp().cpu().numpy(),
        "velocity": velocity.cpu().numpy(),
        "position": position.cpu().numpy(),
        "oracle_velocity": oracle_velocity.cpu().numpy(),
        "oracle_position": oracle_position.cpu().numpy(),
        "true_velocity_integral": label_position.cpu().numpy(),
    }
    variation = {
        "stft": _temporal(features),
        "cnn": _temporal(feat),
        "bilstm": _temporal(recurrent),
        "hidden": _temporal(hidden),
        "logits": _temporal(dense.reshape(dense.shape[0], -1)),
    }
    return dense, prediction, variation


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((RUN / "config.json").read_text())
    normalization = json.loads((RUN / "normalization.json").read_text())
    records = json.loads((RUN / "records.json").read_text())
    device = torch.device("cpu")
    clip = ClipStore(records, config, RUN / "cache").get(CLIP_ID)
    features, selected, times, p0, truth = to_device(clip, normalization, device)
    sigma = float(config["sigma"])

    torch.manual_seed(0)
    model = SkipFramewise(config).to(device)
    groups = _groups(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    history = []
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        saved = {name: [p.detach().clone() for p in params] for name, params in groups.items()}
        dense, _feat, _recurrent, _hidden = model(features, selected, p0)
        ce, entropy = _neighborhood(dense, truth, model.centers, sigma)
        loss_x, loss_y = ce[:, 0].mean(), ce[:, 1].mean()
        excess_x = (ce[:, 0] - entropy[:, 0]).mean()
        excess_y = (ce[:, 1] - entropy[:, 1]).mean()
        loss = loss_x + loss_y
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite neighborhood loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = {name: _grad_norm(params) for name, params in groups.items()}
        if min(grad[name] for name in ("cnn", "lstm", "skip", "vel_x", "vel_y")) <= 0:
            raise RuntimeError(f"required gradient is zero at epoch {epoch}: {grad}")
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        update = {name: _delta_norm(params, saved[name]) for name, params in groups.items()}
        if min(update[name] for name in ("cnn", "lstm", "skip", "vel_x", "vel_y")) <= 0:
            raise RuntimeError(f"required parameters did not change at epoch {epoch}: {update}")
        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "loss_x": float(loss_x.detach()),
            "loss_y": float(loss_y.detach()),
            "excess_x": float(excess_x.detach()),
            "excess_y": float(excess_y.detach()),
        }
        row.update({f"grad_{name}": value for name, value in grad.items()})
        row.update({f"update_{name}": value for name, value in update.items()})
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            eval_dense, _prediction, variation = _predict(model, features, selected, times, p0, truth)
            metrics = _axis_groups(eval_dense, truth, model.centers, times)
            row.update(metrics)
            for place, stats in variation.items():
                row[f"{place}_std"] = stats["std"]
                row[f"{place}_step"] = stats["step"]
            print(
                f"epoch {epoch:03d}  loss {row['loss']:.3f}  excess x {row['excess_x']:.3f} y {row['excess_y']:.3f}  "
                f"mae vx {metrics['vx_mae']:.3f} vy {metrics['vy_mae']:.3f}  "
                f"interior acc pos {metrics['vy_interior_pos_acc']:.3f} neg {metrics['vy_interior_neg_acc']:.3f}  "
                f"grad cnn {grad['cnn']:.3f} lstm {grad['lstm']:.3f} vy {grad['vel_y']:.3f}",
                flush=True,
            )
        history.append(row)
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best_epoch = epoch
            torch.save(
                {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "loss": best_loss,
                    "config": config,
                    "freq_pool": POOL,
                    "ctc_weight": 0.0,
                },
                OUT / "best.pt",
            )

    model.load_state_dict(torch.load(OUT / "best.pt", map_location=device, weights_only=True)["model"])
    dense, prediction, variation = _predict(model, features, selected, times, p0, truth)
    final = _axis_groups(dense, truth, model.centers, times)
    ce, entropy = _neighborhood(dense, truth, model.centers, sigma)
    save_example(CLIP_ID, clip, prediction, model.centers.detach().cpu().numpy(), OUT)
    fieldnames = list(history[-1])
    for row in history:
        for key in fieldnames:
            row.setdefault(key, "")
    with (OUT / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    tracked = (
        final["vy_interior_pos_acc"] >= 0.8
        and final["vy_interior_neg_acc"] >= 0.8
        and final["vy_interior_pos_frames"] > 20
        and final["vy_interior_neg_frames"] > 20
    )
    summary = {
        "clip_id": CLIP_ID,
        "changes": {
            "freq_pool": POOL,
            "layer_norm": "LayerNorm on the CNN vector before the BiLSTM and the skip",
            "skip": "h_t = BiLSTM(f_t, p0)_t + W_skip f_t, W_skip is Linear(hidden, 2*hidden) without bias",
            "ctc": 0.0,
            "weight_decay": 0.0,
            "dropout": None,
            "grad_clip": CLIP_NORM,
            "decoder": "argmax",
            "time_stride": 1,
            "rate_hz": config["sample_rate"] / config["hop"],
            "bin_width_mps": config["bin_width"],
        },
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "excess_ce_x": float((ce[:, 0] - entropy[:, 0]).mean()),
        "excess_ce_y": float((ce[:, 1] - entropy[:, 1]).mean()),
        "entropy_x": float(entropy[:, 0].mean()),
        "entropy_y": float(entropy[:, 1].mean()),
        "feature_variation": variation,
        "metrics": final,
        "interior_seconds": list(INTERIOR),
        "interior_turns_tracked": tracked,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
