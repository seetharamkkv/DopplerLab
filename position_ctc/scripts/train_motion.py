"""Train framewise vx and vy classifiers and integrate them from the start.

The network sees the clip's true initial position. Framewise cross-entropy,
a neighborhood penalty, a separate CTC loss per axis, and a differentiable
trapezoid of the expected velocity share that same anchor. Viterbi is used
only at evaluation. The anchored Viterbi integral is the only trajectory.

A handful of turning clips are fit with framewise supervision alone before
the full run. If that memorization fails, the full run does not start.

Run from the DopplerLab root:

    position_ctc/.venv/bin/python position_ctc/scripts/train_motion.py \\
        --data /path/to/audio_clips \\
        --out position_ctc/outputs/motion
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.alphabet import VEL_STEP_MPS  # noqa: E402
from position_ctc.bins import CENTERS_MPS, trapezoid_velocity_gap, velocity_bin_index  # noqa: E402
from position_ctc.data import (  # noqa: E402
    CROP_FRAMES,
    augment_features,
    crop_pair,
    discover_clips,
    load_arrays,
    split_by_vehicle,
    standardize_mag,
)
from position_ctc.features import DATA_HOP, DATA_SR  # noqa: E402
from position_ctc.metrics import score_path  # noqa: E402
from position_ctc.motion_loss import LossWeights, MotionObjective  # noqa: E402
from position_ctc.motion_model import MotionNet, initial_position  # noqa: E402
from position_ctc.plots import error_over_time_figure, probability_heatmap, trajectory_figure  # noqa: E402
from position_ctc.reconstruct import predict_trajectory  # noqa: E402
from position_ctc.transitions import (  # noqa: E402
    DEFAULT_JUMP_MPS,
    DEFAULT_LAMBDA,
    a_max_for_jump,
    forbidden_bin_fraction,
    jump_limit_mps,
    transition_matrix,
)

DT = DATA_HOP / float(DATA_SR)
TURN_MPS = 2.0


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare(clips, indices: list[int]) -> list[dict]:
    rows = []
    for i in indices:
        clip = clips[i]
        feat, state = load_arrays(clip)
        rows.append(
            {
                "clip_id": clip.clip_id,
                "vehicle": clip.vehicle,
                "feat": feat,
                "state": state.astype(np.float32),
                "sample_dir": clip.sample_dir,
            }
        )
        if len(rows) % 100 == 0:
            print(f"  features {len(rows)}/{len(indices)}", flush=True)
    return rows


def _draw(row: dict, rng: np.random.Generator | None) -> dict | None:
    feat, state = row["feat"], row["state"]
    if rng is not None:
        feat = augment_features(feat, rng)
        if feat.shape[1] > CROP_FRAMES and rng.random() < 0.25:
            feat, state = crop_pair(feat, state, rng, CROP_FRAMES)
    feat = standardize_mag(feat)
    if feat.shape[1] < 8 or state.shape[0] < 8:
        return None
    n = min(feat.shape[1], state.shape[0])
    return {
        "feat": torch.from_numpy(np.ascontiguousarray(feat[:, :n])),
        "state": torch.from_numpy(np.ascontiguousarray(state[:n])),
        "clip_id": row["clip_id"],
    }


def _collate(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(int(item["feat"].shape[-1]) for item in items)
    feat = torch.zeros(len(items), items[0]["feat"].shape[0], length)
    state = torch.zeros(len(items), length, 4)
    lengths = torch.zeros(len(items), dtype=torch.long)
    for i, item in enumerate(items):
        n = int(item["feat"].shape[-1])
        feat[i, :, :n] = item["feat"]
        state[i, :n] = item["state"]
        lengths[i] = n
    return feat, state, lengths


def _turn_mask(velocity: np.ndarray, thresh: float = TURN_MPS) -> np.ndarray:
    n0 = max(int(velocity.shape[0]) // 5, 1)
    base = float(np.mean(velocity[:n0]))
    return np.abs(velocity - base) >= thresh


def _mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = np.abs(pred - truth)
    if mask is not None:
        if not np.any(mask):
            return float("nan")
        err = err[mask]
    return float(np.mean(err))


def _step_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.shape[0] < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(pred, axis=0) - np.diff(truth, axis=0))))


def _path_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    delta = pred - truth
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _bin_histogram(velocity: np.ndarray) -> dict[str, float]:
    counts = np.bincount(velocity_bin_index(velocity), minlength=CENTERS_MPS.size).astype(np.float64)
    total = max(float(counts.sum()), 1.0)
    order = np.argsort(counts)[::-1]
    return {f"{CENTERS_MPS[i]:+.0f}": float(counts[i] / total) for i in order[:5]}


def _label_audit(rows: list[dict], psi: np.ndarray, jump_mps: float) -> dict:
    gaps = [trapezoid_velocity_gap(row["state"], DT) for row in rows]
    vx = np.concatenate([row["state"][:, 1] for row in rows])
    vy = np.concatenate([row["state"][:, 3] for row in rows])
    matched = 0
    single_vx = 0
    single_vy = 0
    both_turn = 0
    forbid_x = []
    forbid_y = []
    for row in rows:
        state = row["state"]
        raw = np.load(row["sample_dir"] / "metadata" / "canonical_state_frames.npy").astype(np.float32)
        speed = np.hypot(raw[:, 1], raw[:, 3])
        moving = np.flatnonzero(speed > 0.5)
        if moving.size:
            start = int(moving[0])
            window = raw[start : start + state.shape[0]]
            if window.shape == state.shape and np.allclose(window, state, atol=1e-4):
                matched += 1
        if np.unique(np.round(state[:, 1], 3)).size == 1:
            single_vx += 1
        if np.unique(np.round(state[:, 3], 3)).size == 1:
            single_vy += 1
        if _turn_mask(state[:, 1]).mean() > 0.05 and _turn_mask(state[:, 3]).mean() > 0.05:
            both_turn += 1
        forbid_x.append(forbidden_bin_fraction(state[:, 1], psi))
        forbid_y.append(forbidden_bin_fraction(state[:, 3], psi))
    return {
        "clips": len(rows),
        "frames": int(vx.size),
        "targets_match_canonical_slice": matched,
        "single_value_vx_clips": single_vx,
        "single_value_vy_clips": single_vy,
        "clips_with_both_axes_turning": both_turn,
        "velocity_gap_median_mps": float(np.median(gaps)),
        "vx_bin_frequency": _bin_histogram(vx),
        "vy_bin_frequency": _bin_histogram(vy),
        "forbidden_vx_steps": float(np.mean(forbid_x)),
        "forbidden_vy_steps": float(np.mean(forbid_y)),
        "jump_limit_mps": float(jump_mps),
        "temporal_pool": "two stride-2 pools, then linear upsample to the audio frame",
        "frame_dt_s": DT,
        "corner_policy": (
            "Steps whose bins are illegal under the graded mask are dropped from "
            "framewise and CTC targets. CTC holds the last legal bin. The trajectory "
            "loss still follows the position label and does not require an illegal bin. "
            "A three-bin limit allows a 6 m/s jump per frame and does not make the path smooth."
        ),
    }


def _score_clip(traj, state: np.ndarray, psi: np.ndarray) -> dict[str, float]:
    n = min(traj.vx.shape[0], state.shape[0])
    gt = state[:n]
    xy = gt[:, [0, 2]]
    turn_x = _turn_mask(gt[:, 1])
    turn_y = _turn_mask(gt[:, 3])
    scored = score_path(traj.xy[:n], xy)
    anchor = float(np.hypot(traj.xy[0, 0] - gt[0, 0], traj.xy[0, 1] - gt[0, 2]))
    return {
        "anchor_error_m": anchor,
        "vx_mae": _mae(traj.vx[:n], gt[:, 1]),
        "vy_mae": _mae(traj.vy[:n], gt[:, 3]),
        "vx_argmax_mae": _mae(traj.vx_argmax[:n], gt[:, 1]),
        "vy_argmax_mae": _mae(traj.vy_argmax[:n], gt[:, 3]),
        "vx_turn_mae": _mae(traj.vx[:n], gt[:, 1], turn_x),
        "vy_turn_mae": _mae(traj.vy[:n], gt[:, 3], turn_y),
        "vx_turn_argmax_mae": _mae(traj.vx_argmax[:n], gt[:, 1], turn_x),
        "vy_turn_argmax_mae": _mae(traj.vy_argmax[:n], gt[:, 3], turn_y),
        "dvx_mae": _step_mae(traj.vx[:n], gt[:, 1]),
        "dvy_mae": _step_mae(traj.vy[:n], gt[:, 3]),
        "dpx_mae": _step_mae(traj.xy[:n, 0], xy[:, 0]),
        "dpy_mae": _step_mae(traj.xy[:n, 1], xy[:, 1]),
        "integrated_rmse_m": _path_rmse(traj.xy[:n], xy),
        "argmax_rmse_m": _path_rmse(traj.xy_argmax[:n], xy),
        "argmax_agreement_x": float(np.mean(np.abs(traj.vx[:n] - traj.vx_argmax[:n]) < 1e-6)),
        "argmax_agreement_y": float(np.mean(np.abs(traj.vy[:n] - traj.vy_argmax[:n]) < 1e-6)),
        "decoded_forbidden_x": forbidden_bin_fraction(traj.vx[:n], psi),
        "decoded_forbidden_y": forbidden_bin_fraction(traj.vy[:n], psi),
        "gt_forbidden_x": forbidden_bin_fraction(gt[:, 1], psi),
        "gt_forbidden_y": forbidden_bin_fraction(gt[:, 3], psi),
        "turn_fraction_x": float(np.mean(turn_x)),
        "turn_fraction_y": float(np.mean(turn_y)),
        "orbit_rms_m": scored["orbit_rms_m"],
        "range_mae_m": scored["range_mae_m"],
        "cpa_abs_m": scored["cpa_abs_m"],
        **_dominant_groups(np.exp(traj.log_px[:n]), gt[:, 1], "vx"),
        **_dominant_groups(np.exp(traj.log_py[:n]), gt[:, 3], "vy"),
    }


def _dominant_groups(probs: np.ndarray, truth: np.ndarray, axis: str) -> dict[str, float]:
    """Per-clip dominant bin versus every other bin. Uses the argmax, not Viterbi."""
    true_bins = velocity_bin_index(truth)
    pred_bins = np.argmax(probs, axis=1)
    dominant = int(np.bincount(true_bins, minlength=CENTERS_MPS.size).argmax())
    minority = true_bins != dominant
    dominant_frames = ~minority
    pred_mps = CENTERS_MPS[pred_bins]

    def _acc(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(pred_bins[mask] == true_bins[mask]))

    def _mae(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.abs(pred_mps[mask] - truth[mask])))

    if np.any(minority):
        correct = float(probs[np.flatnonzero(minority), true_bins[minority]].mean())
    else:
        correct = float("nan")
    return {
        f"{axis}_dominant_acc": _acc(dominant_frames),
        f"{axis}_minority_acc": _acc(minority),
        f"{axis}_dominant_mae": _mae(dominant_frames),
        f"{axis}_minority_mae": _mae(minority),
        f"{axis}_minority_prob": correct,
    }


SCORE_KEYS = (
    "anchor_error_m",
    "vx_mae",
    "vy_mae",
    "vx_argmax_mae",
    "vy_argmax_mae",
    "vx_turn_mae",
    "vy_turn_mae",
    "vx_turn_argmax_mae",
    "vy_turn_argmax_mae",
    "dvx_mae",
    "dvy_mae",
    "dpx_mae",
    "dpy_mae",
    "integrated_rmse_m",
    "argmax_rmse_m",
    "argmax_agreement_x",
    "argmax_agreement_y",
    "decoded_forbidden_x",
    "decoded_forbidden_y",
    "gt_forbidden_x",
    "gt_forbidden_y",
    "orbit_rms_m",
    "range_mae_m",
    "cpa_abs_m",
    "vx_dominant_acc",
    "vx_minority_acc",
    "vx_dominant_mae",
    "vx_minority_mae",
    "vx_minority_prob",
    "vy_dominant_acc",
    "vy_minority_acc",
    "vy_dominant_mae",
    "vy_minority_mae",
    "vy_minority_prob",
)


@torch.no_grad()
def _evaluate(model, rows: list[dict], device, psi) -> list[dict]:
    model.eval()
    scored = []
    for row in rows:
        feat = standardize_mag(row["feat"])
        traj = predict_trajectory(model, feat, row["state"], DT, psi, device)
        item = _score_clip(traj, row["state"], psi)
        item["clip_id"] = row["clip_id"]
        item["vehicle"] = row["vehicle"]
        item["traj"] = traj
        item["state"] = row["state"]
        scored.append(item)
    return scored


def _mean_scores(scored: list[dict]) -> dict[str, float]:
    means = {}
    for key in SCORE_KEYS:
        values = np.array([item[key] for item in scored], dtype=np.float64)
        means[key] = float(np.nanmean(values))
    return means


def _time_curve(scored: list[dict], points: int = 21) -> dict[str, list[float]]:
    grid = np.linspace(0.0, 1.0, points)
    viterbi = []
    argmax = []
    for item in scored:
        n = min(item["traj"].xy.shape[0], item["state"].shape[0])
        truth = item["state"][:n, [0, 2]]
        err_v = np.linalg.norm(item["traj"].xy[:n] - truth, axis=1)
        err_a = np.linalg.norm(item["traj"].xy_argmax[:n] - truth, axis=1)
        ticks = np.linspace(0.0, 1.0, n)
        viterbi.append(np.interp(grid, ticks, err_v))
        argmax.append(np.interp(grid, ticks, err_a))
    return {
        "fraction": grid.tolist(),
        "viterbi_m": np.mean(viterbi, axis=0).tolist(),
        "argmax_m": np.mean(argmax, axis=0).tolist(),
    }


def _plot_clip(item: dict, path: Path) -> None:
    state = item["state"]
    traj = item["traj"]
    n = min(traj.vx.shape[0], state.shape[0])
    trajectory_figure(
        np.arange(n) * DT,
        state[:n, 1],
        state[:n, 3],
        state[:n, [0, 2]],
        traj.vx[:n],
        traj.vy[:n],
        traj.xy[:n],
        path,
        title=(
            f"{item['clip_id']}  {item['vehicle']}  "
            f"anchored Viterbi RMSE {item['integrated_rmse_m']:.2f} m"
        ),
        vx_argmax=traj.vx_argmax[:n],
        vy_argmax=traj.vy_argmax[:n],
        xy_argmax=traj.xy_argmax[:n],
    )


def _plot_heatmaps(item: dict, directory: Path) -> None:
    state = item["state"]
    traj = item["traj"]
    n = min(traj.vx.shape[0], state.shape[0])
    time = np.arange(n) * DT
    probability_heatmap(
        time,
        traj.log_px[:n],
        state[:n, 1],
        traj.vx_argmax[:n],
        traj.vx[:n],
        directory / f"{item['clip_id']}_vx.png",
        title=f"{item['clip_id']}  vx probability",
    )
    probability_heatmap(
        time,
        traj.log_py[:n],
        state[:n, 3],
        traj.vy_argmax[:n],
        traj.vy[:n],
        directory / f"{item['clip_id']}_vy.png",
        title=f"{item['clip_id']}  vy probability",
    )


def _plot_cases(scored: list[dict], directory: Path) -> list[dict]:
    """Best, median, and worst integrated path for each test vehicle."""
    directory.mkdir(parents=True, exist_ok=True)
    by_vehicle: dict[str, list[dict]] = {}
    for item in scored:
        by_vehicle.setdefault(item["vehicle"], []).append(item)
    chosen = []
    for vehicle in sorted(by_vehicle):
        rows = sorted(by_vehicle[vehicle], key=lambda item: item["integrated_rmse_m"])
        picks = (("best", rows[0]), ("median", rows[len(rows) // 2]), ("worst", rows[-1]))
        for rank, item in picks:
            path = directory / f"{item['clip_id']}.png"
            _plot_clip(item, path)
            chosen.append({"clip_id": item["clip_id"], "vehicle": vehicle, "rank": rank, "plot": str(path), **{key: item[key] for key in SCORE_KEYS}})
    turning = sorted(scored, key=lambda item: item["turn_fraction_y"], reverse=True)[:4]
    heat = directory / "heatmaps"
    for item in turning:
        _plot_clip(item, heat / f"{item['clip_id']}.png")
        _plot_heatmaps(item, heat)
    return chosen


def _batches(rows: list[dict], rng: np.random.Generator, batch_size: int, train: bool):
    order = list(range(len(rows)))
    if train:
        rng.shuffle(order)
    bucket: list[dict] = []
    for index in order:
        item = _draw(rows[index], rng if train else None)
        if item is None:
            continue
        bucket.append(item)
        if len(bucket) == batch_size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def _scheduled_lr(epoch: int, total: int, base: float, warmup: int = 3) -> float:
    """Linear warmup, then cosine decay to 2% of the base rate."""
    if epoch <= warmup:
        return base * epoch / max(warmup, 1)
    span = max(total - warmup, 1)
    progress = (epoch - warmup) / span
    floor = base * 0.02
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))


def _forward(model, feat, state, lengths):
    origin = initial_position(state).to(feat.device)
    return model(feat, lengths, origin)


@torch.no_grad()
def _validation_loss(model, objective, rows, rng, batch_size, device) -> float:
    model.eval()
    total = 0.0
    frames = 0
    for bucket in _batches(rows, rng, batch_size, train=False):
        feat, state, lengths = _collate(bucket)
        pred = _forward(model, feat.to(device), state.to(device), lengths.to(device))
        _, parts = objective(pred, state.to(device), lengths.to(device), DT)
        n = int(lengths.sum())
        total += parts["total"] * n
        frames += n
    return total / max(frames, 1)


def _select_turning_clips(rows: list[dict], count: int = 4) -> list[dict]:
    """Clips whose legal steps change both components on a real share of frames."""
    ranked = []
    for row in rows:
        state = row["state"]
        vx, vy = state[:, 1], state[:, 3]
        if state.shape[0] < 32:
            continue
        if np.max(np.abs(np.diff(velocity_bin_index(vx)))) > 3:
            continue
        if np.max(np.abs(np.diff(velocity_bin_index(vy)))) > 3:
            continue
        fx = float(np.mean(_turn_mask(vx)))
        fy = float(np.mean(_turn_mask(vy)))
        if fx < 0.12 or fy < 0.12:
            continue
        if np.ptp(vx) < 4.0 or np.ptp(vy) < 4.0:
            continue
        ranked.append((min(fx, fy), row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in ranked[:count]]


def _axis_turn_accuracy(pred_mps: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    mask = _turn_mask(truth)
    if int(mask.sum()) < 20:
        return float("nan"), int(mask.sum())
    hit = velocity_bin_index(pred_mps) == velocity_bin_index(truth)
    return float(np.mean(hit[mask])), int(mask.sum())


def _overfit_gate(rows: list[dict], device, psi, out_dir: Path, epochs: int = 250) -> dict:
    """Memorize a few turning clips with framewise losses only.

    CTC and the trajectory loss stay at weight 0. Dropout and weight decay
    are off. Success is an argmax that follows the turn, before Viterbi.
    """
    chosen = _select_turning_clips(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(chosen) < 2:
        raise SystemExit("not enough legal turning clips to run the memorization check")
    print(
        "overfit clips "
        + ", ".join(
            f"{row['clip_id']} vx-turn {_turn_mask(row['state'][:, 1]).mean():.2f} "
            f"vy-turn {_turn_mask(row['state'][:, 3]).mean():.2f}"
            for row in chosen
        ),
        flush=True,
    )
    model = MotionNet(dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    items = []
    for row in chosen:
        drawn = _draw(row, None)
        if drawn is None:
            continue
        items.append(drawn)
    feat, state, lengths = _collate(items)
    feat, state, lengths = feat.to(device), state.to(device), lengths.to(device)
    targets_x = torch.from_numpy(velocity_bin_index(state[:, :, 1].detach().cpu().numpy())).to(device)
    targets_y = torch.from_numpy(velocity_bin_index(state[:, :, 3].detach().cpu().numpy())).to(device)
    valid = torch.arange(state.shape[1], device=device)[None, :] < lengths[:, None]
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        pred = _forward(model, feat, state, lengths)
        ce_x = F.cross_entropy(pred["vel_x"].reshape(-1, model.n_vel), targets_x.reshape(-1), reduction="none")
        ce_y = F.cross_entropy(pred["vel_y"].reshape(-1, model.n_vel), targets_y.reshape(-1), reduction="none")
        ce = ((ce_x + ce_y).view_as(valid) * valid).sum() / valid.sum().clamp(min=1) / 2
        opt.zero_grad(set_to_none=True)
        ce.backward()
        if epoch == 1:
            conv = float(model.conv[0].weight.grad.detach().norm())
            vel = float(model.vel_y.weight.grad.detach().norm())
            temporal = float(pred["vel_y"].detach().std(dim=1).mean())
            print(
                f"overfit epoch 1 grad  conv {conv:.4f}  vel_y {vel:.4f}  "
                f"logit temporal std {temporal:.4f}  "
                f"enc {int(pred['enc_lengths'].max())} audio {int(lengths.max())}",
                flush=True,
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if epoch in (1, 30, 80, 150, 300) or epoch == epochs:
            acc = _overfit_accuracy(model, items, device, psi)
            history.append({"epoch": epoch, "ce": float(ce.detach()), "path_weight": 0.0, "ctc_weight": 0.0, **acc})
            print(
                f"overfit epoch {epoch:03d}  ce {float(ce.detach()):.3f}  "
                f"turn acc vx {acc['turn_acc_x']:.3f} vy {acc['turn_acc_y']:.3f}  "
                f"all acc vx {acc['acc_x']:.3f} vy {acc['acc_y']:.3f}",
                flush=True,
            )
            if (
                epoch >= 150
                and acc["turn_acc_x"] >= 0.85
                and acc["turn_acc_y"] >= 0.85
                and acc["passed_clips"] == acc["clips"]
            ):
                break
    model.eval()
    scored = _evaluate(model, chosen, device, psi)
    for item in scored:
        _plot_clip(item, out_dir / f"{item['clip_id']}.png")
        _plot_heatmaps(item, out_dir)
    acc = _overfit_accuracy(model, items, device, psi)
    passed = bool(acc["turn_acc_x"] >= 0.8 and acc["turn_acc_y"] >= 0.8 and acc["passed_clips"] == acc["clips"])
    report = {"passed": passed, "clips": [row["clip_id"] for row in chosen], "history": history, "final": acc}
    (out_dir / "overfit.json").write_text(json.dumps(report, indent=2))
    print(
        f"overfit {'passed' if passed else 'FAILED'}  "
        f"turn acc vx {acc['turn_acc_x']:.3f} vy {acc['turn_acc_y']:.3f}  "
        f"clips {acc['passed_clips']}/{acc['clips']}",
        flush=True,
    )
    return report


@torch.no_grad()
def _overfit_accuracy(model, items: list[dict], device, psi) -> dict[str, float]:
    model.eval()
    turn_x, turn_y, all_x, all_y = [], [], [], []
    passed = 0
    per_clip = []
    for item in items:
        feat = item["feat"].numpy()
        state = item["state"].numpy()
        traj = predict_trajectory(model, feat, state, DT, psi, device)
        n = min(traj.vx_argmax.shape[0], state.shape[0])
        ax, nx = _axis_turn_accuracy(traj.vx_argmax[:n], state[:n, 1])
        ay, ny = _axis_turn_accuracy(traj.vy_argmax[:n], state[:n, 3])
        hit_x = velocity_bin_index(traj.vx_argmax[:n]) == velocity_bin_index(state[:n, 1])
        hit_y = velocity_bin_index(traj.vy_argmax[:n]) == velocity_bin_index(state[:n, 3])
        all_x.append(float(np.mean(hit_x)))
        all_y.append(float(np.mean(hit_y)))
        if nx >= 20:
            turn_x.append(ax)
        if ny >= 20:
            turn_y.append(ay)
        clip_ok = (nx < 20 or ax >= 0.8) and (ny < 20 or ay >= 0.8) and (nx >= 20 or ny >= 20)
        passed += int(clip_ok)
        per_clip.append({"clip_id": item["clip_id"], "turn_acc_x": ax, "turn_acc_y": ay, "turn_frames_x": nx, "turn_frames_y": ny})
    model.train()
    return {
        "turn_acc_x": float(np.mean(turn_x)) if turn_x else 0.0,
        "turn_acc_y": float(np.mean(turn_y)) if turn_y else 0.0,
        "acc_x": float(np.mean(all_x)) if all_x else 0.0,
        "acc_y": float(np.mean(all_y)) if all_y else 0.0,
        "passed_clips": passed,
        "clips": len(items),
        "per_clip": per_clip,
    }


def _capped_frequency_weight(rows: list[dict], column: int, cap: float) -> torch.Tensor:
    """Inverse frequency of bins that occur, capped at ``cap`` times their median.

    Bins with no training frames get weight 0. Counts come from the training
    clips only. This is separate from the per-clip dominant/non-dominant split
    and is not combined with it.
    """
    values = np.concatenate([row["state"][:, column] for row in rows])
    counts = np.bincount(velocity_bin_index(values), minlength=CENTERS_MPS.size).astype(np.float64)
    present = counts > 0
    weights = np.zeros(CENTERS_MPS.size, dtype=np.float64)
    inverse = counts.sum() / counts[present]
    inverse = np.minimum(inverse, float(cap) * float(np.median(inverse)))
    inverse = inverse / max(float(inverse.mean()), 1e-8)
    weights[present] = inverse
    return torch.tensor(weights, dtype=torch.float32)


def _decode_note(metrics: dict[str, float]) -> str:
    agree = min(metrics["argmax_agreement_x"], metrics["argmax_agreement_y"])
    if agree > 0.98:
        return (
            "Viterbi matches the framewise argmax, so decoding is not flattening a different raw path."
        )
    turn_raw = metrics["vx_turn_argmax_mae"] + metrics["vy_turn_argmax_mae"]
    turn_dec = metrics["vx_turn_mae"] + metrics["vy_turn_mae"]
    if turn_dec > turn_raw * 1.25 and metrics["integrated_rmse_m"] + 0.25 < metrics["argmax_rmse_m"]:
        return "Viterbi lowers path error while increasing turning-region velocity error."
    if turn_dec <= turn_raw * 1.05 and metrics["integrated_rmse_m"] <= metrics["argmax_rmse_m"] * 1.05:
        return "Viterbi matches or improves both the turning-region velocity and the path."
    return "Viterbi and the framewise argmax differ. Compare the per-axis and turning-region errors."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "position_ctc" / "outputs" / "motion")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Use the first N clips. 0 means all.")
    parser.add_argument("--jump-mps", type=float, default=DEFAULT_JUMP_MPS)
    parser.add_argument("--transition-lambda", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--lambda-dist", type=float, default=1.0)
    parser.add_argument("--lambda-ctc", type=float, default=0.25)
    parser.add_argument("--lambda-path", type=float, default=1.0)
    parser.add_argument("--frame-balance", choices=("mean", "dominant", "frequency"), default="mean")
    parser.add_argument("--frequency-cap", type=float, default=10.0)
    parser.add_argument("--overfit-epochs", type=int, default=300)
    parser.add_argument("--skip-overfit", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = _device()
    args.out.mkdir(parents=True, exist_ok=True)

    clips = discover_clips(args.data)
    if args.limit > 0:
        clips = clips[: args.limit]
    splits = split_by_vehicle(clips, seed=args.seed)
    print(f"device {device}  clips {len(clips)}  dt {DT:.6f} s", flush=True)
    print("splits " + ", ".join(f"{name} {len(idxs)}" for name, idxs in splits.items()), flush=True)
    rows = {name: _prepare(clips, idxs) for name, idxs in splits.items()}
    if not rows["train"] or not rows["val"] or not rows["test"]:
        raise SystemExit("split produced an empty train, val, or test set")
    a_max = a_max_for_jump(args.jump_mps, DT)
    limit = jump_limit_mps(DT, a_max)
    psi = transition_matrix(DT, a_max, lam=args.transition_lambda)
    audit = _label_audit(rows["train"], psi, limit)
    print(
        f"jump limit {limit:.2f} m/s per frame ({limit / VEL_STEP_MPS:.0f} bins, "
        f"{a_max:.0f} m/s²). This prices jumps; it does not force a smooth path.",
        flush=True,
    )
    print(
        f"labels  canonical slices {audit['targets_match_canonical_slice']}/{audit['clips']}  "
        f"single-value vx {audit['single_value_vx_clips']} vy {audit['single_value_vy_clips']}  "
        f"both axes turn {audit['clips_with_both_axes_turning']}",
        flush=True,
    )
    print(
        f"bins vx {audit['vx_bin_frequency']}  vy {audit['vy_bin_frequency']}",
        flush=True,
    )
    print(
        f"forbidden GT steps  vx {audit['forbidden_vx_steps']:.4%}  vy {audit['forbidden_vy_steps']:.4%}  "
        f"median |dp/dt - v| {audit['velocity_gap_median_mps']:.4f} m/s",
        flush=True,
    )
    print(f"clock {DT * 1000:.2f} ms/frame  encoder time is T/4 then upsampled", flush=True)

    overfit = {"passed": True, "skipped": True}
    if not args.skip_overfit:
        overfit = _overfit_gate(rows["train"], device, psi, args.out / "overfit", epochs=args.overfit_epochs)
        if not overfit["passed"]:
            raise SystemExit("framewise memorization of turning clips failed; full training was not started")

    weights = LossWeights(dist=args.lambda_dist, ctc=args.lambda_ctc, path=args.lambda_path)
    model = MotionNet().to(device)
    class_x = class_y = None
    if args.frame_balance == "frequency":
        class_x = _capped_frequency_weight(rows["train"], 1, args.frequency_cap)
        class_y = _capped_frequency_weight(rows["train"], 3, args.frequency_cap)
        print(
            f"frequency cap {args.frequency_cap:g}  "
            f"vx weights {class_x.tolist()}  vy weights {class_y.tolist()}",
            flush=True,
        )
    objective = MotionObjective(
        weights,
        jump_mps=args.jump_mps,
        balance=args.frame_balance,
        class_weight_x=class_x,
        class_weight_y=class_y,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = None
    best_rmse = None
    best_epoch = 0
    stale = 0
    history = []
    ckpt_path = args.out / "motion.pt"
    report_path = args.out / "metrics.json"

    for epoch in range(1, args.epochs + 1):
        lr = _scheduled_lr(epoch, args.epochs, args.lr)
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        totals: dict[str, list[float]] = {}
        for bucket in _batches(rows["train"], rng, args.batch_size, train=True):
            feat, state, lengths = _collate(bucket)
            feat, state, lengths = feat.to(device), state.to(device), lengths.to(device)
            pred = _forward(model, feat, state, lengths)
            if epoch == 1 and not totals:
                print(
                    f"batch shapes feat {tuple(feat.shape)} vel_x {tuple(pred['vel_x'].shape)} "
                    f"ctc_x {tuple(pred['ctc_x'].shape)} enc_len {int(pred['enc_lengths'][0])} "
                    f"audio_len {int(lengths[0])}",
                    flush=True,
                )
            loss, parts = objective(pred, state, lengths, DT)
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            for key, value in parts.items():
                totals.setdefault(key, []).append(value)
        train_loss = float(np.mean(totals["total"])) if totals else float("nan")
        val_loss = _validation_loss(model, objective, rows["val"], rng, args.batch_size, device)
        scored = _evaluate(model, rows["val"], device, psi)
        metrics = _mean_scores(scored)
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ce": float(np.mean(totals.get("ce", [math.nan]))),
            "train_ctc": float(np.mean(totals.get("ctc", [math.nan]))),
            "train_path": float(np.mean(totals.get("path", [math.nan]))),
            **metrics,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  lr {lr:.2e}  loss {train_loss:.3f}  val {val_loss:.3f}  "
            f"vx {metrics['vx_mae']:.2f}  vy {metrics['vy_mae']:.2f}  "
            f"vx minority {metrics['vx_minority_acc']:.2f}  vy minority {metrics['vy_minority_acc']:.2f}  "
            f"vy p {metrics['vy_minority_prob']:.2f}  "
            f"anchor {metrics['anchor_error_m']:.3e} m  "
            f"path {metrics['integrated_rmse_m']:.2f} m  "
            f"argmax path {metrics['argmax_rmse_m']:.2f} m",
            flush=True,
        )
        improved_loss = best_val is None or val_loss < best_val * 0.998
        if improved_loss:
            best_val = val_loss
            best_rmse = metrics["integrated_rmse_m"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "dt": DT,
                    "jump_mps": args.jump_mps,
                    "a_max_mps2": a_max,
                    "transition_lambda": args.transition_lambda,
                    "epoch": epoch,
                    "lr": lr,
                    "val_loss": val_loss,
                    "val": metrics,
                    "origin_conditioned": True,
                },
                ckpt_path,
            )
        else:
            stale += 1
        report_path.write_text(
            json.dumps({"best_epoch": best_epoch, "best_val_loss": best_val, "history": history}, indent=2)
        )
        if stale >= args.patience:
            print(f"stop at epoch {epoch}; validation loss stalled. best epoch {best_epoch}", flush=True)
            break

    if ckpt_path.is_file() and args.epochs > 0:
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        best_epoch = int(saved.get("epoch", best_epoch))
    test = _evaluate(model, rows["test"], device, psi)
    test_metrics = _mean_scores(test)
    curve = _time_curve(test)
    figure_dir = args.out / "inference"
    cases = _plot_cases(test, figure_dir)
    median = sorted(test, key=lambda item: item["integrated_rmse_m"])[len(test) // 2]
    _plot_clip(median, args.out / "trajectory.png")
    error_over_time_figure(
        np.array(curve["fraction"]),
        np.array(curve["viterbi_m"]),
        np.array(curve["argmax_m"]),
        args.out / "error_over_time.png",
        title="test position error",
    )
    note = _decode_note(test_metrics)
    report = {
        "dt_s": DT,
        "coordinate_frame": "canonical meters, microphone at the origin, anchor = state[0] (x, y)",
        "frame_balance": args.frame_balance,
        "output": "trapezoid integral of per-axis Viterbi bin centers from the supplied initial position",
        "jump_mps": limit,
        "a_max_mps2": a_max,
        "transition_lambda": args.transition_lambda,
        "label_audit": audit,
        "overfit": overfit,
        "decode_note": note,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "history": history,
        "test": test_metrics,
        "error_over_time": curve,
        "cases": cases,
        "plot": str(args.out / "trajectory.png"),
        "checkpoint": str(ckpt_path),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(note, flush=True)
    print(
        f"test  anchor {test_metrics['anchor_error_m']:.3e} m  "
        f"vx {test_metrics['vx_mae']:.2f} (argmax {test_metrics['vx_argmax_mae']:.2f})  "
        f"vy {test_metrics['vy_mae']:.2f} (argmax {test_metrics['vy_argmax_mae']:.2f})  "
        f"turn vx {test_metrics['vx_turn_mae']:.2f} (argmax {test_metrics['vx_turn_argmax_mae']:.2f})  "
        f"turn vy {test_metrics['vy_turn_mae']:.2f} (argmax {test_metrics['vy_turn_argmax_mae']:.2f})  "
        f"path {test_metrics['integrated_rmse_m']:.2f} m  "
        f"argmax path {test_metrics['argmax_rmse_m']:.2f} m",
        flush=True,
    )
    print(
        f"forbidden decoded steps vx {test_metrics['decoded_forbidden_x']:.4%} "
        f"vy {test_metrics['decoded_forbidden_y']:.4%}  "
        f"GT vx {test_metrics['gt_forbidden_x']:.4%} vy {test_metrics['gt_forbidden_y']:.4%}",
        flush=True,
    )
    print(f"wrote {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
