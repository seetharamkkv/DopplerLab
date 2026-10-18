"""Overlay two full-dataset checkpoints on the same held-out clips.

The previous weights are the epoch-27 fine-tune that produced the Renault
Captur figure. The current weights are the epoch-35 origin-conditioned model
that produced the Kia Sportage figure. Both are full-dataset runs. The
single-clip memorization was not saved and is not evaluated here.

Preprocessing, the integration anchor, and the Viterbi transition matrix are
shared. The epoch-27 network has no initial-position input, so its origin
layer is held at zero and does not move the features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.bins import CENTERS_MPS, velocity_bin_index  # noqa: E402
from position_ctc.data import discover_clips, load_arrays, split_by_vehicle, standardize_mag  # noqa: E402
from position_ctc.features import DATA_HOP, DATA_SR  # noqa: E402
from position_ctc.integrate import integrate_trapezoid  # noqa: E402
from position_ctc.motion_model import MotionNet  # noqa: E402
from position_ctc.reconstruct import reconstruct  # noqa: E402
from position_ctc.paths import resolve_audio_clips  # noqa: E402
from position_ctc.transitions import a_max_for_jump, transition_matrix  # noqa: E402

OUT = ROOT / "position_ctc" / "outputs" / "motion" / "compare"
CKPT_DIR = ROOT / "position_ctc" / "outputs" / "motion"
DT = DATA_HOP / float(DATA_SR)
JUMP_MPS = 6.0
LAMBDA = 1.0
TURN_MPS = 2.0
PLOT_IDS = (
    "sample_0000172",
    "sample_0000498",
    "sample_0000199",
    "sample_0000225",
    "sample_0000241",
    "sample_0000324",
    "sample_0000334",
    "sample_0000346",
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load(path: Path, *, zero_origin: bool, device: torch.device) -> MotionNet:
    model = MotionNet(dropout=0.0).to(device)
    saved = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(saved["model"], strict=False)
    if zero_origin:
        model.origin.weight.data.zero_()
        model.origin.bias.data.zero_()
    model.eval()
    return model, {
        "path": str(path),
        "epoch": int(saved.get("epoch", -1)),
        "trained_transition_lambda": float(saved.get("transition_lambda", float("nan"))),
        "origin_conditioned": bool(saved.get("origin_conditioned", False)) and not zero_origin,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def _turn_mask(velocity: np.ndarray) -> np.ndarray:
    n0 = max(int(velocity.shape[0]) // 5, 1)
    base = float(np.mean(velocity[:n0]))
    return np.abs(velocity - base) >= TURN_MPS


def _mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = np.abs(pred - truth)
    if mask is not None:
        if not np.any(mask):
            return float("nan")
        err = err[mask]
    return float(np.mean(err))


def _score(vx: np.ndarray, vy: np.ndarray, xy: np.ndarray, state: np.ndarray) -> dict[str, float]:
    n = min(vx.shape[0], state.shape[0])
    gt = state[:n]
    truth = gt[:, [0, 2]]
    delta = xy[:n] - truth
    return {
        "vx_mae": _mae(vx[:n], gt[:, 1]),
        "vy_mae": _mae(vy[:n], gt[:, 3]),
        "vx_turn_mae": _mae(vx[:n], gt[:, 1], _turn_mask(gt[:, 1])),
        "vy_turn_mae": _mae(vy[:n], gt[:, 3], _turn_mask(gt[:, 3])),
        "trajectory_rmse_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "endpoint_error_m": float(np.hypot(delta[-1, 0], delta[-1, 1])),
        "anchor_error_m": float(np.hypot(xy[0, 0] - gt[0, 0], xy[0, 1] - gt[0, 2])),
    }


def _mean(rows: list[dict]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}


@torch.no_grad()
def _predict(model: MotionNet, feat: np.ndarray, state: np.ndarray, psi: np.ndarray, device: torch.device):
    frames = min(int(feat.shape[-1]), int(state.shape[0]))
    batch = torch.from_numpy(np.ascontiguousarray(feat[:, :frames])).unsqueeze(0).to(device)
    lengths = torch.tensor([frames], dtype=torch.long, device=device)
    origin = torch.tensor([[float(state[0, 0]), float(state[0, 2])]], dtype=torch.float32, device=device)
    out = model(batch, lengths, origin)
    log_px = out["vel_x"][0, :frames].log_softmax(-1).cpu().numpy()
    log_py = out["vel_y"][0, :frames].log_softmax(-1).cpu().numpy()
    traj = reconstruct(log_px, log_py, DT, x0=float(state[0, 0]), y0=float(state[0, 2]), psi=psi)
    return traj


def _oracle(state: np.ndarray):
    vx = CENTERS_MPS[velocity_bin_index(state[:, 1])]
    vy = CENTERS_MPS[velocity_bin_index(state[:, 3])]
    xy = integrate_trapezoid(vx, vy, DT, x0=float(state[0, 0]), y0=float(state[0, 2]))
    return vx, vy, xy


def _limits(series: list[np.ndarray], pad: float = 0.04) -> tuple[float, float]:
    lo = min(float(np.min(item)) for item in series)
    hi = max(float(np.max(item)) for item in series)
    span = max(hi - lo, 1.0)
    return lo - pad * span, hi + pad * span


def _overlay(clip_id: str, vehicle: str, state: np.ndarray, curves: dict[str, tuple], path: Path) -> None:
    time = np.arange(state.shape[0]) * DT
    gt_xy = state[:, [0, 2]]
    fig = plt.figure(figsize=(12.6, 7.4))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.15, 1.15, 1.05))
    ax_vx = fig.add_subplot(grid[0, 0])
    ax_vy = fig.add_subplot(grid[0, 1], sharex=ax_vx)
    ax_x = fig.add_subplot(grid[1, 0], sharex=ax_vx)
    ax_y = fig.add_subplot(grid[1, 1], sharex=ax_vx)
    ax_path = fig.add_subplot(grid[:, 2])
    styles = {
        "ground truth": ("black", 1.6, "-"),
        "epoch 27": ("#1f77b4", 1.35, "-"),
        "epoch 35": ("#d62728", 1.35, "-"),
        "2 m/s bins": ("0.35", 1.0, ":"),
    }
    vx_series = [state[:, 1]] + [curves[name][0] for name in ("epoch 27", "epoch 35", "2 m/s bins")]
    vy_series = [state[:, 3]] + [curves[name][1] for name in ("epoch 27", "epoch 35", "2 m/s bins")]
    x_series = [gt_xy[:, 0]] + [curves[name][2][:, 0] for name in ("epoch 27", "epoch 35", "2 m/s bins")]
    y_series = [gt_xy[:, 1]] + [curves[name][2][:, 1] for name in ("epoch 27", "epoch 35", "2 m/s bins")]
    t_lim = (float(time[0]), float(time[-1]))
    for ax, truth, column, ylabel, ylim in (
        (ax_vx, state[:, 1], 0, r"$v_x$ (m/s)", _limits(vx_series)),
        (ax_vy, state[:, 3], 1, r"$v_y$ (m/s)", _limits(vy_series)),
    ):
        ax.plot(time, truth, color="black", lw=1.6, label="ground truth")
        for name in ("epoch 27", "epoch 35", "2 m/s bins"):
            color, width, ls = styles[name]
            ax.plot(time, curves[name][column], color=color, lw=width, ls=ls, label=name)
        ax.set_xlim(*t_lim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax, column, ylabel, ylim in (
        (ax_x, 0, r"$x$ (m)", _limits(x_series)),
        (ax_y, 1, r"$y$ (m)", _limits(y_series)),
    ):
        ax.plot(time, gt_xy[:, column], color="black", lw=1.6, label="ground truth")
        for name in ("epoch 27", "epoch 35", "2 m/s bins"):
            color, width, ls = styles[name]
            ax.plot(time, curves[name][2][:, column], color=color, lw=width, ls=ls, label=name)
        ax.set_xlim(*t_lim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("t (s)")
        ax.grid(True, alpha=0.3)
    ax_path.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", lw=1.6, label="ground truth")
    for name in ("epoch 27", "epoch 35", "2 m/s bins"):
        color, width, ls = styles[name]
        xy = curves[name][2]
        ax_path.plot(xy[:, 0], xy[:, 1], color=color, lw=width, ls=ls, label=name)
    ax_path.scatter(gt_xy[0, 0], gt_xy[0, 1], s=28, color="black", zorder=4)
    ax_path.scatter([0.0], [0.0], marker="+", s=40, color="tab:red", zorder=4, label="microphone")
    x_lim = _limits(x_series + [[0.0]])
    y_lim = _limits(y_series + [[0.0]])
    ax_path.set_xlim(*x_lim)
    ax_path.set_ylim(*y_lim)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_title("path")
    ax_path.grid(True, alpha=0.3)
    ax_vx.set_title(r"$v_x$")
    ax_vy.set_title(r"$v_y$")
    ax_x.set_title(r"$x$")
    ax_y.set_title(r"$y$")
    handles, labels = ax_vx.get_legend_handles_labels()
    mic_h, mic_l = ax_path.get_legend_handles_labels()
    for handle, label in zip(mic_h, mic_l):
        if label == "microphone":
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"{clip_id}  {vehicle}", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join((line, rule, body))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = _device()
    psi = transition_matrix(DT, a_max_for_jump(JUMP_MPS, DT), lam=LAMBDA)
    previous, prev_meta = _load(CKPT_DIR / "motion_polar_epoch27.pt", zero_origin=True, device=device)
    current, curr_meta = _load(CKPT_DIR / "motion.pt", zero_origin=False, device=device)
    epoch38, meta38 = _load(CKPT_DIR / "motion_epoch38.pt", zero_origin=True, device=device)
    clips = discover_clips(resolve_audio_clips())
    by_id = {clip.clip_id: clip for clip in clips}
    test_ids = {clips[i].clip_id for i in split_by_vehicle(clips, seed=0)["test"]}
    missing = [clip_id for clip_id in PLOT_IDS if clip_id not in by_id]
    if missing:
        raise SystemExit(f"missing clips: {missing}")
    print(
        f"device {device}  dt {DT:.6f} s  jump {JUMP_MPS:.0f} m/s  lambda {LAMBDA:.2f}  "
        f"previous epoch {prev_meta['epoch']}  current epoch {curr_meta['epoch']}",
        flush=True,
    )
    names = {
        "epoch 27": previous,
        "epoch 35": current,
        "epoch 38": epoch38,
    }
    per_clip: dict[str, dict] = {}
    order = [clip_id for clip_id in PLOT_IDS if clip_id in test_ids]
    order += sorted(test_ids - set(order))
    for index, clip_id in enumerate(order, start=1):
        clip = by_id[clip_id]
        feat, state = load_arrays(clip)
        feat = standardize_mag(feat)
        n = min(feat.shape[1], state.shape[0])
        feat, state = feat[:, :n], state[:n]
        curves = {}
        scores = {}
        for name, model in names.items():
            traj = _predict(model, feat, state, psi, device)
            curves[name] = (traj.vx, traj.vy, traj.xy)
            scores[name] = _score(traj.vx, traj.vy, traj.xy, state)
        ox, oy, oxy = _oracle(state)
        curves["2 m/s bins"] = (ox, oy, oxy)
        scores["2 m/s bins"] = _score(ox, oy, oxy, state)
        per_clip[clip_id] = {"vehicle": clip.vehicle, "split": "test" if clip_id in test_ids else "other", "scores": scores}
        if clip_id in PLOT_IDS:
            _overlay(
                clip_id,
                clip.vehicle,
                state,
                {key: curves[key] for key in ("epoch 27", "epoch 35", "2 m/s bins")},
                OUT / f"{clip_id}.png",
            )
        if index % 25 == 0 or clip_id in PLOT_IDS:
            print(f"  {index}/{len(order)}  {clip_id}", flush=True)
    test_rows = {name: [per_clip[clip_id]["scores"][name] for clip_id in per_clip if per_clip[clip_id]["split"] == "test"] for name in ("epoch 27", "epoch 35", "epoch 38", "2 m/s bins")}
    means = {name: _mean(rows) for name, rows in test_rows.items()}
    report = {
        "decoder": {"dt_s": DT, "jump_mps": JUMP_MPS, "transition_lambda": LAMBDA, "anchor": "state[0] x,y canonical meters"},
        "previous": prev_meta,
        "current": curr_meta,
        "epoch38": meta38,
        "current_is_single_clip_overfit": False,
        "test_mean": means,
        "clips": {
            clip_id: per_clip[clip_id]
            for clip_id in PLOT_IDS
        },
    }
    def _jsonable(value):
        if isinstance(value, dict):
            return {key: _jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_jsonable(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    (OUT / "compare.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False))
    keys = [
        ("vx_mae", r"$v_x$ MAE (m/s)"),
        ("vy_mae", r"$v_y$ MAE (m/s)"),
        ("vx_turn_mae", r"turning $v_x$ MAE (m/s)"),
        ("vy_turn_mae", r"turning $v_y$ MAE (m/s)"),
        ("trajectory_rmse_m", "trajectory RMSE (m)"),
        ("endpoint_error_m", "endpoint error (m)"),
    ]
    print("test mean", flush=True)
    for key, _label in keys:
        print(
            f"  {key:22s}  e27 {_fmt(means['epoch 27'][key])}  "
            f"e35 {_fmt(means['epoch 35'][key])}  "
            f"e38 {_fmt(means['epoch 38'][key])}  "
            f"bins {_fmt(means['2 m/s bins'][key])}",
            flush=True,
        )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
