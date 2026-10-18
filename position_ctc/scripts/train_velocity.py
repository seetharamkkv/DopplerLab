"""Continuous vx, vy regression with trapezoid integration from the true start.

No CTC, no bins. First audits whether ground-truth velocity reconstructs
labeled positions, then overfits a few clips that contain heading changes.
A full split run starts only after that overfit.

    position_ctc/.venv/bin/python position_ctc/scripts/train_velocity.py
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.data import discover_clips, split_by_vehicle  # noqa: E402
from position_ctc.features import DATA_HOP, DATA_SR, features_from_wav, N_MELS  # noqa: E402
from position_ctc.integrate import integrate_trapezoid as np_trap  # noqa: E402
from position_ctc.paths import ENV_PATH2D_CLIPS, resolve_audio_clips  # noqa: E402
from position_ctc.velocity_track import (  # noqa: E402
    VelocityNet,
    corner_mask,
    integrate_trapezoid,
    velocity_objective,
)

OUT = ROOT / "position_ctc" / "outputs" / "velocity_track"
DT = DATA_HOP / float(DATA_SR)
OVERFIT_EPOCHS = 200
FULL_EPOCHS = 40
BATCH = 4
LR = 1e-3


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip(clip) -> dict:
    feat = features_from_wav(clip.wav_path)
    state = np.load(clip.sample_dir / "metadata" / "canonical_state_frames.npy").astype(np.float64)
    times_path = clip.sample_dir / "metadata" / "frame_times.npy"
    if times_path.is_file():
        times = np.load(times_path).astype(np.float64)
    else:
        times = np.arange(len(state), dtype=np.float64) * DT
    n = min(feat.shape[1], state.shape[0], times.shape[0])
    feat = feat[:, :n]
    state = state[:n]
    times = times[:n]
    speed = np.hypot(state[:, 1], state[:, 3])
    moving = np.flatnonzero(speed > 0.5)
    if moving.size:
        start, stop = int(moving[0]), int(moving[-1]) + 1
        feat = feat[:, start:stop]
        state = state[start:stop]
        times = times[start:stop]
    mag = feat[:N_MELS]
    feat = feat.copy()
    feat[:N_MELS] = (mag - mag.mean()) / (mag.std() + 1e-4)
    return {
        "clip_id": clip.clip_id,
        "vehicle": clip.vehicle,
        "feat": feat.astype(np.float32),
        "state": np.asarray(state, dtype=np.float32),
        "times": np.asarray(times, dtype=np.float64),
    }


def _heading_jumps(state: np.ndarray, min_deg: float = 12.0) -> int:
    vel = np.stack([state[:, 1], state[:, 3]], axis=1)
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > 0.5
    if int(moving.sum()) < 3:
        return 0
    head = np.unwrap(np.arctan2(vel[moving, 1], vel[moving, 0]))
    return int(np.sum(np.abs(np.diff(head)) > np.deg2rad(min_deg)))


def audit_labels(rows: list[dict]) -> dict:
    """Integrate labeled velocity on the clip clock and compare to labeled x, y."""
    scores = []
    for row in rows:
        state = row["state"]
        times = row["times"]
        xy = np.stack([state[:, 0], state[:, 2]], axis=1)
        velx, vely = state[:, 1], state[:, 3]
        dt = np.diff(times)
        trap = np_trap(velx, vely, dt, x0=float(xy[0, 0]), y0=float(xy[0, 1]))
        euler = np.empty_like(xy)
        euler[0] = xy[0]
        euler[1:] = xy[0] + np.cumsum(np.stack([velx[:-1] * dt, vely[:-1] * dt], axis=1), axis=0)
        trap_err = trap - xy
        euler_err = euler - xy
        corners = _heading_jumps(state) > 0
        scores.append(
            {
                "clip_id": row["clip_id"],
                "n": int(len(state)),
                "has_heading_jump": bool(corners),
                "trap_rmse_m": float(np.sqrt(np.mean(np.sum(trap_err**2, axis=1)))),
                "euler_rmse_m": float(np.sqrt(np.mean(np.sum(euler_err**2, axis=1)))),
                "trap_end_m": float(np.linalg.norm(trap_err[-1])),
                "euler_end_m": float(np.linalg.norm(euler_err[-1])),
            }
        )
    trap = np.array([s["trap_rmse_m"] for s in scores])
    euler = np.array([s["euler_rmse_m"] for s in scores])
    return {
        "n_clips": len(scores),
        "n_with_heading_jump": int(sum(s["has_heading_jump"] for s in scores)),
        "trapezoid_rmse_mean_m": float(trap.mean()),
        "trapezoid_rmse_max_m": float(trap.max()),
        "euler_rmse_mean_m": float(euler.mean()),
        "euler_rmse_max_m": float(euler.max()),
        "clips": scores,
    }


def _collate(rows: list[dict], device: torch.device):
    length = max(row["feat"].shape[1] for row in rows)
    feat = torch.zeros(len(rows), rows[0]["feat"].shape[0], length, device=device)
    state = torch.zeros(len(rows), length, 4, device=device)
    times = torch.zeros(len(rows), length, device=device)
    lengths = torch.zeros(len(rows), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        n = row["feat"].shape[1]
        feat[i, :, :n] = torch.from_numpy(row["feat"])
        state[i, :n] = torch.from_numpy(row["state"])
        t = row["times"]
        times[i, :n] = torch.from_numpy(t.astype(np.float32))
        if n < length:
            times[i, n:] = float(t[-1])
        lengths[i] = n
    return feat, state, times, lengths


def _pick_overfit(rows: list[dict], n: int = 4) -> list[dict]:
    moving = [row for row in rows if float(np.median(np.hypot(row["state"][:, 1], row["state"][:, 3]))) > 5.0]
    ranked = sorted(moving or rows, key=_heading_jumps_row, reverse=True)
    return ranked[:n]


def _heading_jumps_row(row: dict) -> int:
    return _heading_jumps(row["state"])


@torch.no_grad()
def _score(model: VelocityNet, rows: list[dict], device: torch.device) -> dict:
    model.eval()
    vx_err, vy_err, path_err, bias_x, bias_y = [], [], [], [], []
    corner_vx, straight_vx = [], []
    for row in rows:
        feat, state, times, lengths = _collate([row], device)
        pred_v = model(feat, lengths)[0, : int(lengths[0])]
        gt_v = torch.stack((state[0, : int(lengths[0]), 1], state[0, : int(lengths[0]), 3]), dim=-1)
        origin = torch.stack((state[0, 0, 0], state[0, 0, 2]))
        pred_xy = integrate_trapezoid(pred_v[None], times[:, : int(lengths[0])], origin[None])[0]
        gt_xy = torch.stack((state[0, : int(lengths[0]), 0], state[0, : int(lengths[0]), 2]), dim=-1)
        dv = (pred_v - gt_v).cpu().numpy()
        corners = corner_mask(gt_v[None])[0].cpu().numpy()
        vx_err.append(float(np.mean(np.abs(dv[:, 0]))))
        vy_err.append(float(np.mean(np.abs(dv[:, 1]))))
        path_err.append(float(np.sqrt(np.mean(np.sum((pred_xy.cpu().numpy() - gt_xy.cpu().numpy()) ** 2, axis=1)))))
        bias_x.append(float(np.mean(dv[:, 0])))
        bias_y.append(float(np.mean(dv[:, 1])))
        if corners.any():
            corner_vx.append(float(np.mean(np.abs(dv[corners, 0]))))
        if (~corners).any():
            straight_vx.append(float(np.mean(np.abs(dv[~corners, 0]))))
    return {
        "vx_mae": float(np.mean(vx_err)),
        "vy_mae": float(np.mean(vy_err)),
        "path_rmse": float(np.mean(path_err)),
        "signed_vx": float(np.mean(bias_x)),
        "signed_vy": float(np.mean(bias_y)),
        "corner_vx_mae": float(np.mean(corner_vx)) if corner_vx else None,
        "straight_vx_mae": float(np.mean(straight_vx)) if straight_vx else None,
    }


def _plot_clip(row: dict, pred_v: np.ndarray, pred_xy: np.ndarray, path: Path, title: str) -> None:
    state = row["state"]
    t = row["times"] - row["times"][0]
    gt_v = np.stack([state[:, 1], state[:, 3]], axis=1)
    gt_xy = np.stack([state[:, 0], state[:, 2]], axis=1)
    cum = np.cumsum(pred_v - gt_v, axis=0) * np.mean(np.diff(row["times"]))
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 9.0))
    axes[0, 0].plot(t, gt_v[:, 0], "k", lw=1.4, label="ground truth")
    axes[0, 0].plot(t, pred_v[:, 0], color="tab:blue", lw=1.2, label="pred")
    axes[0, 0].set_title("vx")
    axes[0, 1].plot(t, gt_v[:, 1], "k", lw=1.4)
    axes[0, 1].plot(t, pred_v[:, 1], color="tab:blue", lw=1.2)
    axes[0, 1].set_title("vy")
    axes[1, 0].plot(t, gt_xy[:, 0], "k", lw=1.4)
    axes[1, 0].plot(t, pred_xy[:, 0], color="tab:blue", lw=1.2)
    axes[1, 0].set_title("x")
    axes[1, 1].plot(t, gt_xy[:, 1], "k", lw=1.4)
    axes[1, 1].plot(t, pred_xy[:, 1], color="tab:blue", lw=1.2)
    axes[1, 1].set_title("y")
    axes[2, 0].plot(t, cum[:, 0], color="tab:red", lw=1.3)
    axes[2, 0].axhline(0.0, color="0.5", lw=0.8)
    axes[2, 0].set_title("cumulative signed vx error × mean Δt (m)")
    axes[2, 1].plot(t, cum[:, 1], color="tab:red", lw=1.3)
    axes[2, 1].axhline(0.0, color="0.5", lw=0.8)
    axes[2, 1].set_title("cumulative signed vy error × mean Δt (m)")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("t (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _window(row: dict, rng: np.random.Generator, length: int = 256, prefer_corner: bool = False) -> dict:
    n = row["feat"].shape[1]
    if n <= length:
        return row
    jumps = np.flatnonzero(
        np.abs(np.diff(np.unwrap(np.arctan2(row["state"][:, 3], row["state"][:, 1])))) > np.deg2rad(12.0)
    )
    if prefer_corner and jumps.size:
        center = int(rng.choice(jumps))
        start = int(np.clip(center - length // 2, 0, n - length))
    else:
        start = int(rng.integers(0, n - length + 1))
    sl = slice(start, start + length)
    return {
        "clip_id": row["clip_id"],
        "vehicle": row["vehicle"],
        "feat": row["feat"][:, sl],
        "state": row["state"][sl],
        "times": row["times"][sl],
    }


def _train_loop(model, rows, device, epochs, prefer_corners: bool, crop: bool) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    rng = np.random.default_rng(0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.arange(len(rows))
        rng.shuffle(order)
        losses = []
        for i in range(0, len(rows), BATCH):
            batch_rows = []
            for idx in order[i : i + BATCH]:
                row = rows[int(idx)]
                if crop:
                    row = _window(row, rng, prefer_corner=prefer_corners and rng.random() < 0.5)
                batch_rows.append(row)
            feat, state, times, lengths = _collate(batch_rows, device)
            pred_v = model(feat, lengths)
            loss, parts = velocity_objective(pred_v, times, state, lengths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            parts["grad"] = float(nn_clip)
            losses.append(parts)
        scored = _score(model, rows, device)
        scored["epoch"] = epoch
        scored["train_loss"] = float(np.mean([p["total"] for p in losses]))
        history.append(scored)
        print(
            f"epoch {epoch:03d}  loss {scored['train_loss']:.3f}  "
            f"vx {scored['vx_mae']:.3f}  vy {scored['vy_mae']:.3f}  path {scored['path_rmse']:.2f} m",
            flush=True,
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = _device()
    clips = discover_clips(resolve_audio_clips(args.data, env=ENV_PATH2D_CLIPS))
    print(f"device {device}  clips {len(clips)}", flush=True)
    loaded = [load_clip(clip) for clip in clips]
    audit = audit_labels(loaded)
    (OUT / "gt_integrate_audit.json").write_text(json.dumps(audit, indent=2))
    print(
        f"GT integrate  trap {audit['trapezoid_rmse_mean_m']:.4f} m  "
        f"euler {audit['euler_rmse_mean_m']:.4f} m  "
        f"jumps {audit['n_with_heading_jump']}/{audit['n_clips']}",
        flush=True,
    )
    splits = split_by_vehicle(clips, seed=0)
    train = [loaded[i] for i in splits["train"]]
    overfit = _pick_overfit(train, n=4)
    print("overfit", [row["clip_id"] for row in overfit], flush=True)
    model = VelocityNet(dropout=0.0).to(device)
    history = _train_loop(model, overfit, device, OVERFIT_EPOCHS, prefer_corners=True, crop=False)
    torch.save({"model": model.state_dict(), "history": history}, OUT / "overfit.pt")
    (OUT / "overfit.json").write_text(json.dumps({"clips": [r["clip_id"] for r in overfit], "history": history}, indent=2))
    model.eval()
    with torch.no_grad():
        for row in overfit:
            feat, state, times, lengths = _collate([row], device)
            pred_v = model(feat, lengths)[0, : int(lengths[0])].cpu().numpy()
            origin = np.array([row["state"][0, 0], row["state"][0, 2]], dtype=np.float64)
            pred_xy = np_trap(pred_v[:, 0], pred_v[:, 1], np.diff(row["times"]), x0=float(origin[0]), y0=float(origin[1]))
            _plot_clip(row, pred_v, pred_xy, OUT / f"overfit_{row['clip_id']}.png", f"overfit {row['clip_id']}")
    last = history[-1]
    fitted = last["path_rmse"] < 2.0 and last["vx_mae"] < 1.0 and last["vy_mae"] < 1.5
    print(f"overfit path {last['path_rmse']:.2f} m  vx {last['vx_mae']:.3f}  fitted={fitted}", flush=True)
    if not fitted:
        print("Overfit did not reconstruct the handful of clips. Not starting the full split.", flush=True)
        return
    print("Overfit passed. Training the vehicle holdout.", flush=True)
    model = VelocityNet().to(device)
    history = _train_loop(model, train, device, FULL_EPOCHS, prefer_corners=True, crop=True)
    torch.save({"model": model.state_dict()}, OUT / "velocity.pt")
    val = [loaded[i] for i in splits["val"]]
    test = [loaded[i] for i in splits["test"]]
    report = {
        "audit": {k: audit[k] for k in audit if k != "clips"},
        "overfit_last": last,
        "val": _score(model, val, device),
        "test": _score(model, test, device),
        "history": history,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"val": report["val"], "test": report["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
