"""Trajectory figure: velocities, the anchored path, and probability heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from position_ctc.bins import CENTERS_MPS


def trajectory_figure(
    time_s: np.ndarray,
    gt_vx: np.ndarray,
    gt_vy: np.ndarray,
    gt_xy: np.ndarray,
    pred_vx: np.ndarray,
    pred_vy: np.ndarray,
    pred_xy: np.ndarray,
    path: Path,
    *,
    title: str,
    vx_argmax: np.ndarray | None = None,
    vy_argmax: np.ndarray | None = None,
    xy_argmax: np.ndarray | None = None,
) -> None:
    """Write one PNG.

    Blue is the Viterbi velocity and the position obtained by integrating
    those bin centers from the true first sample. Gray is the framewise
    argmax, integrated from the same anchor. Both start at that anchor.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    fig = plt.figure(figsize=(12.4, 7.2))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.15, 1.15, 1.05))
    ax_vx = fig.add_subplot(grid[0, 0])
    ax_vy = fig.add_subplot(grid[0, 1], sharex=ax_vx)
    ax_x = fig.add_subplot(grid[1, 0], sharex=ax_vx)
    ax_y = fig.add_subplot(grid[1, 1], sharex=ax_vx)
    ax_path = fig.add_subplot(grid[:, 2])

    for ax, truth, decoded, ylabel, name, framewise in (
        (ax_vx, gt_vx, pred_vx, r"$v_x$ (m/s)", r"$v_x$", vx_argmax),
        (ax_vy, gt_vy, pred_vy, r"$v_y$ (m/s)", r"$v_y$", vy_argmax),
    ):
        ax.plot(time_s, truth, color="black", lw=1.4, label="ground truth")
        ax.plot(time_s, decoded, color="tab:blue", lw=1.3, label="Viterbi")
        if framewise is not None:
            ax.plot(time_s, framewise, color="0.45", lw=0.9, ls=":", label="framewise argmax")
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    argmax_xy = None if xy_argmax is None else np.asarray(xy_argmax, dtype=np.float64)
    for ax, column, name in ((ax_x, 0, r"$x$"), (ax_y, 1, r"$y$")):
        ax.plot(time_s, gt_xy[:, column], color="black", lw=1.4, label="ground truth")
        ax.plot(time_s, pred_xy[:, column], color="tab:blue", lw=1.3, label="anchored Viterbi")
        if argmax_xy is not None:
            ax.plot(time_s, argmax_xy[:, column], color="0.45", lw=0.9, ls=":", label="argmax integral")
        ax.set_ylabel(f"{name} (m)")
        ax.set_xlabel("t (s)")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    ax_path.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", lw=1.5, label="ground truth")
    ax_path.plot(pred_xy[:, 0], pred_xy[:, 1], color="tab:blue", lw=1.4, label="anchored Viterbi")
    if argmax_xy is not None:
        ax_path.plot(argmax_xy[:, 0], argmax_xy[:, 1], color="0.45", lw=0.9, ls=":", label="argmax integral")
    ax_path.scatter(gt_xy[0, 0], gt_xy[0, 1], s=28, color="black", zorder=4, label="initial position")
    ax_path.scatter(pred_xy[0, 0], pred_xy[0, 1], s=16, color="tab:blue", zorder=4)
    ax_path.scatter([0.0], [0.0], marker="+", s=40, color="tab:red", label="microphone", zorder=3)
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_title("path")
    ax_path.grid(True, alpha=0.3)

    handles, labels = [], []
    for ax in (ax_vx, ax_x, ax_path):
        found_handles, found_labels = ax.get_legend_handles_labels()
        for handle, label in zip(found_handles, found_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=3, frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11, y=0.98)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def probability_heatmap(
    time_s: np.ndarray,
    logp: np.ndarray,
    gt_mps: np.ndarray,
    argmax_mps: np.ndarray,
    decoded_mps: np.ndarray,
    path: Path,
    *,
    title: str,
    centers: np.ndarray | None = None,
) -> None:
    """Per-axis probability before decoding, with the three velocity traces."""
    centers = CENTERS_MPS if centers is None else np.asarray(centers, dtype=np.float64)
    probs = np.exp(np.asarray(logp, dtype=np.float64))
    time_s = np.asarray(time_s, dtype=np.float64)
    half = 0.5 * float(centers[1] - centers[0]) if centers.size > 1 else 1.0
    extent = [float(time_s[0]), float(time_s[-1]), float(centers[0] - half), float(centers[-1] + half)]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    image = ax.imshow(
        probs.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        interpolation="nearest",
    )
    ax.plot(time_s, gt_mps, color="cyan", lw=1.3, label="ground truth")
    ax.plot(time_s, argmax_mps, color="white", lw=1.0, ls=":", label="framewise argmax")
    ax.plot(time_s, decoded_mps, color="lime", lw=1.1, ls="--", label="Viterbi")
    ax.set_ylabel("velocity (m/s)")
    ax.set_xlabel("t (s)")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="probability")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def error_over_time_figure(
    fraction: np.ndarray,
    viterbi_m: np.ndarray,
    argmax_m: np.ndarray,
    path: Path,
    *,
    title: str,
) -> None:
    """Mean position error against normalized clip time."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(fraction, viterbi_m, color="tab:blue", lw=1.6, label="anchored Viterbi")
    ax.plot(fraction, argmax_m, color="0.35", lw=1.2, ls=":", label="argmax integral")
    ax.set_xlabel("fraction of the clip")
    ax.set_ylabel("position error (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
