import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Shared.data.dataset import PATH_TO_ID
from Shared.features.extraction import MAX_SPEED_MPS

def plot_results(df, model_names, colors, fig_dir, speed_lim, dist_lim,
                 dist_bins, bin_labels, max_dist):
    os.makedirs(fig_dir, exist_ok=True)
    path_names = ["straight", "parabola", "bezier"]

    # 1. Confusion matrices
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1: axes = [axes]
    fig.suptitle("Path Classification - Confusion Matrices", fontsize=13)
    for ax, mname in zip(axes, model_names):
        sub = df[df["model"] == mname]
        conf = np.zeros((3, 3), dtype=int)
        for _, row in sub.iterrows():
            conf[PATH_TO_ID[row["path_gt"]], PATH_TO_ID[row["path_pred"]]] += 1
        acc = 100 * np.trace(conf) / conf.sum()
        ax.imshow(conf, cmap="Blues", vmin=0)
        ax.set_xticks(range(3))
        ax.set_xticklabels(path_names, rotation=20)
        ax.set_yticks(range(3))
        ax.set_yticklabels(path_names)
        ax.set_title(f"{mname} Acc: {acc:.1f}%")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(conf[r, c]), ha="center", va="center")
    plt.savefig(os.path.join(fig_dir, "confusion_matrices.png"))
    plt.close()

    # 2. Speed scatter
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 4.5))
    if len(model_names) == 1: axes = [axes]
    for ax, mname in zip(axes, model_names):
        sub = df[df["model"] == mname]
        ax.scatter(sub["speed_gt"], sub["speed_pred"], s=10, alpha=0.45, color=colors.get(mname, 'blue'))
        ax.plot(speed_lim, speed_lim, "k--")
        ax.set_title(f"{mname} Speed MAE: {sub['speed_err'].mean():.2f}")
    plt.savefig(os.path.join(fig_dir, "speed_scatter.png"))
    plt.close()

    # 3. Distance scatter
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 4.5))
    if len(model_names) == 1: axes = [axes]
    for ax, mname in zip(axes, model_names):
        sub = df[df["model"] == mname]
        ax.scatter(sub["dist_gt"], sub["dist_pred"], s=10, alpha=0.45, color=colors.get(mname, 'blue'))
        ax.plot(dist_lim, dist_lim, "k--")
        ax.set_title(f"{mname} Dist MAE: {sub['dist_err'].mean():.2f}")
    plt.savefig(os.path.join(fig_dir, "distance_scatter.png"))
    plt.close()
