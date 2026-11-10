#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ebranch_tracker.core import discover, read_split, orbit_align
from ebranch_tracker.data import Clips, collate
from ebranch_tracker.model import Tracker
from ebranch_tracker.evaluation import evaluate
from train_ebranchformer_tracker import choose_device


def main():
    p = argparse.ArgumentParser(description='Evaluate this package checkpoint and save trajectories/plots')
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True, help='Training run directory')
    p.add_argument('--split-name', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--plots', type=int, default=20)
    p.add_argument('--cache')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--decode', choices=['viterbi', 'argmax'], default='viterbi')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out).expanduser().resolve()
    checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    available = discover(Path(a.root).expanduser().resolve())
    split = read_split(out / 'split.json', available)
    with np.load(out / 'feature_stats.npz') as z:
        stats = {'mean': z['mean'], 'std': z['std']}
    data = Clips(available, split[a.split_name], config['features'], a.cache, stats)
    loader = DataLoader(data, batch_size=a.batch_size, shuffle=False, collate_fn=collate)
    device = choose_device(a.device)
    model = Tracker(**config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    destination = out / f'inference_{a.split_name}_{a.decode}'
    destination.mkdir(parents=True, exist_ok=True)
    plot_count = 0

    def save(name, times, pred, target, velocity, truth_velocity, mask, row):
        nonlocal plot_count
        aligned = orbit_align(pred, target, mask)
        np.savez_compressed(destination / f'{name}.npz', times=times, pred_xy=pred,
                            target_xy=target, orbit_aligned_xy=aligned, pred_velocity=velocity,
                            target_velocity=truth_velocity, moving=mask)
        if plot_count >= a.plots:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].plot(target[:, 0], target[:, 1], label='True')
        axes[0].plot(pred[:, 0], pred[:, 1], label='Predicted')
        axes[0].plot(aligned[:, 0], aligned[:, 1], '--', label='Orbit aligned')
        axes[0].scatter([0], [0], c='black', marker='x', label='Microphone')
        axes[0].scatter(*target[0], marker='o', c='green', label='Labeled start')
        axes[0].set_aspect('equal', adjustable='datalim')
        axes[0].set(xlabel='x (m)', ylabel='y (m)', title=f'ADE {row["ade_m"]:.2f} m')
        axes[0].legend(fontsize=8)
        for axis in range(2):
            axes[axis + 1].plot(times, truth_velocity[:, axis], label='True')
            axes[axis + 1].plot(times, velocity[:, axis], label='Decoded')
            axes[axis + 1].set(xlabel='WAV time (s)', ylabel='m/s', title=('vx', 'vy')[axis])
            axes[axis + 1].legend()
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(destination / f'{name}.png', dpi=150)
        plt.close(fig)
        plot_count += 1

    penalty = config['viterbi_penalty'] if a.decode == 'viterbi' else 0.
    summary, rows = evaluate(model, loader, np.asarray(checkpoint['bins']), device,
                              penalty, config['max_jump'], callback=save)
    summary.update(checkpoint_epoch=checkpoint['epoch'], decode=a.decode,
                   viterbi_penalty=penalty, max_jump=config['max_jump'])
    (destination / 'metrics.json').write_text(json.dumps(summary, indent=2))
    with (destination / 'per_clip.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Predictions and plots: {destination}', flush=True)


if __name__ == '__main__':
    main()
