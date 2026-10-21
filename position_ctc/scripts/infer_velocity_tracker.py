#!/usr/bin/env python3
"""Run inference from best.pt and write a few path-estimation figures."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_velocity_tracker as tr

TRUE = '#1b3a4b'
PRED = '#c44536'
MIC = '#111111'
START = '#2a9d8f'


def path_limits(item, pad_frac=0.1):
    pts = np.vstack([item['true_p'], item['p']])
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    pad_x = max(pad_frac * (xmax - xmin), 3.0)
    pad_y = max(pad_frac * (ymax - ymin), 0.07 * (xmax - xmin), 3.0)
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def maybe_mic(ax, xlim, ylim):
    in_x = xlim[0] <= 0 <= xlim[1]
    in_y = ylim[0] <= 0 <= ylim[1]
    if in_x and in_y:
        ax.scatter([0], [0], marker='x', s=70, c=MIC, linewidths=1.8, zorder=5, label='microphone')
        return
    ax.annotate('mic at (0, 0)', xy=(0.02, 0.96), xycoords='axes fraction',
                fontsize=8, color='#555555', va='top')


def device_of(name):
    if name != 'auto':
        return name
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_run(out, root, device):
    config = json.loads((out / 'config.json').read_text())
    split = json.loads((out / 'split.json').read_text())
    ck = torch.load(out / 'best.pt', map_location=device, weights_only=False)
    centers = np.asarray(ck['centers'], dtype=np.float32)
    norm = np.load(out / 'normalization.npz')
    records = tr.discover(root.resolve())
    index = {r['sample']: r for r in records}
    partitions = {k: [index[n] for n in split[k]] for k in ('val', 'test')}
    cache = out / 'feature_cache'
    loaders = {
        k: DataLoader(
            tr.Clips(v, cache, norm['mean'], norm['std']),
            batch_size=int(config.get('batch_size', 4)),
            shuffle=False, num_workers=0, collate_fn=tr.collate,
        )
        for k, v in partitions.items()
    }
    model = tr.Tracker(len(centers), int(config.get('hidden', 128))).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return ck, centers, loaders, model


def decode_one(name, lp, s, t, centers, penalty):
    v = np.stack([tr.viterbi(lp[:, d], centers, penalty) for d in range(2)], axis=1)
    p = tr.integrate(v, t, s[0, [0, 2]])
    true_p, true_v = s[:, [0, 2]], s[:, [1, 3]]
    error = p - true_p
    return dict(
        sample=name, v=v, p=p, true_p=true_p, true_v=true_v, t=t, error=error,
        path_rmse_m=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        ade_m=float(np.mean(np.linalg.norm(error, axis=1))),
        vx_mae=float(np.mean(np.abs(v[:, 0] - true_v[:, 0]))),
        vy_mae=float(np.mean(np.abs(v[:, 1] - true_v[:, 1]))),
        endpoint_error_m=float(np.linalg.norm(error[-1])),
        heading_change=float(heading_change(true_v)),
    )


def heading_change(v):
    speed = np.linalg.norm(v, axis=1)
    ang = np.arctan2(v[:, 1], v[:, 0])
    da = np.diff(ang)
    da = np.arctan2(np.sin(da), np.cos(da))
    valid = (speed[1:] > 0.5) & (speed[:-1] > 0.5)
    return np.abs(da[valid]).sum() if valid.any() else 0.0


def decode_all(predictions, centers, penalty):
    return [decode_one(name, lp, s, t, centers, penalty) for name, lp, s, t in predictions]


def style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#d9d9d9',
        'axes.linewidth': 0.8,
        'grid.color': '#ececec',
        'grid.linewidth': 0.7,
        'legend.frameon': False,
        'savefig.dpi': 180,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.12,
    })


def add_path(ax, item, colorbar=False, time_color=False):
    true_p, pred_p, t = item['true_p'], item['p'], item['t']
    ax.plot(true_p[:, 0], true_p[:, 1], color='#9aa8b2', lw=4.6, solid_capstyle='round',
            label='labeled path', zorder=2, alpha=0.95)
    lc = None
    if time_color:
        segs = np.stack([pred_p[:-1], pred_p[1:]], axis=1)
        lc = LineCollection(segs, cmap='plasma', linewidths=2.2, zorder=3, capstyle='round')
        lc.set_array(t[:-1])
        ax.add_collection(lc)
    else:
        ax.plot(pred_p[:, 0], pred_p[:, 1], color=PRED, lw=2.1, solid_capstyle='round',
                label='predicted path', zorder=3)
    ax.scatter(true_p[0, 0], true_p[0, 1], s=36, c=START, zorder=6, edgecolors='white',
               linewidths=0.6, label='start')
    ax.scatter(pred_p[-1, 0], pred_p[-1, 1], s=28, marker='s', c=PRED, zorder=6,
               edgecolors='white', linewidths=0.5, label='predicted end')
    xlim = path_limits(item)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    ax.set_aspect('equal', adjustable='box')
    maybe_mic(ax, (xlim[0], xlim[1]), (xlim[2], xlim[3]))
    ax.grid(True, alpha=0.9)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(
        f"{Path(item['sample']).name}\n"
        f"path RMSE {item['path_rmse_m']:.1f} m  ·  end {item['endpoint_error_m']:.1f} m"
    )
    if colorbar and lc is not None:
        cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('observer time (s)')
    return lc


def pick_clips(decoded):
    by_rmse = sorted(decoded, key=lambda d: d['path_rmse_m'])
    good = by_rmse[:3]
    median = by_rmse[len(by_rmse) // 2]
    turning = sorted(by_rmse[: max(20, len(by_rmse) // 3)], key=lambda d: -d['heading_change'])
    chosen, seen = [], set()
    for item in (*good, median, *turning[:2]):
        if item['sample'] in seen:
            continue
        chosen.append(item)
        seen.add(item['sample'])
        if len(chosen) == 6:
            break
    return chosen, by_rmse[0]


def write_figures(decoded, dest, penalty, best_epoch):
    style()
    chosen, best = pick_clips(decoded)
    gallery = dest / 'gallery_paths.png'
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2))
    handles = labels = None
    for ax, item in zip(axes.flat, chosen):
        add_path(ax, item, time_color=False)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f'Test-set path estimates  ·  best checkpoint epoch {best_epoch}'
        f'  ·  Viterbi L1 penalty {penalty:g}',
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(gallery)
    plt.close(fig)

    detail = dest / 'detail_best_path.png'
    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 8.4),
        gridspec_kw={'width_ratios': [1.15, 1], 'height_ratios': [1.15, 1]},
    )
    add_path(axes[0, 0], best, colorbar=True, time_color=True)
    axes[0, 0].legend(loc='center left', fontsize=8, bbox_to_anchor=(1.18, 0.5))
    axes[0, 1].plot(best['t'], best['error'][:, 0], color='#3d5a80', lw=1.6, label='x error')
    axes[0, 1].plot(best['t'], best['error'][:, 1], color='#e07a5f', lw=1.6, label='y error')
    axes[0, 1].axhline(0, color='#bbbbbb', lw=0.8)
    axes[0, 1].set(xlabel='observer time (s)', ylabel='position error (m)')
    axes[0, 1].set_title('Trapezoid-integrated position error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    for d, ax, ylab in ((0, axes[1, 0], 'vx (m/s)'), (1, axes[1, 1], 'vy (m/s)')):
        ax.plot(best['t'], best['true_v'][:, d], color=TRUE, lw=2.0, label='labeled')
        ax.plot(best['t'], best['v'][:, d], color='#c44536', lw=1.7, label='predicted')
        ax.set(xlabel='observer time (s)', ylabel=ylab)
        ax.legend()
        ax.grid(True)
    fig.suptitle(f"Best test clip  ·  {Path(best['sample']).name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(detail)
    plt.close(fig)

    turn = max(chosen, key=lambda d: d['heading_change'])
    turn_path = dest / 'detail_turning_path.png'
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    add_path(ax, turn, colorbar=True, time_color=True)
    ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Turning clip  ·  predicted path colored by time', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(turn_path)
    plt.close(fig)

    sample844 = next(d for d in decoded if '0000844' in Path(d['sample']).name)
    extra = write_sample_detail(sample844, dest)
    return [gallery, detail, turn_path, *extra], chosen, best


def write_sample_detail(item, dest):
    """Standalone path + time-series figures for one clip, matching the gallery path style."""
    true_p, pred_p, t = item['true_p'], item['p'], item['t']
    path_png = dest / f"{Path(item['sample']).name}_path.png"
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.plot(true_p[:, 0], true_p[:, 1], color=TRUE, lw=2.5, solid_capstyle='round',
            label='labeled path', zorder=2)
    segs = np.stack([pred_p[:-1], pred_p[1:]], axis=1)
    lc = LineCollection(segs, cmap='plasma', linewidths=2.3, zorder=3, capstyle='round')
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    ax.scatter([0], [0], marker='x', s=90, c=MIC, linewidths=2.0, zorder=5, label='microphone')
    ax.scatter(true_p[0, 0], true_p[0, 1], s=42, c=START, zorder=6, edgecolors='white',
               linewidths=0.7, label='start')
    ax.scatter(pred_p[-1, 0], pred_p[-1, 1], s=34, marker='s', c=PRED, zorder=6,
               edgecolors='white', linewidths=0.6, label='predicted end')
    pts = np.vstack([true_p, pred_p, np.array([[0.0, 0.0]])])
    pad = 6.0
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.9)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(
        f"{Path(item['sample']).name}\n"
        f"path RMSE {item['path_rmse_m']:.1f} m  ·  end {item['endpoint_error_m']:.1f} m"
    )
    ax.legend(loc='upper right')
    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('observer time (s)')
    fig.tight_layout()
    fig.savefig(path_png)
    plt.close(fig)

    series_png = dest / f"{Path(item['sample']).name}_xy_vxvy.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)
    series = (
        (axes[0], true_p[:, 0], pred_p[:, 0], 'x (m)'),
        (axes[1], true_p[:, 1], pred_p[:, 1], 'y (m)'),
    )
    for ax, truth, pred, ylab in series:
        ax.plot(t, truth, color=TRUE, lw=2.1, label='ground truth')
        ax.plot(t, pred, color=PRED, lw=1.8, label='predicted')
        ax.set_ylabel(ylab)
        ax.set_xlabel('observer time (s)')
        ax.grid(True)
        ax.legend(loc='best')
    fig.suptitle(f"{Path(item['sample']).name}  ·  position vs time", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(series_png)
    plt.close(fig)
    return [path_png, series_png]


def mean_metrics(decoded):
    ades = np.array([
        d.get('ade_m', float(np.mean(np.linalg.norm(d['error'], axis=1))))
        for d in decoded
    ])
    fdes = np.array([d['endpoint_error_m'] for d in decoded])
    keys = ('path_rmse_m', 'vx_mae', 'vy_mae', 'endpoint_error_m')
    metrics = {k: float(np.mean([d[k] for d in decoded])) for k in keys}
    metrics.update(
        n_clips=len(decoded),
        ade_m=float(ades.mean()),
        ade_median_m=float(np.median(ades)),
        fde_m=float(fdes.mean()),
        fde_median_m=float(np.median(fdes)),
        headline=(
            f"Test ADE: {ades.mean():.2f} m across {len(decoded)} held-out clips"
        ),
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/ordered_velocity_full'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--figures-only', action='store_true',
                        help='Redraw figures from a previous inference pickle.')
    args = parser.parse_args()
    dest = args.out / 'inference'
    dest.mkdir(parents=True, exist_ok=True)
    cache_path = dest / 'test_decoded.pkl'
    decoder_path = dest / 'decoder.json'
    val_scores = None
    if args.figures_only:
        payload = pickle.loads(cache_path.read_bytes())
        test, chosen, best_epoch = payload['test'], payload['penalty'], payload['best_epoch']
        test_metrics = mean_metrics(test)
        if decoder_path.exists():
            val_scores = json.loads(decoder_path.read_text()).get('validation')
    else:
        device = device_of(args.device)
        ck, centers, loaders, model = load_run(args.out, args.root, device)
        print(json.dumps(dict(device=device, best_epoch=int(ck['epoch']))), flush=True)
        validation = tr.predict(model, loaders['val'], device)
        candidates = [0., 0.01, 0.03, 0.1, 0.3, 1., 3.]
        val_scores = [{'penalty': p, **mean_metrics(decode_all(validation, centers, p))} for p in candidates]
        chosen = min(val_scores, key=lambda x: x['path_rmse_m'])['penalty']
        test = decode_all(tr.predict(model, loaders['test'], device), centers, chosen)
        test_metrics = mean_metrics(test)
        best_epoch = int(ck['epoch'])
        cache_path.write_bytes(pickle.dumps(dict(test=test, penalty=chosen, best_epoch=best_epoch)))
    decoder_path.write_text(json.dumps(
        dict(penalty=chosen, validation=val_scores, test=test_metrics, best_epoch=best_epoch),
        indent=2,
    ))
    paths, clips, best = write_figures(test, dest, chosen, best_epoch)
    print(test_metrics['headline'], flush=True)
    print(json.dumps(dict(
        penalty=chosen,
        test=test_metrics,
        figures=[str(p) for p in paths],
        shown=[c['sample'] for c in clips],
        best_clip=best['sample'],
    ), indent=2), flush=True)


if __name__ == '__main__':
    main()
