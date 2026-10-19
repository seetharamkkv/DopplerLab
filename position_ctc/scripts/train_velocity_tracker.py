#!/usr/bin/env python3
"""Offline, full-frame ordered-bin vx/vy tracker. Python 3.10+.
Install: pip install torch numpy scipy matplotlib
Run: python train_velocity_tracker.py --root /path/to/audio_clips --out outputs/ordered_velocity
Optional strict vehicle holdout: --groups-csv groups.csv (columns: sample,vehicle).
Default split is random CLIP holdout, NOT vehicle holdout. True p0 is used for evaluation.
No early stopping. Labels: metadata/canonical_state_frames.npy = x,vx,y,vy.
"""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader


def integrate(v, t, p0):
    steps = .5 * (v[1:] + v[:-1]) * np.diff(t)[:, None]
    return np.concatenate([np.asarray(p0)[None], p0 + np.cumsum(steps, axis=0)])


def discover(root):
    records = []
    for label in sorted(root.rglob('canonical_state_frames.npy')):
        sample = label.parent.parent
        time = label.parent / 'frame_times.npy'
        if not time.exists():
            raise FileNotFoundError(time)
        waves = sorted(sample.rglob('*.wav'))
        if len(waves) != 1:
            raise ValueError(f'{sample}: expected ONE WAV, found {len(waves)}; remove ambiguity.')
        records.append(dict(sample=str(sample.relative_to(root)), wav=str(waves[0]),
                            state=str(label), time=str(time)))
    if len(records) < 10:
        raise ValueError(f'Only {len(records)} labeled clips discovered under {root}')
    return records


def load_labels(r):
    state = np.load(r['state']).astype(np.float32)
    t = np.load(r['time']).reshape(-1).astype(np.float64)
    if state.ndim != 2 or state.shape != (len(t), 4):
        raise ValueError(f"Bad state shape {state.shape}: {r['sample']}")
    if len(t) < 2 or not np.isfinite(state).all() or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError(f"Invalid labels/times: {r['sample']}")
    return state, t


def features(r, cache):
    # Centered STFT: sample index 0 corresponds to observer t=0. Interpolate
    # features to actual label times rather than assuming identical frame counts.
    paths = [Path(r[k]) for k in ('wav', 'state', 'time')]
    signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    key = hashlib.sha256(json.dumps(['stft2048-hop512-v1', signature]).encode()).hexdigest()
    dest = cache / (key + '.npy')
    if dest.exists():
        return np.load(dest)
    sr, audio = wavfile.read(r['wav'])
    if sr != 44100 or audio.ndim != 1:
        raise ValueError(f"Need mono 44100 Hz: {r['wav']}")
    if audio.dtype != np.int16:
        raise ValueError(f"Need 16-bit PCM, got {audio.dtype}: {r['wav']}")
    a = torch.from_numpy(audio.astype(np.float32) / 32768.)
    if len(a) < 2048:
        raise ValueError('Audio too short')
    spec = torch.stft(a, n_fft=2048, hop_length=512, window=torch.hann_window(2048),
                      center=True, return_complex=True).abs().clamp_min(1e-5).log().numpy().T
    _, t = load_labels(r)
    if t[0] < -1e-6 or t[-1] > (len(audio)-1)/sr + 512/sr:
        raise ValueError(f"Label times outside audio: {r['sample']}")
    st = np.arange(len(spec)) * 512 / sr
    x = np.stack([np.interp(t, st, spec[:, j]) for j in range(spec.shape[1])], axis=1).astype(np.float32)
    np.save(dest, x)
    return x


class Clips(Dataset):
    def __init__(self, records, cache, mean, std):
        self.records, self.cache, self.mean, self.std = records, cache, mean, std
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        s, t = load_labels(r)
        x = (features(r, self.cache) - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(s), t, r['sample']


def collate(items):
    lengths = torch.tensor([len(i[0]) for i in items])
    n = int(lengths.max())
    x = torch.zeros(len(items), n, items[0][0].shape[1])
    s = torch.zeros(len(items), n, 4)
    for j, (a, b, _, _) in enumerate(items):
        x[j, :len(a)], s[j, :len(b)] = a, b
    return x, s, lengths, [i[2] for i in items], [i[3] for i in items]


class Tracker(nn.Module):
    def __init__(self, bins, hidden=128):
        super().__init__()
        # Time stride is one everywhere: no temporal downsampling.
        self.cnn = nn.Sequential(nn.Conv2d(1, 16, (5, 9), padding=(2, 4), stride=(1, 4)),
            nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, (5, 7), padding=(2, 3), stride=(1, 4)),
            nn.GroupNorm(8, 32), nn.GELU(), nn.AdaptiveAvgPool2d((None, 8)))
        self.gru = nn.GRU(256, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=.1)
        self.head = nn.Linear(hidden*2, bins*2)
        self.bins = bins
    def forward(self, x, lengths):
        # 1025 STFT bins -> 65 after two stride-4 convs; MPS adaptive pool
        # requires the pooled axis to divide the output size, so drop Nyquist.
        if x.shape[-1] > 1024:
            x = x[..., :1024]
        # Process clips individually through CNN to exclude padded frames from GroupNorm.
        seq = []
        for i, length in enumerate(lengths.tolist()):
            z = self.cnn(x[i:i+1, :length].unsqueeze(1))
            seq.append(z.permute(0, 2, 1, 3).reshape(length, 256))
        z = nn.utils.rnn.pad_sequence(seq, batch_first=True)
        if x.device.type == 'mps':
            z, _ = self.gru(z)
        else:
            packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
            z, _ = self.gru(packed)
            z, _ = pad_packed_sequence(z, batch_first=True, total_length=x.shape[1])
        return self.head(z).reshape(x.shape[0], x.shape[1], 2, self.bins)


def targets(v, centers):
    return ((v - centers[0]) / (centers[1]-centers[0])).round().long().clamp(0, len(centers)-1)


def ordinal_loss(logits, state, lengths, centers, weights, corner_weight):
    v = state[..., [1, 3]]
    k = targets(v, centers)
    cdf = logits.float().softmax(-1).cumsum(-1)[..., :-1]
    j = torch.arange(len(centers)-1, device=v.device)
    truth = (j >= k.unsqueeze(-1)).float()
    loss = (cdf-truth).square().sum(-1)
    mask = torch.arange(state.shape[1], device=v.device)[None] < lengths.to(v.device)[:, None]
    # Upweight both sides of heading jumps; do not penalize valid discontinuities.
    corner = torch.zeros_like(mask)
    speed = v.norm(dim=-1)
    angles = torch.atan2(v[..., 1], v[..., 0])
    da = angles[:, 1:] - angles[:, :-1]
    da = torch.atan2(da.sin(), da.cos()).abs()
    valid_pair = mask[:, 1:] & mask[:, :-1] & (speed[:, 1:] > .5) & (speed[:, :-1] > .5)
    jumps = (da >= math.radians(12)) & valid_pair
    corner[:, 1:] |= jumps
    corner[:, :-1] |= jumps
    corner = torch.nn.functional.max_pool1d(corner.float()[:, None],  nine := 9, stride=1,
                                             padding=nine//2).squeeze(1) > 0
    cw = 1 + corner.float() * (corner_weight-1)
    total = 0
    for d in range(2):
        w = weights[d][k[..., d]] * cw * mask
        # Normalize by valid frames, not sum of weights: rare-target emphasis remains.
        total = total + (loss[..., d]*w).sum() / mask.sum().clamp_min(1)
    return total


def viterbi(logp, centers, penalty):
    """Exact O(T*K) DP for an L1 transition cost on uniformly spaced bins."""
    t, k = logp.shape
    back = np.zeros((t, k), dtype=np.int32)
    cost = -logp[0].astype(np.float64)
    step = penalty * float(centers[1]-centers[0])
    for i in range(1, t):
        best = cost.copy()
        arg = np.arange(k, dtype=np.int32)
        for j in range(1, k):
            if best[j-1]+step < best[j]:
                best[j], arg[j] = best[j-1]+step, arg[j-1]
        for j in range(k-2, -1, -1):
            if best[j+1]+step < best[j]:
                best[j], arg[j] = best[j+1]+step, arg[j+1]
        cost = best-logp[i]
        cost -= cost.min()
        back[i] = arg
    path = np.empty(t, dtype=np.int32)
    path[-1] = cost.argmin()
    for i in range(t-1, 0, -1):
        path[i-1] = back[i, path[i]]
    return centers[path]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    result = []
    for x, s, lengths, times, names in loader:
        lp = model(x.to(device), lengths).log_softmax(-1).cpu().numpy()
        for i, n in enumerate(lengths.tolist()):
            result.append((names[i], lp[i, :n], s[i, :n].numpy(), times[i]))
    return result


def score(predictions, centers, penalty, out=None, plots=0):
    rows = []
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for index, (name, lp, s, t) in enumerate(predictions):
        v = np.stack([viterbi(lp[:, d], centers, penalty) for d in range(2)], axis=1)
        p = integrate(v, t, s[0, [0, 2]])
        true_p, true_v = s[:, [0, 2]], s[:, [1, 3]]
        error = p-true_p
        rows.append(dict(sample=name, path_rmse_m=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            vx_mae=float(np.mean(np.abs(v[:, 0]-true_v[:, 0]))),
            vy_mae=float(np.mean(np.abs(v[:, 1]-true_v[:, 1]))),
            endpoint_error_m=float(np.linalg.norm(error[-1]))))
        if out:
            tag = hashlib.sha256(name.encode()).hexdigest()[:10]
            np.savez_compressed(out/f'{tag}.npz', sample=name, times=t, pred_velocity=v,
                                pred_position=p, true_state=s)
            if index < plots:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, 2, figsize=(12, 8))
                ax[0, 0].plot(true_p[:, 0], true_p[:, 1], label='truth')
                ax[0, 0].plot(p[:, 0], p[:, 1], label='predicted')
                ax[0, 0].scatter([0], [0], marker='x', label='mic')
                ax[0, 0].axis('equal'); ax[0, 0].set(xlabel='x (m)', ylabel='y (m)')
                for d in range(2):
                    ax[1, d].plot(t, true_v[:, d], label='truth')
                    ax[1, d].plot(t, v[:, d], label='predicted')
                    ax[1, d].set(xlabel='observer time (s)', ylabel=['vx (m/s)', 'vy (m/s)'][d])
                ax[0, 1].plot(t, error[:, 0], label='x error')
                ax[0, 1].plot(t, error[:, 1], label='y error')
                ax[0, 1].set(xlabel='observer time (s)', ylabel='cumulative error (m)')
                for a in ax.flat: a.legend(); a.grid(alpha=.2)
                fig.suptitle(name); fig.tight_layout(); fig.savefig(out/f'{tag}.png'); plt.close(fig)
    metrics = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'sample'}
    if out:
        (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
        with (out/'per_clip.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('outputs/ordered_velocity'))
    p.add_argument('--groups-csv', type=Path)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--bin-width', type=float, default=1.)
    p.add_argument('--vmin', type=float, default=-60.)
    p.add_argument('--vmax', type=float, default=60.)
    p.add_argument('--corner-weight', type=float, default=3.)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--plots', type=int, default=20)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    if args.bin_width <= 0 or args.vmax <= args.vmin or args.epochs < 1 or args.eval_every < 1:
        p.error('Invalid bin range, epochs, or eval interval')
    if abs((args.vmax-args.vmin)/args.bin_width-round((args.vmax-args.vmin)/args.bin_width)) > 1e-5:
        p.error('Bin range must be divisible by bin width')
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out/'feature_cache'; cache.mkdir(exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    centers_np = np.linspace(args.vmin, args.vmax, round((args.vmax-args.vmin)/args.bin_width)+1).astype(np.float32)
    centers = torch.from_numpy(centers_np).to(device)
    records = discover(args.root.resolve())
    # Fixed manifest persists exact sample membership and prevents accidental split changes.
    manifest_path = args.out/'split.json'
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if (args.out/'last.pt').exists() and not args.resume:
        raise ValueError('Output already contains a run. Use --resume or a new --out.')
    if args.resume:
        old = json.loads((args.out/'config.json').read_text())
        for k in ('root', 'hidden', 'bin_width', 'vmin', 'vmax', 'seed', 'corner_weight', 'epochs'):
            if old[k] != config[k]: raise ValueError(f'Resume config mismatch: {k}')
    if manifest_path.exists():
        split = json.loads(manifest_path.read_text())
    else:
        rng = random.Random(args.seed)
        if args.groups_csv:
            with args.groups_csv.open() as f:
                groups = {r['sample']: r['vehicle'] for r in csv.DictReader(f)}
            missing = [r['sample'] for r in records if r['sample'] not in groups]
            if missing: raise ValueError(f'groups.csv missing samples: {missing[:5]}')
            unique = sorted(set(groups[r['sample']] for r in records)); rng.shuffle(unique)
            if len(unique) < 3: raise ValueError('Vehicle holdout needs at least 3 groups')
            n = max(1, round(.15*len(unique)))
            val_groups, test_groups = set(unique[:n]), set(unique[n:2*n])
            split = {k: [] for k in ('train', 'val', 'test')}
            for r in records:
                g = groups[r['sample']]
                split['val' if g in val_groups else 'test' if g in test_groups else 'train'].append(r['sample'])
        else:
            names = [r['sample'] for r in records]; rng.shuffle(names)
            n = max(1, round(.15*len(names)))
            split = dict(val=names[:n], test=names[n:2*n], train=names[2*n:])
        manifest_path.write_text(json.dumps(split, indent=2))
    index = {r['sample']: r for r in records}
    flat = sum((split[k] for k in ('train', 'val', 'test')), [])
    if len(flat) != len(set(flat)) or set(flat) != set(index):
        raise ValueError('Dataset differs from saved split or split contains duplicates')
    partitions = {k: [index[n] for n in split[k]] for k in split}
    if any(not partitions[k] for k in ('train', 'val', 'test')): raise ValueError('Empty split')
    print(f'Device={device}; clips=' + str({k: len(v) for k,v in partitions.items()}), flush=True)
    print('Split: vehicle holdout' if args.groups_csv else 'Split: random clip holdout (vehicle types may overlap)', flush=True)
    # Cache all full-frame features and audit EVERY clip; never silently drop stationary frames.
    for i, r in enumerate(records):
        s, t = load_labels(r)
        v = s[:, [1,3]]
        if (v < args.vmin).any() or (v > args.vmax).any():
            raise ValueError(f"{r['sample']}: velocities outside bin range; expand --vmin/--vmax")
        features(r, cache)
        if (i+1) % 50 == 0: print(f'Cached {i+1}/{len(records)}', flush=True)
    norm_path = args.out/'normalization.npz'
    if args.resume:
        norm = np.load(norm_path); mean, std = norm['mean'], norm['std']
    else:
        count = 0; sums = np.zeros(1025); squares = np.zeros(1025)
        for r in partitions['train']:
            x = features(r, cache).astype(np.float64)
            count += len(x); sums += x.sum(0); squares += (x*x).sum(0)
        mean = (sums/count).astype(np.float32)
        std = np.sqrt(np.maximum(squares/count-mean.astype(np.float64)**2, 1e-4)).astype(np.float32)
        np.savez(norm_path, mean=mean, std=std)
    counts = np.zeros((2, len(centers_np)), dtype=np.float64)
    for r in partitions['train']:
        s, _ = load_labels(r)
        k = np.rint((s[:, [1,3]]-args.vmin)/args.bin_width).astype(int).clip(0, len(centers_np)-1)
        for d in range(2): counts[d] += np.bincount(k[:,d], minlength=len(centers_np))
    weights = np.zeros_like(counts)
    for d in range(2):
        raw = 1/np.sqrt(np.maximum(counts[d], 1))
        # Normalize under the training frame distribution, then cap extremes.
        raw /= np.sum(raw*counts[d])/counts[d].sum()
        weights[d] = np.minimum(raw, 5)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    loaders = {k: DataLoader(Clips(v, cache, mean, std), batch_size=args.batch_size,
                shuffle=(k=='train'), num_workers=args.workers, collate_fn=collate,
                pin_memory=(device=='cuda')) for k,v in partitions.items()}
    model = Tracker(len(centers_np), args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.05)
    best = float('inf'); start = 1
    if args.resume:
        ck = torch.load(args.out/'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); start=ck['epoch']+1; best=ck['best']
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device=='cuda' and ck.get('cuda_rng') is not None: torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
    (args.out/'config.json').write_text(json.dumps(config, indent=2))
    for epoch in range(start, args.epochs+1):
        model.train(); total=0.; batches=0
        for x, s, lengths, _, _ in loaders['train']:
            x, s = x.to(device), s.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = ordinal_loss(model(x,lengths), s, lengths, centers, weights, args.corner_weight)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); total+=loss.item(); batches+=1
        scheduler.step()
        row = dict(epoch=epoch, train_loss=total/batches, lr=optimizer.param_groups[0]['lr'])
        if epoch==1 or epoch%args.eval_every==0 or epoch==args.epochs:
            pred = predict(model, loaders['val'], device)
            m = score(pred, centers_np, 0.)
            row.update({f'val_{k}': v for k,v in m.items()})
            # Select network by velocity error; choose decoder separately afterward.
            selection = .5*(m['vx_mae']+m['vy_mae'])
            if selection < best:
                best=selection
                torch.save(dict(model=model.state_dict(), epoch=epoch, centers=centers_np,
                                config=config), args.out/'best.pt')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch, best=best,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None), args.out/'last.pt')
        with (args.out/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    ck = torch.load(args.out/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    validation = predict(model, loaders['val'], device)
    candidates = [0., .01, .03, .1, .3, 1., 3.]
    scores = [{ 'penalty': v, **score(validation, centers_np, v)} for v in candidates]
    chosen = min(scores, key=lambda x:x['path_rmse_m'])['penalty']
    (args.out/'decoder.json').write_text(json.dumps(dict(penalty=chosen, validation=scores), indent=2))
    score(validation, centers_np, chosen, args.out/'validation', args.plots)
    test = score(predict(model, loaders['test'], device), centers_np, chosen, args.out/'test', args.plots)
    print(json.dumps(dict(best_epoch=ck['epoch'], penalty=chosen, test=test), indent=2), flush=True)


if __name__ == '__main__':
    main()
