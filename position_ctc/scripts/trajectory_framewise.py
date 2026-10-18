#!/usr/bin/env python3
"""Audio -> frequency Conv1d -> BiLSTM -> framewise velocity -> Viterbi -> x,y.

INSTALL: python -m pip install torch numpy scipy matplotlib
TRAIN + PLOT:
  python trajectory_framewise.py --root /path/to/DopplerSim --out runs/trajectory86
EVALUATE AN EXISTING RUN:
  python trajectory_framewise.py --root /path/to/DopplerSim --out runs/trajectory86 --eval-only

Default clip layout (searched recursively):
  clip/metadata/canonical_state_frames.npy  [T,4], columns x,vx,y,vy
  clip/metadata/frame_times.npy             [T], seconds from WAV sample zero
  clip/**/one_recording.wav                 mono; --wav-name resolves ambiguity

Alternative: --manifest clips.csv with columns id,wav,state,times and optional
split (train/val/test), group (shared source/path family). Paths are relative to
CSV directory. All rows must have split, or none. Existing splits are preserved.
Automatic splitting keeps every rotation of one path together, and keeps every
clip that shares a source recording in one split. Recordings whose trajectories
match within 0.5 m are merged so near-copies cannot leak across splits.
sample_0000014 and close path variants stay out of the test split. Validation
uses several of the remaining source groups, chosen so their y-span and signed
lateral displacement match the clips left in training.

Conventions:
* WAV sample rate is read from its header and resampled to --sample-rate.
* STFT stays high-rate; frequency convolutions do not downsample time.
* BiLSTM sees ALL waveform frames including trailing propagation-delay context.
* Every STFT frame receives a prediction; there is NO temporal pooling/upsampling.
* Default output rate is 44100/512 = 86.13 Hz, determined by the actual STFT hop.
* Labels are matched to frame centers; velocity resampling is nearest-neighbor.
  This preserves discrete corner changes instead of inventing intermediate values.
  Position is linearly interpolated to those same centers without extrapolation.
* N point velocities produce N positions via trapezoidal integration.
* p0 is the vehicle position at the FIRST SUPERVISED STFT frame in canonical coordinates.
* The network receives only that audio spectrum and p0. Labels never enter the encoder.
* Neighbor soft-target cross-entropy, plus a gradually weighted Huber loss on the
  trajectory formed by trapezoid-integrating expected bin velocities. CTC is off
  unless --ctc-weight is positive. Position residuals are divided by a fixed
  10 m scale; that scale is not estimated from the data.
* Frequency pool is 64, CNN features are LayerNormed, and each frame adds a residual skip.
* No independent position/acceleration head or unaligned CTC-token integration.
* Viterbi jump limit is m/s per STFT FRAME, not a physical guarantee.
* Output probabilities are scores; calibrated uncertainty is not assumed.

The script trains on paired data; it does not promise a particular accuracy.
It intentionally uses one clip per update to keep variable lengths uncomplicated.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F


# ------------------------- Data and fixed splits -------------------------
def read_records(args):
    records = []
    if args.manifest:
        source = Path(args.manifest).resolve()
        with source.open(newline='') as f:
            for row in csv.DictReader(f):
                if not all(row.get(k) for k in ('id', 'wav', 'state', 'times')):
                    raise ValueError('Manifest requires id,wav,state,times columns.')
                r = {k: str((source.parent / row[k]).resolve())
                     for k in ('wav', 'state', 'times')}
                r.update(id=row['id'], group=row.get('group') or row['id'],
                         split=row.get('split', '').strip())
                records.append(r)
    else:
        if not args.root:
            raise ValueError('Supply --root or --manifest.')
        root = Path(args.root).resolve()
        for state in sorted(root.rglob('canonical_state_frames.npy')):
            clip = state.parent.parent
            times = state.parent / 'frame_times.npy'
            wavs = [clip / args.wav_name] if args.wav_name else sorted(clip.rglob('*.wav'))
            if len(wavs) != 1:
                raise ValueError(f'{clip}: found {len(wavs)} WAVs. Use --wav-name or manifest.')
            clip_id = clip.relative_to(root).as_posix() or clip.name
            records.append(dict(id=clip_id, wav=str(wavs[0]), state=str(state),
                                times=str(times), group=clip_id, split=''))
        assign_recording_groups(records)
    if not records:
        raise ValueError('No clips found; expected canonical state labels and WAVs.')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Clip IDs must be unique.')
    for r in records:
        for k in ('wav', 'state', 'times'):
            if not Path(r[k]).is_file():
                raise FileNotFoundError(r[k])
    # Duplicate WAV/state files must not masquerade as independent clips.
    for k in ('wav', 'state'):
        paths = [str(Path(r[k]).resolve()) for r in records]
        if len(set(paths)) != len(paths):
            raise ValueError(f'Duplicate {k} paths. Use one record per physical clip.')
    return records


def _path_signature(state_path):
    params = Path(state_path).parent / 'simulation_parameters.json'
    if not params.is_file():
        return None, None
    payload = json.loads(params.read_text())
    source = payload.get('source_clip') or None
    xy = np.asarray(payload.get('path_xy') or [], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 2:
        return source, None
    fraction = np.linspace(0., 1., len(xy))
    grid = np.linspace(0., 1., 40)
    signature = np.stack([np.interp(grid, fraction, xy[:, 0]),
                          np.interp(grid, fraction, xy[:, 1])])
    return source, signature


def assign_recording_groups(records, path_merge_m=0.5, related_m=3.0,
                            held_sample='sample_0000014'):
    """Group rerenders of one recording, and near-duplicate trajectories.

    Merging every pair closer than about 1 m collapses this dataset into one
    component, because different recordings bridge one another. Pairs closer
    than path_merge_m are the same trajectory rendered twice. Clips within
    related_m of held_sample, and every recording grouped with them, stay out
    of the test split.
    """
    sources, signatures = [], []
    for record in records:
        source, signature = _path_signature(record['state'])
        sources.append(source or record['id'])
        signatures.append(signature)
        record['source'] = Path(sources[-1]).name
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    by_source = {}
    for index, source in enumerate(sources):
        by_source.setdefault(source, []).append(index)
    for members in by_source.values():
        for index in members[1:]:
            union(members[0], index)
    # 0/45/90 renders share a WAV name and the same canonical labels. They stay
    # together even when the exported path_xy has been rotated apart.
    by_wav = {}
    for index, record in enumerate(records):
        by_wav.setdefault(Path(record['wav']).name, []).append(index)
    for members in by_wav.values():
        for index in members[1:]:
            union(members[0], index)
    for left in range(len(records)):
        if signatures[left] is None:
            continue
        for right in range(left + 1, len(records)):
            if signatures[right] is None or sources[left] == sources[right]:
                continue
            distance = float(np.mean(np.linalg.norm(signatures[left] - signatures[right], axis=0)))
            if distance < path_merge_m:
                union(left, right)
    names = {}
    for index, record in enumerate(records):
        root = find(index)
        if root not in names:
            members = sorted({Path(sources[i]).name for i in range(len(records)) if find(i) == root})
            names[root] = '+'.join(members)
        record['group'] = names[root]
    held = next((i for i, record in enumerate(records)
                 if Path(record['id']).name == held_sample), None)
    held_groups = set()
    if held is not None:
        held_groups.add(records[held]['group'])
        if signatures[held] is not None:
            for index, signature in enumerate(signatures):
                if signature is None:
                    continue
                distance = float(np.mean(np.linalg.norm(signature - signatures[held], axis=0)))
                if distance < related_m:
                    held_groups.add(records[index]['group'])
                    records[index]['related_to_held_m'] = distance
    # MercedesGLA_45.wav is the ten-clip test already inspected in trajectory86_skip64.
    # Those clips stay in development data; the final test is a different recording.
    for record in records:
        if record['source'] == 'MercedesGLA_45.wav':
            held_groups.add(record['group'])
            record['inspected_development'] = True
    for record in records:
        record['hold_from_test'] = record['group'] in held_groups


def _wav_lateral(records):
    """One y-span and signed-displacement row per WAV name.

    Rotations repeat the canonical labels, so counting every copy would weight
    a path by how many angles were rendered.
    """
    chosen = {}
    for record in records:
        wav = Path(record['wav']).name
        if wav in chosen:
            continue
        y = np.load(record['state'])[:, 2]
        chosen[wav] = (record['group'], float(y.max() - y.min()), float(y[-1] - y[0]))
    return list(chosen.values())


def choose_val_groups(val_pool, test_groups, records):
    """Pick several source groups whose trajectories match the training clips.

    Taking the first 15% of a shuffled pool can leave validation as one
    recording. On this data that recording was VW Passat, whose median y span
    is about 42 m against about 24 m for the other sources. With two or more
    groups available, combinations that stay within 25% of the non-test paths
    are ranked by how close they are to 15%, then by the gap in median y span
    (scaled by 10 m) and signed displacement (scaled by 5 m). Rotations of one
    WAV share labels, so each WAV votes once.
    """
    pool = sorted(val_pool)
    if len(pool) < 2:
        return set(pool)
    test = set(test_groups)
    by_group = {}
    for group, span, disp in _wav_lateral(records):
        if group in test:
            continue
        by_group.setdefault(group, []).append((span, disp))
    ranked = []
    for width in range(2, len(pool) + 1):
        for combo in itertools.combinations(pool, width):
            val_rows = [item for group in combo for item in by_group.get(group, ())]
            rest_rows = [item for group, items in by_group.items()
                         if group not in combo for item in items]
            if len(val_rows) < 2 or len(rest_rows) < 2:
                continue
            val_med = np.median(val_rows, axis=0)
            rest_med = np.median(rest_rows, axis=0)
            gap = (abs(val_med[0] - rest_med[0]) / 10.
                   + abs(val_med[1] - rest_med[1]) / 5.)
            frac = len(val_rows) / (len(val_rows) + len(rest_rows))
            ranked.append((frac > 0.25, abs(frac - 0.15), gap, width, combo))
    if not ranked:
        return set(pool[:2])
    ranked.sort()
    return set(ranked[0][-1])


def make_splits(records, output, seed, n_examples, eval_only=False):
    filename = output / 'splits.json'
    ids = {r['id'] for r in records}
    if filename.exists():
        splits = json.loads(filename.read_text())
        assigned = [x for part in ('train', 'val', 'test') for x in splits[part]]
        if len(assigned) != len(set(assigned)) or set(assigned) != ids:
            raise ValueError('Existing splits do not match these records. Use a new --out.')
    else:
        if eval_only:
            raise FileNotFoundError('Evaluation requires this run\'s splits.json.')
        supplied = [r['split'] for r in records]
        if any(supplied):
            if not all(x in ('train', 'val', 'test') for x in supplied):
                raise ValueError('Every manifest row must have train/val/test, or none.')
            splits = {part: [r['id'] for r in records if r['split'] == part]
                      for part in ('train', 'val', 'test')}
        else:
            held_groups = {r['group'] for r in records if r.get('hold_from_test')}
            sizes = {}
            for record in records:
                sizes.setdefault(record['group'], set()).add(Path(record['wav']).name)
            sizes = {group: len(wavs) for group, wavs in sizes.items()}
            eligible = sorted(g for g in sizes if g not in held_groups and sizes[g] <= n_examples)
            oversized = sorted(g for g in sizes if g not in held_groups and sizes[g] > n_examples)
            if not eligible:
                eligible = list(oversized)
                oversized = []
            random.Random(seed).shuffle(eligible)
            test_groups, test_count = [], 0
            while test_count < n_examples and eligible:
                fits = [g for g in eligible if test_count + sizes[g] <= n_examples or test_count == 0]
                if not fits:
                    break
                group = fits[0]
                eligible.remove(group)
                test_groups.append(group)
                test_count += sizes[group]
            remainder = eligible + oversized + sorted(held_groups)
            random.Random(seed + 7).shuffle(remainder)
            val_pool = [g for g in remainder if g not in held_groups]
            val_groups = choose_val_groups(val_pool, test_groups, records)
            train_groups = set(remainder) - val_groups
            test_groups = set(test_groups)
            splits = {'train': [r['id'] for r in records if r['group'] in train_groups],
                      'val': [r['id'] for r in records if r['group'] in val_groups],
                      'test': [r['id'] for r in records if r['group'] in test_groups],
                      'validation_groups': sorted(val_groups)}
        if not splits['train'] or not splits['val'] or len(splits['test']) < n_examples:
            raise ValueError('Need nonempty train/val and at least --examples test clips.')
        held_ids = {r['id'] for r in records if r.get('hold_from_test')}
        if held_ids & set(splits['test']):
            raise ValueError('Held recordings leaked into the test split.')
        selected = random.Random(seed + 1).sample(sorted(splits['test']), n_examples)
        splits['examples'] = selected  # Selected BEFORE training or inspecting errors.
    owner = {x: part for part in ('train', 'val', 'test') for x in splits[part]}
    group_owner = {}
    for r in records:
        part = owner[r['id']]
        if r['group'] in group_owner and group_owner[r['group']] != part:
            raise ValueError(f'Group {r["group"]} crosses splits.')
        group_owner[r['group']] = part
    if len(splits['examples']) != n_examples:
        raise ValueError('Keep --examples identical to the original run.')
    filename.write_text(json.dumps(splits, indent=2))
    return splits


def load_audio(path, requested_rate):
    rate, data = wavfile.read(path)
    if data.ndim != 1:
        raise ValueError(f'{path}: expected one mono microphone, got {data.shape}.')
    if np.issubdtype(data.dtype, np.integer):
        if data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.) / 128.
        else:
            scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
            data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    if not np.isfinite(data).all() or len(data) < 2:
        raise ValueError(f'{path}: invalid audio.')
    if rate != requested_rate:
        divisor = math.gcd(int(rate), requested_rate)
        data = resample_poly(data, requested_rate // divisor, int(rate) // divisor)
    return np.ascontiguousarray(data, dtype=np.float32)


class ClipStore:
    def __init__(self, records, config, cache):
        self.records = {r['id']: r for r in records}
        self.config, self.cache = config, cache
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, clip_id):
        r, c = self.records[clip_id], self.config
        signature = {'config': c, 'record': r,
                     'files': [(Path(r[k]).stat().st_size, Path(r[k]).stat().st_mtime_ns)
                               for k in ('wav', 'state', 'times')]}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
        filename = self.cache / f'{key}.npz'
        if filename.exists():
            with np.load(filename, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        audio = load_audio(r['wav'], c['sample_rate'])
        state = np.load(r['state'], allow_pickle=False)
        label_times = np.load(r['times'], allow_pickle=False).astype(np.float64)
        if state.ndim != 2 or state.shape[1] != 4:
            raise ValueError(f'{clip_id}: requires 2D [T,4] state ordered x,vx,y,vy.')
        if label_times.ndim != 1 or len(label_times) != len(state) or len(state) < 2:
            raise ValueError(f'{clip_id}: state/time shape mismatch.')
        if not np.isfinite(state).all() or not np.isfinite(label_times).all():
            raise ValueError(f'{clip_id}: nonfinite labels.')
        if not (np.diff(label_times) > 0).all():
            raise ValueError(f'{clip_id}: times must increase strictly.')
        duration = (len(audio) - 1) / c['sample_rate']
        if label_times[0] < -1e-6 or label_times[-1] > duration + 1e-6:
            raise ValueError(f'{clip_id}: label times lie outside WAV. Check time origin/rate.')
        waveform = torch.from_numpy(audio)
        spectrum = torch.stft(waveform, c['n_fft'], hop_length=c['hop'],
                              window=torch.hann_window(c['n_fft']), center=True,
                              pad_mode='constant', return_complex=True)
        times = np.arange(spectrum.shape[1], dtype=np.float64) * c['hop'] / c['sample_rate']
        audio_mask = times <= duration
        audio_times = times[audio_mask]
        features = torch.log1p(spectrum.abs()).T.numpy()[audio_mask].astype(np.float32)
        # Keep all audio as context, but supervise only within the label clock.
        selected = np.flatnonzero((audio_times >= label_times[0]) &
                                  (audio_times <= label_times[-1]))
        if len(selected) < 2:
            raise ValueError(f'{clip_id}: need at least two supervised STFT frames.')
        target_times = audio_times[selected]
        position = np.column_stack([np.interp(target_times, label_times, state[:, col])
                                    for col in (0, 2)])
        # Match velocity labels to the nearest source frame. At identical clocks
        # this is exact. At a changed clock it preserves discontinuous labels.
        right = np.searchsorted(label_times, target_times).clip(0, len(label_times)-1)
        left = (right-1).clip(0, len(label_times)-1)
        nearest = np.where(np.abs(target_times-label_times[left]) <=
                           np.abs(label_times[right]-target_times), left, right)
        velocity = state[nearest][:, (1, 3)].astype(np.float32)
        if (np.abs(velocity) > c['vmax'] + 1e-5).any():
            raise ValueError(f'{clip_id}: frame velocity exceeds bins; increase --vmax.')
        result = dict(features=features, audio_times=audio_times, selected=selected,
                      target_times=target_times, position=position.astype(np.float32),
                      velocity=velocity)
        np.savez(filename, **result)
        return result


def training_normalization(store, train_ids):
    # One global mean/std from TRAIN features; absolute time-varying level remains.
    total = square = 0.
    count = 0
    for clip_id in train_ids:
        x = store.get(clip_id)['features'].astype(np.float64)
        total += x.sum()
        square += np.square(x).sum()
        count += x.size
    mean = total / count
    std = math.sqrt(max(square / count - mean * mean, 1e-8))
    return {'mean': float(mean), 'std': float(std), 'clips': list(train_ids), 'values': int(count)}


def audit_dataset(store, splits, config):
    report = {'splits': splits, 'per_clip': {}}
    for part in ('train', 'val', 'test'):
        rows = []
        for clip_id in splits[part]:
            clip = store.get(clip_id)
            v, p = clip['velocity'], clip['position']
            bins = np.rint(v / config['bin_width'])
            rows.append(dict(id=clip_id, frames=len(v),
                x_span=float(np.ptp(p[:, 0])), y_span=float(np.ptp(p[:, 1])),
                lateral_travel=float(np.abs(np.diff(p[:, 1])).sum()),
                vy_std=float(v[:, 1].std()), zero_vy_baseline=float(np.abs(v[:, 1]).mean()),
                zero_fraction=float((bins[:, 1] == 0).mean()),
                positive_fraction=float((bins[:, 1] > 0).mean()),
                negative_fraction=float((bins[:, 1] < 0).mean()),
                forbidden_transition_fraction=float((np.abs(np.diff(bins, axis=0)) * config['bin_width'] > config['max_jump']).mean())))
        report['per_clip'][part] = rows
    return report


def audit_preprocessing(store, splits, normalization):
    """Confirm the encoder sees audio and the first position, fit on train only."""
    train, val, test = set(splits['train']), set(splits['val']), set(splits['test'])
    if train & val or train & test or val & test:
        raise ValueError('Splits overlap.')
    if set(normalization['clips']) != train:
        raise ValueError('Normalization was not fitted on the training recordings only.')
    clip_id = splits['train'][0]
    clip = store.get(clip_id)
    features = clip['features']
    if features.ndim != 2 or features.shape[1] < 8:
        raise ValueError('Features must be the log-magnitude spectrum, with no label columns.')
    p0 = np.asarray(clip['position'][0], dtype=np.float32)
    if p0.shape != (2,) or not np.isfinite(p0).all():
        raise ValueError('The permitted initial position must be a finite 2-vector.')
    if clip['velocity'].shape[1] != 2 or clip['position'].shape[0] < 2:
        raise ValueError('Labels are present but must stay out of the feature tensor.')
    held = [clip_id for clip_id in test if Path(clip_id).name == 'sample_0000014']
    if held:
        raise ValueError('sample_0000014 is in the test split.')
    return {
        'network_inputs': [
            'normalized log1p absolute STFT of the waveform',
            'position of the first supervised frame',
        ],
        'excluded_from_network': [
            'velocity labels', 'later positions', 'source recording name', 'clip id',
        ],
        'normalization_split': 'train',
        'normalization_clips': len(train),
        'normalization_values': normalization['values'],
        'initialized_from': None,
        'ctc': 'off',
    }


# ------------------------- Model and training loss -------------------------
class MotionFramewise(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config['hidden']
        freq_pool = int(config.get('freq_pool', 64))
        width, vmax = config['bin_width'], config['vmax']
        centers = torch.arange(-vmax, vmax + width / 2, width)
        self.register_buffer('centers', centers)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(freq_pool), nn.Flatten(),
            nn.Linear(32 * freq_pool, hidden), nn.ReLU())
        self.norm = nn.LayerNorm(hidden)
        self.anchor = nn.Linear(2, 16)
        self.lstm = nn.LSTM(hidden + 16, hidden, batch_first=True, bidirectional=True)
        self.skip = nn.Linear(hidden, hidden * 2, bias=False)
        self.velocity = nn.ModuleList([nn.Linear(hidden * 2, len(centers)) for _ in range(2)])

    def forward(self, features, selected, p0):
        # Time is the Conv1d batch, so every STFT frame is kept.
        if features.ndim != 2:
            raise ValueError('Network input must be [T, F] audio features plus a separate p0.')
        frames = self.cnn(features.unsqueeze(1))
        if frames.shape[0] != features.shape[0]:
            raise RuntimeError(f'CNN changed time from {features.shape[0]} to {frames.shape[0]}.')
        feat = self.norm(frames)
        cond = torch.tanh(self.anchor(p0 / 100.)).expand(feat.shape[0], -1)
        recurrent, _ = self.lstm(torch.cat([feat, cond], dim=-1).unsqueeze(0))
        hidden = recurrent[0] + self.skip(feat)
        dense = torch.stack([head(hidden[selected]) for head in self.velocity], dim=1)
        return dense, None


def to_device(clip, normalization, device):
    features = (clip['features'] - normalization['mean']) / normalization['std']
    return (torch.as_tensor(features, dtype=torch.float32, device=device),
            torch.as_tensor(clip['selected'], dtype=torch.long, device=device),
            torch.as_tensor(clip['target_times'], dtype=torch.float64, device=device),
            torch.as_tensor(clip['position'][0], dtype=torch.float32, device=device),
            torch.as_tensor(clip['velocity'], dtype=torch.float32, device=device))


def loss_function(dense, ctc, truth, centers, sigma, ctc_weight, vy_balance):
    distance = truth[..., None] - centers
    targets = torch.softmax(-0.5 * (distance / sigma).square(), dim=-1)
    per_frame = -(targets * dense.log_softmax(-1)).sum(-1)  # [N,2]
    x_loss = per_frame[:, 0].mean()
    y_ordinary = per_frame[:, 1].mean()
    bins = distance.abs().argmin(-1)
    zero_bin = centers.abs().argmin()
    is_zero = bins[:, 1] == zero_bin
    if is_zero.any() and (~is_zero).any():
        y_balanced = 0.5 * (per_frame[is_zero, 1].mean() + per_frame[~is_zero, 1].mean())
    else:
        y_balanced = y_ordinary
    neighborhood = x_loss + (1 - vy_balance) * y_ordinary + vy_balance * y_balanced
    sequence = dense.new_zeros(())
    if ctc_weight > 0:
        if ctc is None:
            raise ValueError('CTC is off in this architecture. Train with --ctc-weight 0.')
        for axis in range(2):
            tokens = torch.unique_consecutive(bins[:, axis]) + 1  # Blank=0, physical bins=1..K
            log_probs = ctc[:, axis].log_softmax(-1).unsqueeze(1)
            sequence = sequence + F.ctc_loss(
                log_probs, tokens, torch.tensor([len(truth)], dtype=torch.long),
                torch.tensor([len(tokens)], dtype=torch.long), blank=0,
                reduction='mean', zero_infinity=False)
    total = neighborhood + ctc_weight * sequence
    if not torch.isfinite(total):
        raise ValueError('Nonfinite loss; inspect labels/logits and CTC alignment.')
    return total, neighborhood.detach().item(), sequence.detach().item()


# ------------------------- Graph decoding and integration -------------------------
@torch.no_grad()
def viterbi(log_probs, centers, penalty, max_jump):
    delta = (centers[:, None] - centers[None, :]).abs()
    step = centers[1] - centers[0]
    transition = -penalty * (delta / step).square()
    transition[delta > max_jump] = -torch.inf
    score = log_probs[0]
    back = []
    for emission in log_probs[1:]:
        best, predecessor = (score[:, None] + transition).max(dim=0)
        score = best + emission
        back.append(predecessor)
    current = score.argmax()
    path = [current]
    for predecessor in reversed(back):
        current = predecessor[current]
        path.append(current)
    return torch.stack(path[::-1])


def integrate_frames(velocity, times, p0):
    # Point velocities at STFT frame centers -> trapezoidal integration.
    dt = times.diff().to(velocity)
    if not (dt > 0).all() or velocity.shape != (len(times), 2) or len(times) < 2:
        raise ValueError('Velocity must be [T,2] on T strictly increasing frame times.')
    increments = 0.5 * (velocity[:-1] + velocity[1:]) * dt[:, None]
    return p0[None] + torch.cat([velocity.new_zeros(1, 2), increments.cumsum(0)])


def anchored_position_loss(dense, times, p0, truth_position, centers, scale, delta):
    """Huber loss on the trapezoid integral of expected velocity.

    Expected velocity is the probability-weighted bin center on each axis.
    Integration uses the clip's actual frame timestamps and starts at the true
    position of the first supervised frame, so the first residual is exactly
    zero. Residuals are divided by ``scale`` meters before Huber. ``scale`` is
    a fixed constant, not a statistic of the dataset. Huber delta is in those
    normalized units: delta 1 with scale 10 m is linear beyond a 10 m error.
    """
    if scale <= 0 or delta <= 0:
        raise ValueError('Position scale and Huber delta must be positive.')
    expected = (dense.softmax(-1) * centers).sum(-1)
    position = integrate_frames(expected, times, p0)
    if position.shape != truth_position.shape:
        raise ValueError('Integrated positions and labels must share [T, 2].')
    residual = (position - truth_position) / scale
    return F.huber_loss(residual, torch.zeros_like(residual), reduction='mean', delta=delta)


@torch.no_grad()
def infer(model, clip, normalization, device, config):
    model.eval()
    features, selected, times, p0, truth = to_device(clip, normalization, device)
    dense, _ = model(features, selected, p0)
    log_probs = dense.log_softmax(-1)
    if config['decoder'] == 'greedy':
        indices = log_probs.argmax(-1)
    else:
        indices = torch.stack([viterbi(log_probs[:, axis], model.centers,
                                       config['penalty'], config['max_jump'])
                               for axis in range(2)], dim=1)
    velocity = model.centers[indices]
    position = integrate_frames(velocity, times, p0)
    oracle_velocity = model.centers[(truth[..., None] - model.centers).abs().argmin(-1)]
    oracle_position = integrate_frames(oracle_velocity, times, p0)
    expected_velocity = (log_probs.exp() * model.centers).sum(-1)
    expected_position = integrate_frames(expected_velocity, times, p0)
    return dict(probabilities=log_probs.exp().cpu().numpy(),
                raw_velocity=model.centers[log_probs.argmax(-1)].cpu().numpy(),
                expected_velocity=expected_velocity.cpu().numpy(),
                expected_position=expected_position.cpu().numpy(),
                velocity=velocity.cpu().numpy(), position=position.cpu().numpy(),
                oracle_velocity=oracle_velocity.cpu().numpy(),
                oracle_position=oracle_position.cpu().numpy(),
                true_velocity_integral=integrate_frames(truth, times, p0).cpu().numpy())


def scores(clip, prediction, width):
    truth_v, truth_p = clip['velocity'], clip['position']
    v, p = prediction['velocity'], prediction['position']
    error = p - truth_p
    opening = truth_v[:max(1, len(truth_v) // 5)].mean(axis=0)
    turning = np.linalg.norm(truth_v - opening, axis=1) >= width
    zero = np.abs(truth_v[:, 1]) < width / 2
    raw = prediction['raw_velocity']
    return dict(vx_mae=float(np.abs(v[:, 0] - truth_v[:, 0]).mean()),
                raw_vx_mae=float(np.abs(raw[:, 0] - truth_v[:, 0]).mean()),
                raw_vy_mae=float(np.abs(raw[:, 1] - truth_v[:, 1]).mean()),
                zero_vy_baseline=float(np.abs(truth_v[:, 1]).mean()),
                predicted_zero_fraction=float((np.abs(raw[:, 1]) < width / 2).mean()),
                vy_mae=float(np.abs(v[:, 1] - truth_v[:, 1]).mean()),
                turning_vx_mae=float(np.abs(v[turning, 0] - truth_v[turning, 0]).mean())
                                if turning.any() else None,
                turning_vy_mae=float(np.abs(v[turning, 1] - truth_v[turning, 1]).mean())
                               if turning.any() else None,
                zero_false_motion=float((np.abs(v[zero, 1]) > 1e-6).mean()) if zero.any() else None,
                trajectory_rmse=float(np.sqrt(np.square(error).sum(axis=1).mean())),
                endpoint_error=float(np.linalg.norm(error[-1])),
                max_y_error=float(np.abs(error[:, 1]).max()),
                anchor_error=float(np.linalg.norm(error[0])),
                label_integration_rmse=float(np.sqrt(np.square(prediction['true_velocity_integral'] - truth_p)
                                                     .sum(axis=1).mean())),
                oracle_rmse=float(np.sqrt(np.square(prediction['oracle_position'] - truth_p)
                                          .sum(axis=1).mean())))


def motion_metrics(truth_v, truth_p, velocity, position, width):
    error = position - truth_p
    opening = truth_v[:max(1, len(truth_v) // 5)].mean(axis=0)
    turning = np.linalg.norm(truth_v - opening, axis=1) >= width
    zero = np.abs(truth_v[:, 1]) < width / 2
    signed = (velocity - truth_v).mean(axis=0)
    return dict(vx_mae=float(np.abs(velocity[:, 0] - truth_v[:, 0]).mean()),
                vy_mae=float(np.abs(velocity[:, 1] - truth_v[:, 1]).mean()),
                turning_vy_mae=float(np.abs(velocity[turning, 1] - truth_v[turning, 1]).mean())
                               if turning.any() else None,
                x_rmse=float(np.sqrt(np.square(error[:, 0]).mean())),
                y_rmse=float(np.sqrt(np.square(error[:, 1]).mean())),
                trajectory_rmse=float(np.sqrt(np.square(error).sum(axis=1).mean())),
                endpoint_error=float(np.linalg.norm(error[-1])),
                signed_mean_vx=float(signed[0]),
                signed_mean_vy=float(signed[1]),
                zero_false_motion=float((np.abs(velocity[zero, 1]) > 1e-6).mean()) if zero.any() else None)


def assess_clips(model, store, ids, normalization, device, config, forbidden_ids=()):
    """Score expected, argmax, and Viterbi velocities without reading forbidden clips."""
    leaked = set(ids) & set(forbidden_ids)
    if leaked:
        raise ValueError(f'Refusing to score {len(leaked)} final-test clips.')
    model.eval()
    buckets = {name: [] for name in ('expected', 'argmax', 'viterbi', 'zero_vy')}
    frame_rows = []
    with torch.no_grad():
        for clip_id in ids:
            clip = store.get(clip_id)
            features, selected, times, p0, truth = to_device(clip, normalization, device)
            dense, ctc = model(features, selected, p0)
            log_probs = dense.log_softmax(-1)
            expected = (log_probs.exp() * model.centers).sum(-1)
            argmax = model.centers[log_probs.argmax(-1)]
            decoded = torch.stack([viterbi(log_probs[:, axis], model.centers,
                                           config['penalty'], config['max_jump'])
                                   for axis in range(2)], dim=1)
            viterbi_velocity = model.centers[decoded]
            zero_vy = torch.zeros_like(truth)
            zero_vy[:, 0] = truth[:, 0].mean()
            methods = {'expected': expected, 'argmax': argmax,
                       'viterbi': viterbi_velocity, 'zero_vy': zero_vy}
            truth_v = clip['velocity']
            truth_p = clip['position']
            for name, velocity in methods.items():
                position = integrate_frames(velocity, times, p0).cpu().numpy()
                buckets[name].append(motion_metrics(truth_v, truth_p, velocity.cpu().numpy(),
                                                    position, config['bin_width']))
            loss, _, _ = loss_function(dense, ctc, truth, model.centers, config['sigma'], 0., 0.)
            targets = torch.softmax(-0.5 * ((truth[..., None] - model.centers) / config['sigma']).square(), -1)
            axis_loss = -(targets * dense.log_softmax(-1)).sum(-1).mean(0)
            row = dict(loss=float(loss), vx_loss=float(axis_loss[0]), vy_loss=float(axis_loss[1]))
            bins = (truth[..., None] - model.centers).abs().argmin(-1)
            probs = dense.softmax(-1)
            for name, mask in [('positive', truth[:, 1] >= config['bin_width'] / 2),
                               ('negative', truth[:, 1] <= -config['bin_width'] / 2),
                               ('zero', truth[:, 1].abs() < config['bin_width'] / 2)]:
                row[name + '_correct_probability'] = (
                    float(probs[mask, 1].gather(1, bins[mask, 1, None]).mean()) if mask.any() else None)
            frame_rows.append(row)
    report = {name: average_scores(rows) for name, rows in buckets.items()}
    report['framewise'] = average_scores(frame_rows)
    report['zero_vy_definition'] = (
        'vy is 0 on every frame and vx is that clip\'s mean ground-truth vx. '
        'The same true initial position and timestamps are integrated. This is a '
        'label baseline for omitted lateral motion, not a network output.'
    )
    return report


def average_scores(rows):
    return {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
            if any(r[key] is not None for r in rows) else None for key in rows[0]}


def evaluate(model, store, ids, normalization, device, config):
    return average_scores([scores(store.get(i), infer(model, store.get(i), normalization,
                                                     device, config), config['bin_width'])
                           for i in ids])


@torch.no_grad()
def framewise_metrics(model, store, ids, normalization, device, config):
    model.eval()
    rows = []
    for clip_id in ids:
        clip = store.get(clip_id)
        features, selected, times, p0, truth = to_device(clip, normalization, device)
        dense, ctc = model(features, selected, p0)
        loss, _, _ = loss_function(dense, ctc, truth, model.centers,
                                   config['sigma'], 0., 0.)
        targets = torch.softmax(-0.5 * ((truth[..., None] - model.centers) / config['sigma']).square(), -1)
        axis_loss = -(targets * dense.log_softmax(-1)).sum(-1).mean(0)
        row = dict(loss=float(loss), vx_loss=float(axis_loss[0]), vy_loss=float(axis_loss[1]))
        bins = (truth[..., None] - model.centers).abs().argmin(-1)
        probs = dense.softmax(-1)
        for name, mask in [('positive', truth[:, 1] >= config['bin_width']/2),
                           ('negative', truth[:, 1] <= -config['bin_width']/2),
                           ('zero', truth[:, 1].abs() < config['bin_width']/2)]:
            row[name + '_correct_probability'] = float(probs[mask, 1].gather(1, bins[mask, 1, None]).mean()) if mask.any() else None
        rows.append(row)
    return average_scores(rows)


# ------------------------- Saved figures, probabilities and CSV -------------------------
def save_example(clip_id, clip, prediction, centers, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', clip_id)
    safe += '_' + hashlib.sha256(clip_id.encode()).hexdigest()[:6]
    destination = directory / safe
    destination.mkdir(parents=True, exist_ok=True)
    times = clip['target_times']
    # Cell edges around the actual frame centers, used only for plotting heatmaps.
    time_edges = np.r_[times[0] - (times[1]-times[0])/2,
                       (times[:-1]+times[1:])/2,
                       times[-1] + (times[-1]-times[-2])/2]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis in range(2):
        ax = axes[0, axis]
        ax.plot(times, clip['position'][:, axis], 'k', label='Ground truth')
        ax.plot(times, prediction['position'][:, axis], label='Decoded + integrated')
        if 'expected_position' in prediction:
            ax.plot(times, prediction['expected_position'][:, axis], color='tab:orange',
                    label='Expected + integrated')
        ax.plot(times, prediction['oracle_position'][:, axis], ':', color='gray', label='Quantized oracle')
        ax.set(title=('x(t)', 'y(t)')[axis], ylabel='Position (m)')
        ax = axes[1, axis]
        ax.plot(times, clip['velocity'][:, axis], 'k', label='Ground truth')
        ax.step(times, prediction['velocity'][:, axis], where='mid', label='Decoded velocity')
        ax.plot(times, prediction['raw_velocity'][:, axis], alpha=.45, label='Raw argmax')
        if 'expected_velocity' in prediction:
            ax.plot(times, prediction['expected_velocity'][:, axis], color='tab:orange',
                    alpha=.8, label='Expected velocity')
        ax.step(times, prediction['oracle_velocity'][:, axis], where='mid', color='gray',
                linestyle=':', label='Quantized oracle')
        ax.set(title=('vx(t)', 'vy(t)')[axis], ylabel='Velocity (m/s)')
    for ax in axes.flat:
        ax.set_xlabel('Observer time (s)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(clip_id)
    fig.tight_layout()
    fig.savefig(destination / 'four_graphs.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    width = centers[1] - centers[0]
    bin_edges = np.r_[centers - width / 2, centers[-1] + width / 2]
    for axis, ax in enumerate(axes):
        mesh = ax.pcolormesh(time_edges, bin_edges, prediction['probabilities'][:, axis].T,
                             shading='flat', vmin=0, vmax=1, cmap='magma')
        ax.plot(times, clip['velocity'][:, axis], 'c-', lw=1.3, label='Ground truth')
        ax.plot(times, prediction['velocity'][:, axis], 'w--', lw=1, label='Decoded')
        ax.set(title=('vx probability distribution', 'vy probability distribution')[axis],
               ylabel='Bin center (m/s)', xlabel='Observer time (s)')
        ax.legend(loc='upper left')
        fig.colorbar(mesh, ax=ax, label='Dense-head probability')
    fig.suptitle(clip_id)
    fig.tight_layout()
    fig.savefig(destination / 'probability_heatmaps.png', dpi=160)
    plt.close(fig)
    np.savez(destination / 'predictions.npz', clip_id=clip_id, times=times,
             centers=centers, true_velocity=clip['velocity'], true_position=clip['position'],
             **prediction)
    with (destination / 'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'vx_true', 'vy_true', 'vx_pred', 'vy_pred',
                         'x_true', 'y_true', 'x_pred', 'y_pred'])
        for j in range(len(times)):
            writer.writerow([times[j], *clip['velocity'][j], *prediction['velocity'][j],
                             *clip['position'][j], *prediction['position'][j]])
    return destination


def run_checkpoint_comparison(paths, records, splits, config, normalization, device, output):
    """Train/val decoder report. The final test split is never loaded."""
    store = ClipStore(records, config, output / 'cache')
    forbidden = splits['test']
    report = {
        'test_clips_scored': 0,
        'position_scale_m': 10.0,
        'position_scale_note': (
            'A residual of 1 in the trajectory loss is a 10 m position error. '
            'The scale is fixed and was not estimated from this dataset.'
        ),
        'checkpoints': {},
    }
    history = output / 'history.csv'
    if history.exists():
        rows = list(csv.DictReader(history.open()))
        jumps = []
        for left, right in zip(rows, rows[1:]):
            jumps.append((float(right['val_rmse']) - float(left['val_rmse']),
                          int(left['epoch']), int(right['epoch']),
                          float(left['val_rmse']), float(right['val_rmse'])))
        jumps.sort(reverse=True)
        report['largest_path_jumps'] = [
            {'from_epoch': a, 'to_epoch': b, 'from_m': c, 'to_m': d, 'increase_m': delta}
            for delta, a, b, c, d in jumps[:5]]
        report['spike_checkpoints'] = (
            'Only best.pt and best_framewise.pt were saved, so the epochs on either '
            'side of the largest path jumps cannot be reloaded.'
        )
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = MotionFramewise(config).to(device)
        model.load_state_dict(checkpoint['model'])
        entry = {'epoch': int(checkpoint['epoch']), 'source': str(path)}
        for part in ('train', 'val'):
            entry[part] = assess_clips(model, store, splits[part], normalization, device,
                                       config, forbidden)
        destination = output / 'decoder_comparison' / f"epoch_{checkpoint['epoch']}"
        centers = model.centers.cpu().numpy()
        for clip_id in splits['val']:
            clip = store.get(clip_id)
            prediction = infer(model, clip, normalization, device, config)
            save_example(clip_id, clip, prediction, centers, destination)
        report['checkpoints'][Path(path).name] = entry
        print(f"Compared {Path(path).name} epoch {checkpoint['epoch']} on train/val only.", flush=True)
    destination = output / 'decoder_comparison'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


# ------------------------- Main: train, select, test, plot -------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', help='Root containing per-clip metadata and WAVs')
    parser.add_argument('--manifest', help='Alternative CSV: id,wav,state,times[,split,group]')
    parser.add_argument('--wav-name', help='Exact WAV path relative to each clip directory')
    parser.add_argument('--out', default='runs/trajectory86')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--accumulate', type=int, default=8, help='Average gradients across clips before each update')
    parser.add_argument('--examples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=96)
    parser.add_argument('--vmax', type=float, default=30.)
    parser.add_argument('--bin-width', type=float, default=2.)
    parser.add_argument('--sigma', type=float, default=1., help='Neighbor-target width in m/s')
    parser.add_argument('--freq-pool', type=int, default=64)
    parser.add_argument('--vy-balance', type=float, default=0.0,
                        help='Blend ordinary and zero/nonzero-balanced vy loss; 0 disables')
    parser.add_argument('--ctc-weight', type=float, default=0.0,
                        help='CTC is off at 0. This architecture has no CTC heads.')
    parser.add_argument('--path-weight', type=float, default=1.0,
                        help='Final weight on the anchored trajectory Huber loss')
    parser.add_argument('--position-scale', type=float, default=10.0,
                        help='Fixed meters per unit residual. Not fit on the data.')
    parser.add_argument('--init-checkpoint', help='Load these weights, then train with a new optimizer')
    parser.add_argument('--defer-test', action='store_true',
                        help='Do not read the final test clips after training')
    parser.add_argument('--compare-checkpoints', nargs='+',
                        help='Score these checkpoints on train and val only, then exit')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--penalty', type=float, default=0.25)
    parser.add_argument('--max-jump', type=float, default=6.,
                        help='Max component velocity change per STFT frame, m/s')
    parser.add_argument('--decoder', choices=('greedy', 'viterbi'), default='viterbi')
    parser.add_argument('--device', default='auto', help='auto, cpu or cuda; CTC MPS not used')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.epochs < 1 or args.examples < 1 or args.patience < 1 or args.threads < 1 or args.accumulate < 1:
        raise ValueError('epochs/examples/patience/threads must be positive.')
    if args.compare_checkpoints:
        config = json.loads((output / 'config.json').read_text())
        normalization = json.loads((output / 'normalization.json').read_text())
        records = read_records(args)
        splits = json.loads((output / 'splits.json').read_text())
        run_checkpoint_comparison(args.compare_checkpoints, records, splits, config,
                                  normalization, device, output)
        return
    records = read_records(args)
    if args.eval_only:
        config = json.loads((output / 'config.json').read_text())
        normalization = json.loads((output / 'normalization.json').read_text())
        # Use saved training config, regardless of CLI architecture/decoder defaults.
    else:
        if (output / 'best.pt').exists():
            raise FileExistsError('Run exists. Use --eval-only or a new --out.')
        keys = ('sample_rate', 'n_fft', 'hop', 'hidden', 'freq_pool', 'vmax', 'bin_width',
                'sigma', 'vy_balance', 'ctc_weight', 'penalty', 'max_jump', 'decoder')
        config = {k: getattr(args, k) for k in keys}
        config['mode'] = 'framewise'
        config['architecture'] = 'pool64_layernorm_residual'
        config['initialized_from'] = (str(Path(args.init_checkpoint).resolve())
                                      if args.init_checkpoint else None)
        config['position_scale_m'] = args.position_scale
        config['position_huber_delta'] = 1.0
        config['path_weight'] = args.path_weight
        config['path_warmup_fraction'] = 0.1
        config['lr'] = args.lr
        if args.init_checkpoint:
            config['optimization_note'] = (
                'Raw lateral predictions fluctuated together with validation path error, '
                'so training resumes from a saved checkpoint at a lower learning rate. '
                'The Viterbi penalty is unchanged.'
            )
        if args.position_scale <= 0 or args.path_weight < 0:
            raise ValueError('Position scale must be positive and path weight cannot be negative.')
        if min(args.sample_rate, args.n_fft, args.hop, args.freq_pool,
               args.hidden, args.vmax, args.bin_width, args.sigma, args.lr) <= 0:
            raise ValueError('Model rates, dimensions, bin settings and LR must be positive.')
        if not math.isclose(args.vmax / args.bin_width, round(args.vmax / args.bin_width)):
            raise ValueError('vmax must be divisible by bin-width to include zero symmetrically.')
        if not 0 <= args.vy_balance <= 1 or min(args.ctc_weight, args.penalty, args.max_jump) < 0:
            raise ValueError('Invalid loss/transition weights.')
        if args.ctc_weight > 0:
            raise ValueError('CTC heads are off. Start this run with --ctc-weight 0.')
    if config.get('mode') != 'framewise':
        raise ValueError('This script requires a framewise run, not a 10 Hz pooled checkpoint.')
    print(f"STFT and prediction rate: {config['sample_rate']/config['hop']:.3f} Hz", flush=True)
    splits = make_splits(records, output, args.seed, args.examples, args.eval_only)
    store = ClipStore(records, config, output / 'cache')
    if not args.eval_only:
        print('Preparing training-only normalization and feature cache...', flush=True)
        normalization = training_normalization(store, splits['train'])
        audit = audit_preprocessing(store, splits, normalization)
        (output / 'normalization.json').write_text(json.dumps(normalization, indent=2))
        (output / 'dataset_audit.json').write_text(json.dumps(audit_dataset(store, splits, config), indent=2))
        (output / 'preprocess_audit.json').write_text(json.dumps(audit, indent=2))
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        (output / 'records.json').write_text(json.dumps(records, indent=2))
        print('Preprocessing audit: encoder inputs are the normalized spectrum and p0; '
              f"normalization uses {audit['normalization_clips']} training clips.", flush=True)
    else:
        old_records = json.loads((output / 'records.json').read_text())
        # Require identical resolved files, not merely equal IDs.
        if sorted(old_records, key=lambda r: r['id']) != sorted(records, key=lambda r: r['id']):
            raise ValueError('Evaluation records differ from the original run.')
    model = MotionFramewise(config).to(device)
    if args.init_checkpoint and not args.eval_only:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(initial['model'])
        print(f"Loaded weights from epoch {initial['epoch']} at lr={args.lr}.", flush=True)
    if not args.eval_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-5)
        best_path, best_fit, stale = math.inf, math.inf, 0
        with (output / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'neighborhood', 'position', 'path_weight', 'ctc',
                                                   'val_rmse', 'val_vx_mae', 'val_vy_mae',
                                                   'val_turning_vy', 'val_false_motion',
                                                   'val_expected_vy', 'val_argmax_vy', 'val_expected_y_rmse'])
            writer.writeheader()
            warmup = max(1, math.ceil(args.epochs * config.get('path_warmup_fraction', 0.1)))
            for epoch in range(1, args.epochs + 1):
                model.train()
                order = list(splits['train'])
                random.shuffle(order)
                losses = []
                weight = config['ctc_weight'] * min(1., epoch / max(1, math.ceil(args.epochs * 0.1)))
                path_weight = config.get('path_weight', 0.) * min(1., epoch / warmup)
                optimizer.zero_grad(set_to_none=True)
                for clip_index, clip_id in enumerate(order):
                    clip = store.get(clip_id)
                    features, selected, times, p0, truth = to_device(clip, normalization, device)
                    truth_position = torch.as_tensor(clip['position'], dtype=torch.float32, device=device)
                    dense, ctc = model(features, selected, p0)
                    loss, neighbor, sequence = loss_function(dense, ctc, truth, model.centers,
                                                            config['sigma'], weight, config['vy_balance'])
                    position_loss = anchored_position_loss(
                        dense, times, p0, truth_position, model.centers,
                        config.get('position_scale_m', 10.), config.get('position_huber_delta', 1.))
                    group_start = (clip_index // args.accumulate) * args.accumulate
                    group_size = min(args.accumulate, len(order) - group_start)
                    ((loss + path_weight * position_loss) / group_size).backward()
                    if (clip_index + 1) % args.accumulate == 0 or clip_index + 1 == len(order):
                        nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    losses.append([loss.item(), neighbor, sequence, position_loss.item(), path_weight])
                used_lr = optimizer.param_groups[0]['lr']
                train_report = assess_clips(model, store, splits['train'], normalization, device,
                                            config, splits['test'])
                val_report = assess_clips(model, store, splits['val'], normalization, device,
                                          config, splits['test'])
                validation = val_report['viterbi']
                val_fit = val_report['framewise']
                scheduler.step(val_fit['loss'])
                with (output / 'diagnostics.jsonl').open('a') as diagnostics:
                    diagnostics.write(json.dumps(dict(
                        epoch=epoch, lr=used_lr, path_weight=path_weight,
                        train_velocity_loss=float(np.mean(losses, axis=0)[1]),
                        train_position_loss=float(np.mean(losses, axis=0)[3]),
                        train=train_report, val=val_report), allow_nan=False) + '\n')
                values = np.mean(losses, axis=0)
                writer.writerow(dict(epoch=epoch, loss=values[0], neighborhood=values[1],
                                     position=values[3], path_weight=values[4], ctc=values[2],
                                     val_rmse=validation['trajectory_rmse'], val_vx_mae=validation['vx_mae'],
                                     val_vy_mae=validation['vy_mae'],
                                     val_turning_vy=validation['turning_vy_mae'],
                                     val_false_motion=validation['zero_false_motion'],
                                     val_expected_vy=val_report['expected']['vy_mae'],
                                     val_argmax_vy=val_report['argmax']['vy_mae'],
                                     val_expected_y_rmse=val_report['expected']['y_rmse']))
                f.flush()
                print(f'Epoch {epoch:03d}: vel={values[1]:.3f}, pos={values[3]:.3f}, '
                      f'path_w={path_weight:.2f}, val CE y={val_fit["vy_loss"]:.3f}, '
                      f'val vy expected/argmax/viterbi='
                      f'{val_report["expected"]["vy_mae"]:.3f}/'
                      f'{val_report["argmax"]["vy_mae"]:.3f}/'
                      f'{validation["vy_mae"]:.3f}, '
                      f'val y rmse expected/viterbi='
                      f'{val_report["expected"]["y_rmse"]:.2f}/'
                      f'{validation["y_rmse"]:.2f}', flush=True)
                if validation['trajectory_rmse'] < best_path:
                    best_path = validation['trajectory_rmse']
                    torch.save({'model': model.state_dict(), 'config': config, 'normalization': normalization,
                                'epoch': epoch, 'validation': validation}, output / 'best.pt')
                fit_score = val_fit['loss']
                if fit_score < best_fit - 1e-4:
                    best_fit, stale = fit_score, 0
                    torch.save({'model': model.state_dict(), 'config': config, 'normalization': normalization,
                                'epoch': epoch, 'validation': validation}, output / 'best_framewise.pt')
                else:
                    stale += 1
                if stale >= args.patience:
                    print('Early stopping on validation framewise CE.', flush=True)
                    break
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    if args.defer_test:
        summary = {'checkpoint_epoch': checkpoint['epoch'], 'test_evaluated': False,
                   'validation': checkpoint['validation']}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(json.dumps(summary, indent=2), flush=True)
        return
    centers = model.centers.cpu().numpy()
    rows = []
    for clip_id in splits['examples']:
        clip = store.get(clip_id)
        prediction = infer(model, clip, normalization, device, config)
        result = scores(clip, prediction, config['bin_width'])
        directory = save_example(clip_id, clip, prediction, centers, output / 'held_out')
        rows.append({'clip_id': clip_id, **result})
        print(f'Held out {clip_id}: RMSE={result["trajectory_rmse"]:.3f} m -> {directory}', flush=True)
    with (output / 'held_out_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean = average_scores([{k: v for k, v in r.items() if k != 'clip_id'} for r in rows])
    summary = {'checkpoint_epoch': checkpoint['epoch'], 'examples': len(rows),
               'mean_per_clip': mean, 'validation': checkpoint['validation']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
