"""Numpy-only split validation, Viterbi decoding, integration and metrics."""
import json
from pathlib import Path
import numpy as np


def discover(root):
    root = Path(root)
    clips = sorted(p for p in root.glob('sample_*') if p.is_dir())
    if not clips:
        raise ValueError(f'No sample_* directories found in {root}')
    return {p.name: p for p in clips}


def read_split(path, available):
    """Accept explicit sample names/paths, never reinterpret integer indices."""
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and 'splits' in raw:
        raw = raw['splits']
    if not isinstance(raw, dict):
        raise ValueError('Split JSON must be an object containing train/val/test lists')
    result = {}
    for key in ('train', 'val', 'test'):
        aliases = (key, 'valid', 'validation') if key == 'val' else (key,)
        entries = next((raw[k] for k in aliases if k in raw), None)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f'Split must contain a nonempty {key} list of sample names/paths. '
                             'Convert index-based splits using the ORIGINAL sample ordering.')
        names = []
        for item in entries:
            if isinstance(item, dict):
                item = next((item[k] for k in ('sample_id', 'id', 'sample', 'path', 'clip')
                             if k in item), None)
            if not isinstance(item, str):
                raise ValueError('Only explicit sample names/paths are accepted; no integer indices.')
            candidates = [p for p in item.replace('\\', '/').split('/') if p in available]
            if len(set(candidates)) != 1:
                raise ValueError(f'Cannot resolve split entry: {item!r}')
            names.append(candidates[0])
        if len(set(names)) != len(names):
            raise ValueError(f'Duplicate samples in {key}')
        result[key] = names
    groups = [set(result[k]) for k in ('train', 'val', 'test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Train/validation/test overlap detected')
    return result


def velocity_bins(vmin, vmax, width):
    n = (vmax - vmin) / width if width > 0 else -1
    if n < 1 or not np.isclose(n, round(n)):
        raise ValueError('Positive bin width must divide vmax-vmin exactly')
    return np.linspace(vmin, vmax, round(n) + 1, dtype=np.float32)


def viterbi(log_probs, bins, penalty=0.05, max_jump=8):
    """Independent axis MAP chain; quadratic m/s transition cost within a bin band.

    penalty has units inverse (m/s)^2 per transition. max_jump is in bins per
    LABEL frame, not acceleration. Set penalty=0 for framewise argmax without a
    transition constraint. This is a documented replacement, not legacy decoding.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.ndim != 2 or len(log_probs) < 1 or log_probs.shape[1] != len(bins):
        raise ValueError('Expected nonempty [time, number_of_bins] scores')
    if penalty < 0 or max_jump < 0:
        raise ValueError('Viterbi penalty and max_jump must be nonnegative')
    if penalty == 0:
        return np.asarray(bins)[log_probs.argmax(-1)]
    t, k = log_probs.shape
    band = min(max_jump, k - 1)
    offsets = np.arange(-band, band + 1)
    current = np.arange(k)[:, None]
    previous = current + offsets[None, :]
    valid = (previous >= 0) & (previous < k)
    previous = previous.clip(0, k - 1)
    cost = penalty * (np.asarray(bins)[current] - np.asarray(bins)[previous]) ** 2
    score = log_probs[0].copy()
    back = np.zeros((t, k), dtype=np.int32)
    for i in range(1, t):
        candidates = np.where(valid, score[previous] - cost, -np.inf)
        best = candidates.argmax(1)
        back[i] = previous[np.arange(k), best]
        score = candidates[np.arange(k), best] + log_probs[i]
        score -= score.max()
    states = np.empty(t, dtype=np.int32)
    states[-1] = score.argmax()
    for i in range(t - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    return np.asarray(bins)[states]


def integrate(velocity, times, start):
    velocity, times = np.asarray(velocity), np.asarray(times)
    if len(times) != len(velocity) or np.any(np.diff(times) <= 0):
        raise ValueError('Velocity/time lengths must match and times must increase')
    position = np.empty_like(velocity, dtype=np.float64)
    position[0] = start
    position[1:] = start + np.cumsum(
        0.5 * (velocity[1:] + velocity[:-1]) * np.diff(times)[:, None], axis=0)
    return position


def orbit_align(pred, target, mask):
    """One orthogonal transform about the origin; reflection allowed, no translation."""
    u, _, vt = np.linalg.svd(pred[mask].T @ target[mask])
    return pred @ (u @ vt)


def metrics(pred, target, velocity, true_velocity, times, mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError('No moving frames to score')
    errors = np.linalg.norm(pred - target, axis=1)
    aligned = orbit_align(pred, target, mask)
    orbit_errors = np.linalg.norm(aligned - target, axis=1)
    true_range, pred_range = np.linalg.norm(target, axis=1), np.linalg.norm(pred, axis=1)
    indices = np.flatnonzero(mask)
    heading = np.arctan2(true_velocity[:, 1], true_velocity[:, 0])
    delta = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))
    turn_idx = np.flatnonzero((np.abs(delta) >= np.deg2rad(12)) & mask[1:] & mask[:-1]) + 1
    turn_mask = np.zeros(len(times), dtype=bool)
    for i in turn_idx:
        turn_mask |= np.abs(times - times[i]) <= 0.25
    turn_mask &= mask
    return {
        'ade_m': float(errors[mask].mean()),
        'fde_m': float(errors[indices[-1]]),
        'orbit_ade_m': float(orbit_errors[mask].mean()),
        'orbit_rms_m': float(np.sqrt(np.mean(orbit_errors[mask] ** 2))),
        'range_mae_m': float(np.abs(pred_range[mask] - true_range[mask]).mean()),
        'cpa_distance_error_m': float(abs(pred_range[mask].min() - true_range[mask].min())),
        'vx_mae_mps': float(np.abs(velocity[mask, 0] - true_velocity[mask, 0]).mean()),
        'vy_mae_mps': float(np.abs(velocity[mask, 1] - true_velocity[mask, 1]).mean()),
        'turn_ade_m': float(errors[turn_mask].mean()) if turn_mask.any() else None,
        'moving_frames': int(mask.sum()), 'turn_frames': int(turn_mask.sum()),
    }
