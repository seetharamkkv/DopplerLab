import numpy as np
import torch
from tqdm import tqdm
from .core import viterbi, integrate, metrics


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Apple MPS has no float64. Frame times are the only float64 field.
            if value.dtype == torch.float64:
                value = value.float()
            value = value.to(device)
        moved[key] = value
    return moved


def aggregate(rows):
    keys = [k for k in rows[0] if k != 'sample']
    result = {'clips': len(rows)}
    for k in keys:
        values = [r[k] for r in rows if r[k] is not None]
        if k in ('moving_frames', 'turn_frames'):
            result[k] = int(sum(values))
        else:
            result[k] = float(np.mean(values)) if values else None
    return result


@torch.no_grad()
def evaluate(model, loader, bins, device, penalty, max_jump, callback=None, description='Evaluate'):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc=description):
        features = batch['features'].to(device)
        logits = model(features, batch['lengths'].to(device))
        probabilities = logits.log_softmax(-1).cpu().numpy()
        for i, name in enumerate(batch['ids']):
            n = int(batch['lengths'][i])
            velocity = np.stack([viterbi(probabilities[i, :n, axis], bins, penalty, max_jump)
                                 for axis in range(2)], axis=-1)
            times = batch['times'][i, :n].numpy()
            target = batch['position'][i, :n].numpy()
            truth_velocity = batch['velocity'][i, :n].numpy()
            mask = batch['moving'][i, :n].numpy()
            pred = integrate(velocity, times, target[0])
            row = {'sample': name, **metrics(pred, target, velocity, truth_velocity, times, mask)}
            rows.append(row)
            if callback:
                callback(name, times, pred, target, velocity, truth_velocity, mask, row)
    if not rows:
        raise ValueError('Empty evaluation split')
    return aggregate(rows), rows
