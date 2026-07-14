"""SpecAugment for log-mel tensors (opt-in; baseline training does not use this)."""

from __future__ import annotations

import random


def apply_spec_augment(
    x,
    *,
    time_mask_param: int = 20,
    freq_mask_param: int = 8,
    num_time_masks: int = 2,
    num_freq_masks: int = 2,
):
    """Mask random time/frequency bands on mel tensor ``(C, n_mels, T)``."""
    x = x.clone()
    if x.ndim != 3:
        return x
    _, n_mels, n_time = x.shape
    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_param, n_mels))
        if f <= 0:
            continue
        f0 = random.randint(0, max(0, n_mels - f))
        x[:, f0 : f0 + f, :] = 0.0
    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, n_time))
        if t <= 0:
            continue
        t0 = random.randint(0, max(0, n_time - t))
        x[:, :, t0 : t0 + t] = 0.0
    return x
