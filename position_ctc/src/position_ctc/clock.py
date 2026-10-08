"""Shared 100 Hz clock for audio features and kinematic labels.

DopplerSim's exported STFT is magnitude-only at hop 512 (about 43 Hz). This
track recomputes a complex STFT from the WAV so tokens, velocity, and position
share one grid.
"""

from __future__ import annotations

import numpy as np

SR = 22050
HOP = 220
N_FFT = 2048
DT = HOP / SR


def n_frames(n_samples: int) -> int:
    """Frame count for a centered STFT, matching librosa's ``center=True``."""
    if n_samples < 1:
        raise ValueError("n_samples")
    return 1 + int(n_samples) // HOP


def frame_times(n_samples: int, t0: float = 0.0) -> np.ndarray:
    """Center time of each STFT frame, in seconds."""
    return t0 + np.arange(n_frames(n_samples), dtype=np.float64) * DT


def resample_series(values: np.ndarray, times_s: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation of a state array onto ``t_dst``.

    ``values`` is ``(N,)`` or ``(N, C)``. Samples outside ``times_s`` clamp to
    the end values.
    """
    src = np.asarray(values, dtype=np.float64)
    times_s = np.asarray(times_s, dtype=np.float64).reshape(-1)
    t_dst = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    if src.shape[0] != times_s.shape[0]:
        raise ValueError("values and times_s length mismatch")
    if src.ndim == 1:
        return np.interp(t_dst, times_s, src)
    cols = [np.interp(t_dst, times_s, src[:, j]) for j in range(src.shape[1])]
    return np.stack(cols, axis=1)
