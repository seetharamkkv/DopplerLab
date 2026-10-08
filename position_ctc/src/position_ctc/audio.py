"""Complex STFT on the position-CTC clock.

Phase is stored as sine and cosine. The saved DopplerSim ``stft.npy`` is
magnitude in dB and is not an input to this track.
"""

from __future__ import annotations

import numpy as np

from position_ctc.clock import HOP, N_FFT, SR, frame_times


def complex_stft(y: np.ndarray, *, sr: int = SR) -> dict[str, np.ndarray]:
    """Magnitude, sin(phase), and cos(phase) at 100 Hz.

    Returns arrays shaped ``(F, T)`` plus ``times_s`` of length ``T``.
    """
    if int(sr) != SR:
        raise ValueError(f"expected {SR} Hz audio, got {sr}")
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size < HOP:
        raise ValueError("audio shorter than one hop")
    pad = N_FFT // 2
    padded = np.pad(y, (pad, pad), mode="reflect")
    window = np.hanning(N_FFT)
    times = frame_times(y.size)
    n_freq = N_FFT // 2 + 1
    spec = np.empty((n_freq, times.size), dtype=np.complex128)
    for i in range(times.size):
        start = i * HOP
        frame = padded[start : start + N_FFT]
        if frame.size < N_FFT:
            frame = np.pad(frame, (0, N_FFT - frame.size))
        spec[:, i] = np.fft.rfft(frame * window)
    phase = np.angle(spec)
    return {
        "magnitude": np.abs(spec).astype(np.float32),
        "phase_sin": np.sin(phase).astype(np.float32),
        "phase_cos": np.cos(phase).astype(np.float32),
        "times_s": times,
    }
