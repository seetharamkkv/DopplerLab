"""Lightweight STFT for physics feature extraction on IDMT clips."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from idmt_experiments.config import HOP_LENGTH, N_FFT, WIN_LENGTH

FMAX_HZ = 8000.0


@dataclass
class StftBundle:
    power: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    t_peak_s: float


def clip_center_time_s(y: np.ndarray, sr: int) -> float:
    """Fixed clip midpoint — invariant under time-reverse (same index in reversed buffer)."""
    return float(len(y) / (2.0 * sr))


def envelope_peak_time_s(y: np.ndarray, sr: int, hop: int = HOP_LENGTH) -> float:
    frame = hop
    n_frames = 1 + max(0, (len(y) - frame) // hop)
    if n_frames <= 0:
        return 0.0
    env = np.array(
        [np.sqrt(np.mean(y[i * hop : i * hop + frame] ** 2) + 1e-12) for i in range(n_frames)]
    )
    peak_idx = int(np.argmax(env))
    return float(peak_idx * hop / sr)


def compute_stft_bundle(y: np.ndarray, sr: int, *, t_peak_s: float | None = None) -> StftBundle:
    if t_peak_s is None:
        t_peak_s = envelope_peak_time_s(y, sr)
    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        center=True,
    )
    power = np.abs(stft) ** 2
    times = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    keep = freqs <= FMAX_HZ
    return StftBundle(
        power=power[keep],
        freqs=freqs[keep],
        times=times - t_peak_s,
        t_peak_s=t_peak_s,
    )


def load_mono_for_physics(wav_path, mono_source: str = "left") -> tuple[np.ndarray, int]:
    from idmt_experiments.src.features import load_stereo, select_mono_waveform

    if mono_source not in ("left", "right"):
        raise ValueError(
            f"Physics track requires mono_source left|right (no downmix); got {mono_source!r}"
        )
    y, sr = load_stereo(wav_path)
    y = select_mono_waveform(y, mono_source)
    return y.astype(np.float64), sr
