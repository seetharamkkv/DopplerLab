"""Complex STFT features on the DopplerSim frame clock of this batch.

The exported clips are 44.1 kHz with hop 512, which is exactly the
``state_frames`` grid (about 86 Hz). Staying on that grid keeps every
velocity label on the same column as the spectrum. Phase is a
magnitude-weighted sine and cosine inside each mel band.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

DATA_SR = 44100
DATA_HOP = 512
DATA_N_FFT = 2048
N_MELS = 48
FMIN = 40.0
FMAX = 4000.0


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{path} is not 16-bit PCM")
    y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        y = y.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(y), sr


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(freq: float) -> float:
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(FMIN), hz_to_mel(min(FMAX, sr / 2 - 1)), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        center = max(center, left + 1)
        right = max(right, center + 1)
        for j in range(left, min(center, fb.shape[1])):
            if j >= 0:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, min(right, fb.shape[1])):
            if j >= 0:
                fb[i, j] = (right - j) / (right - center)
    return fb.astype(np.float32)


_FB: np.ndarray | None = None


def mel_filterbank() -> np.ndarray:
    global _FB
    if _FB is None:
        _FB = _mel_filterbank(DATA_SR, DATA_N_FFT, N_MELS)
    return _FB


def complex_stft_native(y: np.ndarray) -> np.ndarray:
    """Centered complex STFT, shape ``(F, T)``, matching ``1 + n // hop`` frames."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    pad = DATA_N_FFT // 2
    padded = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + y.size // DATA_HOP
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, DATA_N_FFT),
        strides=(padded.strides[0] * DATA_HOP, padded.strides[0]),
    )
    window = np.hanning(DATA_N_FFT).astype(np.float32)
    spec = np.fft.rfft(np.ascontiguousarray(frames) * window, axis=1)
    return spec.T.astype(np.complex64)


def features_from_wav(path: Path) -> np.ndarray:
    """Return ``(3 * N_MELS, T)`` log-mel, sin phase, cos phase."""
    y, sr = read_wav(path)
    if sr != DATA_SR:
        raise ValueError(f"{path} sample rate {sr}, expected {DATA_SR}")
    spec = complex_stft_native(y)
    power = np.abs(spec).astype(np.float32) ** 2
    phase = np.angle(spec).astype(np.float32)
    fb = mel_filterbank()
    mel = fb @ power
    mel_sin = fb @ (power * np.sin(phase))
    mel_cos = fb @ (power * np.cos(phase))
    denom = np.maximum(mel, 1e-8)
    log_mag = np.log(mel + 1e-6).astype(np.float32)
    return np.concatenate(
        [log_mag, (mel_sin / denom).astype(np.float32), (mel_cos / denom).astype(np.float32)],
        axis=0,
    )
