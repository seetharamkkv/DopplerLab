"""STFT, reassigned, synchrosqueezed, and mel spectrograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np

from length_estimation.config import (
    FMAX_HZ,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SR,
    SSQ_N_SCALES,
    SSQ_SCALE_MAX,
    SSQ_SCALE_MIN,
    SSQ_TARGET_SR,
    WINDOW,
    StftConfig,
)


@dataclass
class ReassignedAtoms:
    times: np.ndarray
    freqs: np.ndarray
    mags: np.ndarray
    sr: int
    n_fft: int
    hop_length: int
    window: str


@dataclass
class SpectrogramBundle:
    t_rel: np.ndarray
    stft_power: np.ndarray
    stft_freqs: np.ndarray
    stft_times: np.ndarray
    reassigned: ReassignedAtoms | None = None
    ssq_power: np.ndarray | None = None
    ssq_freqs: np.ndarray | None = None
    ssq_times: np.ndarray | None = None
    mel: np.ndarray | None = None
    mel_freqs: np.ndarray | None = None
    mel_times: np.ndarray | None = None


def compute_stft_power(
    y: np.ndarray,
    sr: int,
    cfg: StftConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or StftConfig()
    stft = librosa.stft(
        y,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
        center=True,
    )
    power = np.abs(stft) ** 2
    times = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=cfg.hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)
    keep = freqs <= cfg.fmax_hz
    return power[keep], freqs[keep], times


def compute_reassigned_atoms(
    y: np.ndarray,
    sr: int,
    cfg: StftConfig | None = None,
    fill_nan: bool = True,
) -> ReassignedAtoms:
    cfg = cfg or StftConfig()
    freqs, times, mags = librosa.reassigned_spectrogram(
        y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
        fill_nan=fill_nan,
    )
    freq_mask = freqs <= cfg.fmax_hz
    mags = np.where(freq_mask, mags, 0.0)
    return ReassignedAtoms(
        times=np.asarray(times),
        freqs=np.asarray(freqs),
        mags=np.asarray(mags),
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
    )


def compute_synchrosqueezed(
    y: np.ndarray,
    sr: int,
    fmax_hz: float = FMAX_HZ,
    target_sr: int = SSQ_TARGET_SR,
    n_scales: int = SSQ_N_SCALES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ssqueezepy import ssq_cwt

    if sr != target_sr:
        y_ssq = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    else:
        y_ssq = y

    scales = np.geomspace(SSQ_SCALE_MIN, SSQ_SCALE_MAX, n_scales)
    tx, _, ssq_freqs, *_ = ssq_cwt(y_ssq, wavelet="morlet", fs=target_sr, scales=scales)
    power = np.abs(tx) ** 2
    keep = ssq_freqs <= fmax_hz
    power = power[keep]
    ssq_freqs = ssq_freqs[keep]
    times = np.linspace(0, len(y_ssq) / target_sr, power.shape[1], endpoint=False)
    return power, ssq_freqs, times


def compute_log_mel(
    y: np.ndarray,
    sr: int,
    cfg: StftConfig | None = None,
    n_mels: int = N_MELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or StftConfig()
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
        n_mels=n_mels,
        fmax=cfg.fmax_hz,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    times = librosa.frames_to_time(np.arange(mel.shape[1]), sr=sr, hop_length=cfg.hop_length)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=cfg.fmax_hz)
    return mel_db, mel_freqs, times


def build_spectrogram_bundle(
    y: np.ndarray,
    t_rel: np.ndarray,
    sr: int = SR,
    cfg: StftConfig | None = None,
    *,
    include_reassigned: bool = True,
    include_ssq: bool = True,
    include_mel: bool = True,
) -> SpectrogramBundle:
    cfg = cfg or StftConfig()
    power, freqs, stft_times = compute_stft_power(y, sr, cfg)

    bundle = SpectrogramBundle(
        t_rel=t_rel,
        stft_power=power,
        stft_freqs=freqs,
        stft_times=stft_times,
    )

    if include_reassigned:
        bundle.reassigned = compute_reassigned_atoms(y, sr, cfg)

    if include_ssq:
        ssq_power, ssq_freqs, ssq_times = compute_synchrosqueezed(y, sr, cfg.fmax_hz)
        bundle.ssq_power = ssq_power
        bundle.ssq_freqs = ssq_freqs
        bundle.ssq_times = ssq_times

    if include_mel:
        mel, mel_freqs, mel_times = compute_log_mel(y, sr, cfg)
        bundle.mel = mel
        bundle.mel_freqs = mel_freqs
        bundle.mel_times = mel_times

    return bundle


def power_to_db(power: np.ndarray, ref: float | None = None) -> np.ndarray:
    ref = ref if ref is not None else np.max(power)
    return librosa.power_to_db(power, ref=ref)


def interpolate_times_to_grid(values: np.ndarray, times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolate 1D series defined on `times` onto `grid`."""
    if len(times) == 0:
        return np.zeros_like(grid)
    return np.interp(grid, times, values, left=values[0], right=values[-1])
