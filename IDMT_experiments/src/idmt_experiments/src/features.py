"""Audio feature extraction for IDMT direction models.

REPRODUCIBILITY BASELINE — shared CNN dependency (mel_3class / mel_3class_left / mel_3class_right)
---------------------------------------------------------------------------------
``compute_log_mel``, ``select_mono_waveform``, ``extract_feature``, ``fit_norm_stats``, and
``normalize_feature`` are on the CNN baseline critical path. Do not change default behaviour,
numerics, or evaluation outputs without re-benchmarking all three reference runs.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from idmt_experiments.config import (
    CC_BLOCK_S,
    CC_HOP_S,
    CC_MARGIN,
    FMAX_HZ,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SR_MEL,
    SR_NATIVE,
    WIN_LENGTH,
    NormStats,
)


def load_stereo(path, sr: int = SR_NATIVE) -> tuple[np.ndarray, int]:
    y, file_sr = librosa.load(path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    return y.astype(np.float32), file_sr


def load_mono(path, sr: int = SR_NATIVE, *, input_type: str | None = None) -> tuple[np.ndarray, int]:
    """Load a single-channel waveform (downmix if the file is stereo).

    input_type:
      - ``audio`` (default): read audio files (.wav, etc.) with librosa.
      - ``video``: extract the audio track with ffmpeg (full clip, no trim).

    If input_type is omitted, video containers are detected by suffix.
    """
    from idmt_experiments.src.media_io import is_video_path, load_mono_ffmpeg

    path = Path(path)
    use_ffmpeg = input_type == "video" or (input_type is None and is_video_path(path))

    if use_ffmpeg:
        return load_mono_ffmpeg(path, sr=sr)

    try:
        y, file_sr = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32), file_sr
    except Exception as exc:
        if input_type == "audio":
            raise RuntimeError(f"Could not load audio file {path.name}: {exc}") from exc
        return load_mono_ffmpeg(path, sr=sr)


def swap_stereo_channels(y: np.ndarray) -> np.ndarray:
    if y.ndim != 2 or y.shape[0] < 2:
        return y
    out = y.copy()
    out[0], out[1] = y[1], y[0]
    return out


def reverse_time_axis(y: np.ndarray) -> np.ndarray:
    """Time-reverse waveform (same mono channel(s); no stereo fusion)."""
    if y.ndim == 1:
        return y[::-1].copy()
    return y[:, ::-1].copy()


def select_mono_waveform(y: np.ndarray, mono_source: str = "mean") -> np.ndarray:
    """Pick one channel for mel when the file is stereo (no metadata, waveform only).

    mean: (L+R)/2 — prior default, can encode inter-channel level asymmetry.
    left / right: single microphone channel from the stereo pair — true monaural path.
    """
    if y.ndim == 1:
        return y
    if mono_source == "mean":
        return np.mean(y, axis=0)
    if mono_source == "left":
        return y[0]
    if mono_source == "right":
        return y[1]
    raise ValueError(f"mono_source must be mean|left|right, got {mono_source!r}")


def compute_log_mel(
    y: np.ndarray,
    sr: int,
    *,
    n_mels: int = N_MELS,
    mono_source: str = "mean",
) -> np.ndarray:
    if y.ndim == 2:
        y_mono = select_mono_waveform(y, mono_source)
    else:
        y_mono = y
    if sr != SR_MEL:
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=SR_MEL)
        sr = SR_MEL
    mel = librosa.feature.melspectrogram(
        y=y_mono,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=n_mels,
        power=2.0,
    )
    return np.log1p(mel).astype(np.float32)


def compute_stereo_mel(y: np.ndarray, sr: int, *, n_mels: int = N_MELS) -> np.ndarray:
    if sr != SR_MEL:
        y = np.stack([librosa.resample(y[i], orig_sr=sr, target_sr=SR_MEL) for i in range(y.shape[0])])
        sr = SR_MEL
    channels = []
    for ch in range(y.shape[0]):
        mel = librosa.feature.melspectrogram(
            y=y[ch],
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=n_mels,
            power=2.0,
        )
        channels.append(np.log1p(mel))
    return np.stack(channels, axis=0).astype(np.float32)


def compute_complex_stft(
    y_mono: np.ndarray,
    sr: int,
    *,
    fmax_hz: float = FMAX_HZ,
) -> np.ndarray:
    """Complex STFT from a **single-channel** waveform (1-D only).

    Pipeline:
    1. Complex STFT on mono audio (never stereo).
    2. Crop to bins <= fmax_hz.
    3. Scale by peak magnitude so |z| lies in [0, 1] (phase preserved).
    4. Stack real and imaginary -> shape (2, F, T).

    The two output channels are **spectral** (Re/Im), not left/right microphones.
  """
    if y_mono.ndim != 1:
        raise ValueError(
            f"compute_complex_stft requires mono audio (ndim=1); got shape {y_mono.shape}. "
            "Collapse stereo with select_mono_waveform() first."
        )
    if sr != SR_MEL:
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=SR_MEL)
        sr = SR_MEL
    stft = librosa.stft(
        y_mono,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        center=True,
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    stft = stft[freqs <= fmax_hz]
    mag_max = float(np.max(np.abs(stft)))
    if mag_max < 1e-12:
        mag_max = 1.0
    stft_scaled = stft / mag_max
    real = stft_scaled.real.astype(np.float32)
    imag = stft_scaled.imag.astype(np.float32)
    return np.stack([real, imag], axis=0)


def _apply_cpx_bin_norm(feat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Complex STFT: (2, freq, time) — per-channel per-frequency z-score."""
    std_safe = np.where(std < 1e-8, 1.0, std)
    return ((feat - mean[:, :, None]) / std_safe[:, :, None]).astype(np.float32)


def compute_cc_stack(y: np.ndarray, sr: int = SR_NATIVE) -> np.ndarray:
    """Local cross-correlation blocks -> (n_blocks, n_lags)."""
    left, right = y[0], y[1]
    block = int(round(CC_BLOCK_S * sr))
    hop = int(round(CC_HOP_S * sr))
    n_lags = 2 * CC_MARGIN + 1
    blocks: list[np.ndarray] = []
    start = 0
    while start + block <= len(left):
        l_blk = left[start : start + block]
        r_blk = right[start : start + block]
        l_blk = l_blk - np.mean(l_blk)
        r_blk = r_blk - np.mean(r_blk)
        cc = np.correlate(l_blk, r_blk, mode="full")
        center = len(cc) // 2
        seg = cc[center - CC_MARGIN : center + CC_MARGIN + 1]
        if len(seg) != n_lags:
            seg = np.pad(seg, (0, n_lags - len(seg)))[:n_lags]
        blocks.append(seg.astype(np.float32))
        start += hop
    if not blocks:
        return np.zeros((1, n_lags), dtype=np.float32)
    return np.stack(blocks, axis=0)


def _apply_mel_bin_norm(feat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Mel/ssq: (freq, time) — normalize per frequency bin."""
    std_safe = np.where(std < 1e-8, 1.0, std)
    if feat.ndim == 2:
        return ((feat - mean[:, None]) / std_safe[:, None]).astype(np.float32)
    if feat.ndim == 3:
        return ((feat - mean[None, :, None]) / std_safe[None, :, None]).astype(np.float32)
    raise ValueError(f"Unexpected mel feature shape {feat.shape}")


def _apply_cc_bin_norm(feat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """CC stack: (time_blocks, lags) — normalize per lag bin (EUSIPCO paper)."""
    std_safe = np.where(std < 1e-8, 1.0, std)
    return ((feat - mean[None, :]) / std_safe[None, :]).astype(np.float32)


def normalize_feature(feat: np.ndarray, stats: NormStats | None, feature_type: str) -> np.ndarray:
    if stats is None:
        return per_sample_zscore(feat, feature_type=feature_type)
    if feature_type in ("mel", "stereo_mel") and stats.mel_mean and stats.mel_std:
        mean = np.array(stats.mel_mean, dtype=np.float32)
        std = np.array(stats.mel_std, dtype=np.float32)
        return _apply_mel_bin_norm(feat, mean, std)
    if feature_type == "complex_stft" and stats.cpx_mean and stats.cpx_std:
        mean = np.array(stats.cpx_mean, dtype=np.float32)
        std = np.array(stats.cpx_std, dtype=np.float32)
        return _apply_cpx_bin_norm(feat, mean, std)
    if feature_type == "cc" and stats.cc_mean and stats.cc_std:
        mean = np.array(stats.cc_mean, dtype=np.float32)
        std = np.array(stats.cc_std, dtype=np.float32)
        return _apply_cc_bin_norm(feat, mean, std)
    return per_sample_zscore(feat, feature_type=feature_type)


def per_sample_zscore(feat: np.ndarray, *, feature_type: str = "mel") -> np.ndarray:
    if feat.ndim == 2:
        if feature_type == "cc":
            mu = feat.mean(axis=0, keepdims=True)
            sd = feat.std(axis=0, keepdims=True)
        else:
            mu = feat.mean(axis=1, keepdims=True)
            sd = feat.std(axis=1, keepdims=True)
    else:
        mu = feat.mean(axis=(1, 2), keepdims=True)
        sd = feat.std(axis=(1, 2), keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((feat - mu) / sd).astype(np.float32)


def feature_to_batch_tensor(feat: np.ndarray):
    """Convert numpy feature to batched CNN input (1, C, F, T)."""
    torch = __import__("torch")
    if feat.ndim == 2:
        return torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
    return torch.from_numpy(feat).unsqueeze(0).float()


def extract_feature(
    y: np.ndarray,
    sr: int,
    feature_type: str,
    *,
    n_mels: int = N_MELS,
    mono_source: str = "mean",
) -> np.ndarray:
    if feature_type == "mel":
        return compute_log_mel(y, sr, n_mels=n_mels, mono_source=mono_source)
    if feature_type == "stereo_mel":
        return compute_stereo_mel(y, sr, n_mels=n_mels)
    if feature_type == "complex_stft":
        if y.ndim == 2:
            y = select_mono_waveform(y, mono_source)
        return compute_complex_stft(y, sr)
    if feature_type == "cc":
        return compute_cc_stack(y, sr)
    raise ValueError(f"Unknown feature_type: {feature_type}")


def fit_norm_stats(
    records,
    cfg,
    max_samples: int | None = None,
    *,
    show_progress: bool = False,
) -> NormStats:
    """Fit per-bin mean/std on training clips only."""
    from tqdm import tqdm

    if max_samples is None:
        max_samples = getattr(cfg, "norm_fit_max_samples", None)

    mel_bins: list[np.ndarray] = []
    cc_bins: list[np.ndarray] = []
    cpx_ch_bins: list[list[np.ndarray]] = [[], []]
    n = 0
    cap = min(len(records), max_samples) if max_samples is not None else len(records)
    it = tqdm(records, total=cap, desc="norm stats", leave=True) if show_progress else records
    for rec in it:
        if max_samples is not None and n >= max_samples:
            break
        y, sr = load_stereo(rec.wav_path)
        feat = extract_feature(
            y, sr, cfg.feature_type, n_mels=cfg.n_mels, mono_source=getattr(cfg, "mono_source", "mean")
        )
        if cfg.feature_type == "cc":
            cc_bins.append(feat)
        elif cfg.feature_type in ("mel", "stereo_mel"):
            if feat.ndim == 3:
                for ch in range(feat.shape[0]):
                    mel_bins.append(feat[ch])
            else:
                mel_bins.append(feat)
        elif cfg.feature_type == "complex_stft":
            for ch in range(feat.shape[0]):
                cpx_ch_bins[ch].append(feat[ch])
        n += 1

    stats = NormStats()
    if mel_bins:
        stacked = np.stack(mel_bins, axis=0)
        stats.mel_mean = stacked.mean(axis=(0, 2)).tolist()
        stats.mel_std = stacked.std(axis=(0, 2)).tolist()
    if cc_bins:
        stacked = np.stack(cc_bins, axis=0)
        stats.cc_mean = stacked.mean(axis=(0, 1)).tolist()
        stats.cc_std = stacked.std(axis=(0, 1)).tolist()
    if cpx_ch_bins[0]:
        stats.cpx_mean = []
        stats.cpx_std = []
        for ch_bins in cpx_ch_bins:
            stacked = np.stack(ch_bins, axis=0)
            stats.cpx_mean.append(stacked.mean(axis=(0, 2)).tolist())
            stats.cpx_std.append(stacked.std(axis=(0, 2)).tolist())
    return stats
