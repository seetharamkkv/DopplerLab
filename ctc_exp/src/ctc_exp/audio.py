"""Log-mel front-end. Sim prefers exported log_mel.npy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128
CROP_HALF_S = 4.0


def frame_times(n_frames: int, t0: float = 0.0) -> np.ndarray:
    return t0 + (np.arange(n_frames, dtype=np.float64) * HOP) / SR


def load_log_mel(
    wav_path: Path,
    log_mel_path: Path | None = None,
    *,
    cpa_s: float | None = None,
    crop_around_cpa: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(n_mels, T)`` log-mel and per-frame times in seconds."""
    if log_mel_path is not None and log_mel_path.is_file() and not crop_around_cpa:
        mel = np.asarray(np.load(log_mel_path), dtype=np.float32)
        if mel.ndim != 2:
            raise ValueError(f"bad log-mel {log_mel_path}: {mel.shape}")
        return mel, frame_times(mel.shape[1], 0.0)

    import librosa

    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    t0 = 0.0
    if crop_around_cpa and cpa_s is not None:
        duration = len(y) / float(SR)
        start_s = max(0.0, float(cpa_s) - CROP_HALF_S)
        end_s = min(duration, float(cpa_s) + CROP_HALF_S)
        i0 = int(round(start_s * SR))
        i1 = int(round(end_s * SR))
        i1 = max(i0 + SR // 4, i1)
        y = y[i0:i1]
        t0 = start_s
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    db = np.asarray(librosa.power_to_db(mel, ref=np.max), dtype=np.float32)
    return db, frame_times(db.shape[1], t0)
