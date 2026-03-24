# Repurposed from src/features/extraction.py
import os
import numpy as np
import librosa
import scipy.signal

# ── Audio / CQT constants ─────────────────────────────────────────────────────
SR              = 22050
DURATION_S      = 10
N_SAMPLES       = SR * DURATION_S
HOP_LENGTH      = 512
BINS_PER_OCTAVE = 24
N_BINS          = 84
FMIN            = librosa.note_to_hz("C2")
MAX_T           = 432
NYQUIST         = SR / 2.0
MAX_SPEED_MPS   = 50.0

CQT_FREQS = librosa.cqt_frequencies(
    n_bins=N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE
)

def _npy_path(wav_path, name):
    if wav_path is None: return None
    return os.path.join(os.path.dirname(wav_path), name)

def _load_or_none(wav_path, name):
    p = _npy_path(wav_path, name)
    if p and os.path.exists(p):
        return np.load(p)
    return None

def pad_or_trim_time(arr, max_t=MAX_T):
    t = arr.shape[-1]
    if t >= max_t:
        return arr[..., :max_t]
    pad_cfg = [(0, 0)] * (arr.ndim - 1) + [(0, max_t - t)]
    return np.pad(arr, pad_cfg)

def _compute_cqt_log1p(wav):
    C = librosa.cqt(wav, sr=SR, hop_length=HOP_LENGTH,
                    fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    return np.log1p(np.abs(C))

def _compute_dfdt(f_norm):
    dfdt     = np.zeros_like(f_norm)
    dt       = HOP_LENGTH / SR
    dfdt[1:] = (f_norm[1:] - f_norm[:-1]) / dt
    max_abs  = np.max(np.abs(dfdt)) + 1e-8
    return dfdt / max_abs

def _align(arr, target_len):
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)))

def extract_1d_features(wav, wav_path=None, augment=False):
    """7-channel Doppler trajectory → (7, MAX_T)."""
    freq_norm = _load_or_none(wav_path, "frequency.npy")
    if freq_norm is None:
        logC      = _compute_cqt_log1p(wav)
        bins      = np.argmax(logC, axis=0)
        f_hz      = CQT_FREQS[bins]
        f_hz      = scipy.signal.medfilt(np.nan_to_num(f_hz, nan=0.0), kernel_size=5)
        freq_norm = np.clip(f_hz / NYQUIST, 0.0, 1.0)
    else:
        freq_norm = freq_norm.flatten()
        freq_norm = scipy.signal.medfilt(np.nan_to_num(freq_norm, nan=0.0), kernel_size=5)
        freq_norm = np.clip(freq_norm, 0.0, 1.0)

    freq_norm = freq_norm.flatten()
    if augment:
        freq_norm = np.clip(freq_norm + np.random.normal(0, 0.004, size=freq_norm.shape), 0.0, 1.0)

    dfdt = _load_or_none(wav_path, "dfdt.npy")
    if dfdt is None:
        dfdt = _compute_dfdt(freq_norm)
    else:
        dfdt = dfdt.flatten()

    rms = _load_or_none(wav_path, "rms.npy")
    if rms is None:
        rms = librosa.feature.rms(y=wav, frame_length=HOP_LENGTH * 2, hop_length=HOP_LENGTH)[0]
        rms = rms / (np.max(rms) + 1e-8)
    rms = rms.flatten()

    topk = _load_or_none(wav_path, "spec_topk.npy")
    if topk is not None and topk.ndim == 3 and topk.shape[1] >= 1 and topk.shape[2] >= 1:
        topk_freq = topk[:, 0, 0]
    else:
        topk_freq = freq_norm.copy()

    T         = len(dfdt)
    freq_norm = _align(freq_norm, T)
    rms       = _align(rms,       T)
    topk_freq = _align(topk_freq, T)

    dfdt2     = np.gradient(dfdt) / (np.std(np.gradient(dfdt)) + 1e-8)
    sign_dfdt = np.sign(dfdt)
    t_rel     = np.linspace(-1.0, 1.0, T)

    feat = np.stack([dfdt, dfdt2, sign_dfdt, freq_norm, rms, topk_freq, t_rel], axis=0)
    feat = pad_or_trim_time(feat, MAX_T)

    if augment:
        feat = np.roll(feat, np.random.randint(-20, 21), axis=-1)

    return feat.astype(np.float32)

def extract_2d_features(wav, wav_path=None):
    """Log-CQT spectrogram z-scored per bin → (1, 84, MAX_T)."""
    logC = _load_or_none(wav_path, "cqt.npy")
    if logC is None:
        logC = _compute_cqt_log1p(wav)
    else:
        logC = logC.astype(np.float32)
    
    logC = np.nan_to_num(logC)
    mean = logC.mean(axis=1, keepdims=True)
    std  = logC.std(axis=1,  keepdims=True) + 1e-6
    logC = (logC - mean) / std
    logC = pad_or_trim_time(logC, MAX_T)
    return logC[np.newaxis].astype(np.float32)
