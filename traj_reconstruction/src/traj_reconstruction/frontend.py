"""Audio-only signal front end: joint harmonic ridge + amplitude envelope.

**No metadata enters this module.** Public APIs accept WAV samples and/or an
STFT magnitude map (dB) only. Ground-truth trajectories are never arguments.

Tier-1 notes
------------
On a pure tone with retarded-time Doppler, ``f_obs(t)`` follows the Doppler
ridge and ``A_env(t)`` tracks geometric spreading (∝ 1/r at emission), up to
peak-normalization of the clip. The amplitude peak sits near the *retarded*
CPA, which can lag the centerline ``argmin r(t)`` by roughly ``h/c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from traj_reconstruction.contract import (
    SPEC_SR_HZ,
    STFT_HOP_LENGTH,
    STFT_N_FFT,
)
from traj_reconstruction.kinematics import compute_stft_db, stft_frame_times, stft_n_frames


@dataclass(frozen=True)
class RidgeFeatures:
    """Time-aligned Doppler observables extracted from audio/STFT only."""

    frame_times: np.ndarray  # (T,)
    f_obs_hz: np.ndarray  # (T,) fundamental track
    harmonics_hz: np.ndarray  # (T, H) including fundamental as column 0
    A_env: np.ndarray  # (T,) smoothed harmonic energy (linear scale)
    quality: np.ndarray  # (T,) in [0, 1]
    quality_mean: float
    n_harmonics: int
    sr: int
    n_fft: int
    hop_length: int
    stft_db: np.ndarray  # (F, T) echo of input/computed STFT (for overlays)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_times": self.frame_times,
            "f_obs_hz": self.f_obs_hz,
            "harmonics_hz": self.harmonics_hz,
            "A_env": self.A_env,
            "quality": self.quality,
            "quality_mean": self.quality_mean,
            "n_harmonics": self.n_harmonics,
            "sr": self.sr,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
        }


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), always_2d=False)
    except Exception:
        import wave

        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            n = int(wf.getnframes())
            raw = wf.readframes(n)
            sw = int(wf.getsampwidth())
            ch = int(wf.getnchannels())
        if sw == 2:
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        else:
            raise ValueError(f"unsupported wav sampwidth={sw}")
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)
        audio = pcm
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio, int(sr)


def _db_to_linear(stft_db: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(stft_db, dtype=np.float64) / 20.0)


def _freq_axis(n_fft: int, sr: int) -> np.ndarray:
    return np.arange(n_fft // 2 + 1, dtype=np.float64) * float(sr) / float(n_fft)


def _smooth(x: np.ndarray, win: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if win <= 1 or x.size == 0:
        return x
    w = int(win) | 1  # odd
    kernel = np.ones(w, dtype=np.float64) / float(w)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def extract_ridges(
    *,
    stft_db: np.ndarray | None = None,
    audio: np.ndarray | None = None,
    wav_path: Path | str | None = None,
    sr: int = SPEC_SR_HZ,
    n_fft: int = STFT_N_FFT,
    hop_length: int = STFT_HOP_LENGTH,
    n_harmonics: int = 3,
    f_min_hz: float = 80.0,
    f_max_hz: float = 4000.0,
    max_df_hz: float = 80.0,
    smooth_env: int = 7,
) -> RidgeFeatures:
    """Extract joint-harmonic Doppler ridge and amplitude envelope.

    Parameters
    ----------
    stft_db, audio, wav_path
        Provide at least one acoustic source. Prefer ``stft_db`` when it already
        matches the Phase 1 grid; otherwise STFT is computed from audio/wav.
    n_harmonics
        Number of harmonics to score jointly (1 = fundamental only).
    max_df_hz
        Max |Δf| between adjacent frames for the Viterbi transition.
    """
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")

    if stft_db is None:
        if audio is None:
            if wav_path is None:
                raise ValueError("provide stft_db, audio, or wav_path")
            audio, sr_file = _load_wav(Path(wav_path))
            sr = int(sr_file)
        audio = np.asarray(audio, dtype=np.float64)
        stft_db = compute_stft_db(audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    else:
        stft_db = np.asarray(stft_db, dtype=np.float32)
        if stft_db.ndim != 2:
            raise ValueError(f"stft_db must be (F, T), got {stft_db.shape}")

    mag = _db_to_linear(stft_db)
    n_freq, n_time = mag.shape
    freqs = _freq_axis(n_fft, sr)
    if freqs.shape[0] != n_freq:
        # Allow slight mismatch by rebuilding axis from rows.
        freqs = np.linspace(0.0, 0.5 * sr, n_freq)

    f_min = float(f_min_hz)
    f_max = float(f_max_hz)
    # Candidates for fundamental: keep headroom for highest harmonic.
    f0_max = min(f_max, f_max / float(n_harmonics))
    cand_mask = (freqs >= f_min) & (freqs <= f0_max)
    cand_idx = np.flatnonzero(cand_mask)
    if cand_idx.size < 3:
        raise ValueError("frequency search band too narrow for tracking")

    # Emission score per frame/candidate: sum harmonic magnitudes.
    scores = np.full((n_time, cand_idx.size), -np.inf, dtype=np.float64)
    for j, k0 in enumerate(cand_idx):
        f0 = float(freqs[k0])
        acc = np.zeros(n_time, dtype=np.float64)
        for h in range(1, n_harmonics + 1):
            fh = h * f0
            k = int(np.argmin(np.abs(freqs - fh)))
            if abs(freqs[k] - fh) > max(2.0 * sr / n_fft, 5.0):
                continue
            # Local 3-bin peak energy
            lo, hi = max(0, k - 1), min(n_freq, k + 2)
            acc += mag[lo:hi, :].max(axis=0)
        scores[:, j] = acc

    # Viterbi with soft continuity on f0.
    df_bins = max(1, int(np.ceil(float(max_df_hz) / max(freqs[1] - freqs[0], 1e-9))))
    neg_inf = -1e30
    dp = np.full_like(scores, neg_inf)
    back = np.full(scores.shape, -1, dtype=np.int32)
    dp[0] = scores[0]
    for t in range(1, n_time):
        for j in range(cand_idx.size):
            lo = max(0, j - df_bins)
            hi = min(cand_idx.size, j + df_bins + 1)
            prev = dp[t - 1, lo:hi]
            k = int(np.argmax(prev))
            dp[t, j] = scores[t, j] + prev[k]
            back[t, j] = lo + k

    path_j = np.empty(n_time, dtype=np.int32)
    path_j[-1] = int(np.argmax(dp[-1]))
    for t in range(n_time - 2, -1, -1):
        path_j[t] = back[t + 1, path_j[t + 1]]

    f0_track = freqs[cand_idx[path_j]]
    harmonics = np.empty((n_time, n_harmonics), dtype=np.float64)
    for h in range(n_harmonics):
        harmonics[:, h] = (h + 1) * f0_track

    # Envelope from summed linear energy at tracked harmonic bins.
    env = np.zeros(n_time, dtype=np.float64)
    peak_score = np.zeros(n_time, dtype=np.float64)
    for t in range(n_time):
        e = 0.0
        for h in range(n_harmonics):
            fh = float(harmonics[t, h])
            k = int(np.argmin(np.abs(freqs - fh)))
            lo, hi = max(0, k - 1), min(n_freq, k + 2)
            local = float(mag[lo:hi, t].max())
            e += local
        env[t] = e
        peak_score[t] = float(scores[t, path_j[t]])

    env_s = _smooth(env, smooth_env)
    # Quality: relative score vs per-frame best candidate + continuity.
    best = scores.max(axis=1)
    best = np.maximum(best, 1e-12)
    q_score = np.clip(peak_score / best, 0.0, 1.0)
    df = np.abs(np.diff(f0_track, prepend=f0_track[0]))
    q_cont = np.clip(1.0 - df / max(float(max_df_hz) * 2.0, 1.0), 0.0, 1.0)
    quality = _smooth(0.7 * q_score + 0.3 * q_cont, 5)
    quality_mean = float(np.mean(quality))

    frame_times = stft_frame_times(n_time, sr=sr, hop_length=hop_length)
    return RidgeFeatures(
        frame_times=frame_times.astype(np.float64),
        f_obs_hz=f0_track.astype(np.float64),
        harmonics_hz=harmonics.astype(np.float64),
        A_env=env_s.astype(np.float64),
        quality=quality.astype(np.float64),
        quality_mean=quality_mean,
        n_harmonics=int(n_harmonics),
        sr=int(sr),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        stft_db=np.asarray(stft_db, dtype=np.float32),
    )


def expected_doppler_freq(
    *,
    t_obs: np.ndarray,
    t_r: np.ndarray,
    v_radial_at_emission: np.ndarray,
    f0_hz: float,
    c: float = 343.0,
) -> np.ndarray:
    """Eval-only: f_obs = f0 * c / (c + dR/dt) for phase ∝ t_r."""
    del t_obs
    alpha = float(c) / (float(c) + np.asarray(v_radial_at_emission, dtype=np.float64))
    f = float(f0_hz) * alpha
    f = np.where(np.isfinite(t_r), f, np.nan)
    return f


def plot_ridge_overlay(
    features: RidgeFeatures,
    out_path: Path | str,
    *,
    title: str = "Ridge track (audio-only)",
    fmax_hz: float | None = None,
) -> Path:
    """Save spectrogram + tracked fundamental (and harmonics) overlay PNG."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stft = features.stft_db
    n_freq, n_time = stft.shape
    freqs = _freq_axis(features.n_fft, features.sr)
    if freqs.shape[0] != n_freq:
        freqs = np.linspace(0.0, 0.5 * features.sr, n_freq)
    fmax = float(fmax_hz) if fmax_hz is not None else min(4000.0, float(freqs[-1]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    t = features.frame_times
    extent = [float(t[0]), float(t[-1]) if len(t) > 1 else 1.0, float(freqs[0]), float(freqs[-1])]
    ax.imshow(stft, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.plot(t, features.f_obs_hz, color="cyan", lw=1.5, label="f0 track")
    for h in range(1, features.n_harmonics):
        ax.plot(t, features.harmonics_hz[:, h], color="lime", lw=0.8, alpha=0.7)
    ax.set_ylim(0.0, fmax)
    ax.set_ylabel("Hz")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    axes[1].plot(t, features.A_env, color="C1", label="A_env")
    axes[1].plot(t, features.quality, color="C2", alpha=0.7, label="quality")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
