"""Tiered simulation effects for Phase 4 validation (simulated only).

Each tier wraps the Tier-1 free-field tone path with one realism factor so
failures can be attributed. Headline metric elsewhere remains orbit RMS.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from traj_reconstruction.kinematics import SPEED_OF_SOUND, resample_path_constant_speed
from traj_reconstruction.synthesize import synthesize_tone_on_path


def synthesize_tier1(
    xy: np.ndarray,
    *,
    speed_mps: float,
    f0_hz: float = 500.0,
    **kwargs: Any,
) -> dict[str, Any]:
    out = synthesize_tone_on_path(xy, speed_mps=speed_mps, f0_hz=f0_hz, **kwargs)
    out["tier"] = "tier1"
    return out


def synthesize_tier2_harmonics_rpm(
    xy: np.ndarray,
    *,
    speed_mps: float,
    f0_hz: float = 120.0,
    n_harmonics: int = 4,
    gear_shift: bool = True,
    sr: int = 22050,
    pad_s: float = 0.25,
    c: float = SPEED_OF_SOUND,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Harmonic stack + optional RPM step (engine-order confound)."""
    del rng
    base = synthesize_tone_on_path(
        xy, speed_mps=speed_mps, f0_hz=f0_hz, sr=sr, pad_s=pad_s, c=c
    )
    traj = base["trajectory"]
    t = traj["t"]
    t_r = base["t_r"]
    r = base["R_retarded"]
    valid = np.isfinite(t_r) & np.isfinite(r) & (r > 1e-9)

    # RPM profile in emission time: step mid-pass (gear shift).
    f_emit = np.full(len(t), float(f0_hz), dtype=np.float64)
    if gear_shift:
        mid = 0.5 * (float(t[0]) + float(t[-1]))
        f_emit = np.where(t < mid, f0_hz, f0_hz * 1.25)

    audio = np.zeros(len(t), dtype=np.float64)
    # Map emission frequency onto retarded samples.
    f_at_r = np.interp(np.nan_to_num(t_r, nan=0.0), t, f_emit)
    for h in range(1, int(n_harmonics) + 1):
        audio[valid] += (1.0 / h) * np.sin(2.0 * np.pi * h * f_at_r[valid] * t_r[valid]) / r[valid]

    peak = float(np.max(np.abs(audio))) if np.any(valid) else 0.0
    if peak > 1e-12:
        audio *= 0.95 / peak

    out = dict(base)
    out["audio"] = audio
    out["tier"] = "tier2"
    out["f_emit_hz"] = f_emit
    out["gear_shift"] = bool(gear_shift)
    out["n_harmonics"] = int(n_harmonics)
    out["shift_time_sec"] = float(mid) if gear_shift else None
    return out


def synthesize_tier3_noise(
    xy: np.ndarray,
    *,
    speed_mps: float,
    snr_db: float,
    f0_hz: float = 500.0,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Add white noise at a target SNR (signal power / noise power)."""
    rng = rng or np.random.default_rng(0)
    base = synthesize_tone_on_path(xy, speed_mps=speed_mps, f0_hz=f0_hz, rng=rng, **kwargs)
    audio = np.asarray(base["audio"], dtype=np.float64)
    sig_pow = float(np.mean(audio**2)) + 1e-12
    noise_pow = sig_pow / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_pow), size=audio.shape)
    noisy = audio + noise
    peak = float(np.max(np.abs(noisy)))
    if peak > 1e-12:
        noisy *= 0.95 / peak
    out = dict(base)
    out["audio"] = noisy
    out["tier"] = "tier3"
    out["snr_db"] = float(snr_db)
    return out


def synthesize_tier4_multipath(
    xy: np.ndarray,
    *,
    speed_mps: float,
    f0_hz: float = 500.0,
    reflection_delay_s: float = 0.012,
    reflection_gain: float = 0.55,
    sr: int = 22050,
    **kwargs: Any,
) -> dict[str, Any]:
    """Single specular echo (delayed attenuated copy) — ghost ridge stress."""
    base = synthesize_tone_on_path(
        xy, speed_mps=speed_mps, f0_hz=f0_hz, sr=sr, **kwargs
    )
    audio = np.asarray(base["audio"], dtype=np.float64)
    delay = max(int(round(float(reflection_delay_s) * sr)), 1)
    echo = np.zeros_like(audio)
    echo[delay:] = float(reflection_gain) * audio[:-delay]
    mixed = audio + echo
    peak = float(np.max(np.abs(mixed)))
    if peak > 1e-12:
        mixed *= 0.95 / peak
    out = dict(base)
    out["audio"] = mixed
    out["tier"] = "tier4"
    out["reflection_delay_s"] = float(reflection_delay_s)
    out["reflection_gain"] = float(reflection_gain)
    return out


def synthesize_tier5_directivity(
    xy: np.ndarray,
    *,
    speed_mps: float,
    f0_hz: float = 500.0,
    directivity: float = 0.6,
    sr: int = 22050,
    pad_s: float = 0.25,
    c: float = SPEED_OF_SOUND,
    **kwargs: Any,
) -> dict[str, Any]:
    """Front/rear radiation asymmetry modulating amplitude (scale confound)."""
    traj = resample_path_constant_speed(xy, speed_mps=speed_mps, sr=sr, pad_s=pad_s)
    base = synthesize_tone_on_path(
        xy, speed_mps=speed_mps, f0_hz=f0_hz, sr=sr, pad_s=pad_s, c=c, **kwargs
    )
    t = traj["t"]
    t_r = base["t_r"]
    r = base["R_retarded"]
    valid = np.isfinite(t_r) & np.isfinite(r) & (r > 1e-9)
    idx = np.clip(np.searchsorted(t, np.nan_to_num(t_r, nan=0.0), side="left"), 0, len(t) - 1)
    # Heading relative to bearing: cardioid-like pattern.
    heading = np.arctan2(traj["vy"][idx], traj["vx"][idx])
    bearing = np.arctan2(traj["y"][idx], traj["x"][idx])
    aspect = heading - bearing
    gain = np.clip(1.0 + float(directivity) * np.cos(aspect), 0.15, 2.0)

    audio = np.zeros(len(t), dtype=np.float64)
    audio[valid] = (
        gain[valid] * np.sin(2.0 * np.pi * float(f0_hz) * t_r[valid]) / np.maximum(r[valid], 1e-9)
    )
    peak = float(np.max(np.abs(audio))) if np.any(valid) else 0.0
    if peak > 1e-12:
        audio *= 0.95 / peak

    out = dict(base)
    out["audio"] = audio
    out["tier"] = "tier5"
    out["directivity"] = float(directivity)
    out["scale_ambiguous"] = True
    return out
