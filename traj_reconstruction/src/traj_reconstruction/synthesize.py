"""Tier-1 free-field tone synthesis on a freehand path (simulated only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from traj_reconstruction.contract import SPEC_SR_HZ
from traj_reconstruction.kinematics import (
    SPEED_OF_SOUND,
    resample_path_constant_speed,
    solve_retarded_time_path,
)


def synthesize_tone_on_path(
    xy: np.ndarray,
    *,
    speed_mps: float,
    f0_hz: float = 500.0,
    sr: int = SPEC_SR_HZ,
    pad_s: float = 0.25,
    c: float = SPEED_OF_SOUND,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Pure tone + 1/r spreading via retarded time (Tier-1 sanity acoustics)."""
    del rng  # deterministic tone; kept for API symmetry with future noise tiers
    traj = resample_path_constant_speed(xy, speed_mps=speed_mps, sr=sr, pad_s=pad_s)
    t = traj["t"]
    t_r, r = solve_retarded_time_path(t, t, traj["x"], traj["y"], c=c)
    valid = np.isfinite(t_r) & np.isfinite(r) & (r > 1e-9)

    audio = np.zeros(len(t), dtype=np.float64)
    # Emission phase continuous in emission time.
    audio[valid] = np.sin(2.0 * np.pi * float(f0_hz) * t_r[valid]) / r[valid]

    peak = float(np.max(np.abs(audio))) if np.any(valid) else 0.0
    if peak > 1e-12:
        audio *= 0.95 / peak

    range_inst = np.sqrt(traj["x"] ** 2 + traj["y"] ** 2)
    cpa_idx = int(np.argmin(range_inst))

    # Eval-only Doppler factor at emission (not used by the audio-only frontend).
    idx = np.clip(np.searchsorted(t, t_r, side="left"), 0, len(t) - 1)
    x_e, y_e = traj["x"][idx], traj["y"][idx]
    vx_e, vy_e = traj["vx"][idx], traj["vy"][idx]
    r_e = np.maximum(np.sqrt(x_e * x_e + y_e * y_e), 1e-9)
    v_r = (x_e * vx_e + y_e * vy_e) / r_e
    v_r = np.where(valid, v_r, np.nan)
    # Instantaneous freq of sin(2π f0 t_r): f_obs = f0 * dt_r/dt_obs = f0 * c/(c + dR/dt).
    alpha = float(c) / (float(c) + v_r)
    f_obs_true = float(f0_hz) * alpha

    return {
        "audio": audio,
        "sr": int(sr),
        "trajectory": traj,
        "t_r": t_r,
        "R_retarded": r,
        "range_inst": range_inst,
        "cpa_time_sec": float(t[cpa_idx]),
        "cpa_distance_m": float(range_inst[cpa_idx]),
        "f0_hz": float(f0_hz),
        "path_xy": np.asarray(xy, dtype=np.float64),
        "v_radial_emission": v_r,
        "f_obs_hz_true": f_obs_true,
    }
