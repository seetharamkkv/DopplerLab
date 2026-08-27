"""Constant-speed path timing, STFT grid, and canonical state (sim Phase 1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from traj_reconstruction.contract import SPEC_SR_HZ, STFT_HOP_LENGTH, STFT_N_FFT
from traj_reconstruction.orbit import canonical_xy


SPEED_OF_SOUND = 343.0


def polyline_arclength(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2:
        raise ValueError("path must be (N, 2) with N >= 2")
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_path_constant_speed(
    xy: np.ndarray,
    *,
    speed_mps: float,
    sr: int = SPEC_SR_HZ,
    pad_s: float = 0.25,
) -> dict[str, np.ndarray]:
    """Parameterize a polyline at constant along-track speed (DopplerSim-compatible)."""
    xy = np.asarray(xy, dtype=np.float64)
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(xy, axis=0)) > 1e-9, axis=1)
    xy = xy[keep]
    if len(xy) < 2:
        raise ValueError("path needs at least two distinct points")

    s = polyline_arclength(xy)
    length = float(s[-1])
    if length < 1e-3:
        raise ValueError("path length is too short")

    v = float(speed_mps)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("speed_mps must be positive")

    travel_s = length / v
    duration_s = travel_s + 2.0 * float(pad_s)
    n = max(int(np.ceil(duration_s * sr)), 1)
    t = np.arange(n, dtype=np.float64) / float(sr)

    t_motion = np.clip(t - float(pad_s), 0.0, travel_s)
    s_query = v * t_motion

    x = np.interp(s_query, s, xy[:, 0])
    y = np.interp(s_query, s, xy[:, 1])

    ds = np.diff(s)
    dxy = np.diff(xy, axis=0)
    ds_safe = np.maximum(ds, 1e-12)
    tx = dxy[:, 0] / ds_safe
    ty = dxy[:, 1] / ds_safe
    seg = np.searchsorted(s, s_query, side="right") - 1
    seg = np.clip(seg, 0, len(tx) - 1)
    moving = (t_motion > 0.0) & (t_motion < travel_s)
    vx = np.where(moving, v * tx[seg], 0.0)
    vy = np.where(moving, v * ty[seg], 0.0)

    state = np.column_stack([x, vx, y, vy]).astype(np.float64)
    return {
        "t": t,
        "x": x.astype(np.float64),
        "y": y.astype(np.float64),
        "vx": vx.astype(np.float64),
        "vy": vy.astype(np.float64),
        "state": state,
        "s": s_query.astype(np.float64),
        "length_m": np.array([length], dtype=np.float64),
        "duration_s": np.array([duration_s], dtype=np.float64),
        "travel_s": np.array([travel_s], dtype=np.float64),
        "pad_s": np.array([float(pad_s)], dtype=np.float64),
    }


def stft_n_frames(n_samples: int, hop_length: int = STFT_HOP_LENGTH) -> int:
    return 1 + int(n_samples) // int(hop_length)


def stft_frame_times(
    n_frames: int,
    *,
    sr: int = SPEC_SR_HZ,
    hop_length: int = STFT_HOP_LENGTH,
) -> np.ndarray:
    frames = np.arange(int(n_frames), dtype=np.float64)
    return frames * float(hop_length) / float(sr)


def interpolate_state(t_src: np.ndarray, state: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    t_src = np.asarray(t_src, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)
    out = np.empty((t_dst.shape[0], state.shape[1]), dtype=np.float64)
    for j in range(state.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, state[:, j])
    return out


def canonical_state_frames(state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate/reflect full state so CPA is on +y and vx_CPA ≥ 0."""
    state = np.asarray(state, dtype=np.float64)
    xy = np.column_stack([state[:, 0], state[:, 2]])
    _, meta = canonical_xy(xy)
    rot = float(meta["rotation_rad"])
    c, s = np.cos(rot), np.sin(rot)
    x2 = c * state[:, 0] - s * state[:, 2]
    y2 = s * state[:, 0] + c * state[:, 2]
    vx2 = c * state[:, 1] - s * state[:, 3]
    vy2 = s * state[:, 1] + c * state[:, 3]
    reflected = False
    cpa = int(meta["cpa_index"])
    if vx2[cpa] < 0.0:
        x2 = -x2
        vx2 = -vx2
        reflected = True
    out = state.copy()
    out[:, 0] = x2
    out[:, 1] = vx2
    out[:, 2] = y2
    out[:, 3] = vy2
    meta = {**meta, "reflected_x": reflected}
    return out, meta


def derived_from_state(state: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    state = np.asarray(state, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    x, vx, y, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    speed = np.sqrt(vx * vx + vy * vy)
    range_m = np.sqrt(x * x + y * y)
    denom = np.maximum(range_m, 1e-12)
    v_radial = (x * vx + y * vy) / denom
    bearing = np.arctan2(y, x)
    cpa_idx = int(np.argmin(range_m))
    return {
        "speed_mps": speed.astype(np.float32),
        "range_m": range_m.astype(np.float32),
        "radial_velocity_mps": v_radial.astype(np.float32),
        "bearing_rad": bearing.astype(np.float32),
        "cpa_time_sec": float(times[cpa_idx]),
        "cpa_distance_m": float(range_m[cpa_idx]),
        "cpa_index": cpa_idx,
    }


def solve_retarded_time_path(
    t_obs: np.ndarray,
    t_src: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
    *,
    c: float = SPEED_OF_SOUND,
) -> tuple[np.ndarray, np.ndarray]:
    t_obs = np.asarray(t_obs, dtype=np.float64)
    t_src = np.asarray(t_src, dtype=np.float64)
    x_src = np.asarray(x_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)

    r_src = np.sqrt(x_src * x_src + y_src * y_src)
    f_src = t_src + r_src / float(c)

    t_r = np.full(len(t_obs), np.nan, dtype=np.float64)
    r_at = np.full(len(t_obs), np.nan, dtype=np.float64)

    inside = (t_obs >= float(f_src[0])) & (t_obs <= float(f_src[-1]))
    if not np.any(inside):
        return t_r, r_at

    to = t_obs[inside]
    j = np.searchsorted(f_src, to, side="right") - 1
    j = np.clip(j, 0, len(f_src) - 2)
    f0, f1 = f_src[j], f_src[j + 1]
    denom = f1 - f0
    alpha = np.where(np.abs(denom) < 1e-15, 0.0, (to - f0) / denom)
    alpha = np.clip(alpha, 0.0, 1.0)

    tr = t_src[j] + alpha * (t_src[j + 1] - t_src[j])
    xr = x_src[j] + alpha * (x_src[j + 1] - x_src[j])
    yr = y_src[j] + alpha * (y_src[j + 1] - y_src[j])
    rr = np.sqrt(xr * xr + yr * yr)

    t_r[inside] = tr
    r_at[inside] = rr
    return t_r, r_at


def compute_stft_db(
    audio: np.ndarray,
    *,
    sr: int = SPEC_SR_HZ,
    n_fft: int = STFT_N_FFT,
    hop_length: int = STFT_HOP_LENGTH,
) -> np.ndarray:
    """Magnitude STFT in dB, shape (freq, time), librosa-like centered frames.

    Frame count is ``1 + n_samples // hop_length`` so it matches
    ``stft_frame_times`` / Phase 1 state grids.
    """
    del sr  # times come from hop/sr elsewhere; STFT itself is sample-based
    audio = np.asarray(audio, dtype=np.float64)
    n = int(audio.shape[0])
    n_frames = stft_n_frames(n, hop_length)
    pad = int(n_fft // 2)
    y = np.pad(audio, (pad, pad), mode="constant")
    window = np.hanning(n_fft).astype(np.float64)
    n_freqs = n_fft // 2 + 1
    out = np.empty((n_freqs, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * int(hop_length)
        frame = y[start : start + n_fft]
        if frame.shape[0] < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.shape[0]))
        spec = np.fft.rfft(frame * window, n=n_fft)
        out[:, i] = np.abs(spec)
    return (20.0 * np.log10(np.maximum(out, 1e-10))).astype(np.float32)
