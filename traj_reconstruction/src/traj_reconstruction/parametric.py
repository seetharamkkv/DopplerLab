"""Parametric physics fit for straight / arc orbits (Phase 3a baseline).

Fits observer-centered trajectory parameters against audio-only ridge
features (``f_obs``, optional ``A_env``). Absolute heading is a gauge
freedom — parameters are optimized in the canonical plane (CPA on +y,
travel +x); score with ``orbit_align`` against GT.

Inference uses WAV/STFT → ``extract_ridges`` only (no metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

from traj_reconstruction.frontend import RidgeFeatures, extract_ridges
from traj_reconstruction.kinematics import (
    SPEED_OF_SOUND,
    resample_path_constant_speed,
    solve_retarded_time_path,
)
from traj_reconstruction.orbit import OrbitAlignResult, orbit_align
from traj_reconstruction.path_families import make_arc, make_straight


FamilyName = Literal["straight", "arc"]


@dataclass(frozen=True)
class FitResult:
    family: str
    params: dict[str, float]
    xy_pred: np.ndarray
    frame_times: np.ndarray
    f_hat_hz: np.ndarray
    A_hat: np.ndarray
    residual_rms_f: float
    residual_rms_A: float
    success: bool
    message: str
    orbit: OrbitAlignResult | None = None
    metrics: dict[str, float] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "family": self.family,
            "params": self.params,
            "residual_rms_f": self.residual_rms_f,
            "residual_rms_A": self.residual_rms_A,
            "success": self.success,
            "message": self.message,
            "n_frames": int(self.frame_times.shape[0]),
        }
        if self.orbit is not None:
            out["orbit_rms"] = self.orbit.rms
            out["orbit_length_normalized_rms"] = self.orbit.length_normalized_rms
            out["orbit_reflected"] = self.orbit.reflected
            out["orbit_rotation_rad"] = self.orbit.rotation_rad
        if self.metrics is not None:
            out["metrics"] = self.metrics
        return out


def _polyline_from_params(family: str, params: dict[str, float]) -> np.ndarray:
    if family == "straight":
        return make_straight(
            cpa_distance_m=params["h"],
            half_length_m=params["half_length"],
            heading_rad=0.0,
            n_pts=96,
        )
    if family == "arc":
        return make_arc(
            cpa_distance_m=params["h"],
            radius_m=params["radius"],
            sweep_rad=params["sweep"],
            heading_rad=0.0,
            n_pts=96,
        )
    raise ValueError(f"unsupported family {family!r}")


def _with_derived_length(
    family: str,
    params: dict[str, float],
    frame_times: np.ndarray,
    *,
    pad_s: float,
) -> dict[str, float]:
    """Set half_length so the timed path covers the observation window."""
    out = dict(params)
    duration = float(frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 4.0
    v = max(float(out["v"]), 1.0)
    # Cover full clip with margin beyond pads.
    out["half_length"] = float(np.clip(0.5 * v * (duration + 2.0 * pad_s), 15.0, 400.0))
    if family == "arc" and "radius" in out:
        out["radius"] = float(max(out["radius"], out["h"] + 5.0))
    return out


def _forward_observables(
    family: str,
    params: dict[str, float],
    frame_times: np.ndarray,
    *,
    sr: int = 22050,
    pad_s: float = 0.25,
    c: float = SPEED_OF_SOUND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f_hat, A_hat, xy_on_frames) on ``frame_times``."""
    params = _with_derived_length(family, params, frame_times, pad_s=pad_s)
    poly = _polyline_from_params(family, params)
    traj = resample_path_constant_speed(
        poly, speed_mps=params["v"], sr=sr, pad_s=pad_s
    )
    t_path = traj["t"]
    range_inst = np.sqrt(traj["x"] ** 2 + traj["y"] ** 2)
    t_cpa_geom = float(t_path[int(np.argmin(range_inst))])
    t_src = t_path - t_cpa_geom + float(params["t_cpa"])

    t_obs = np.linspace(float(min(t_src[0], frame_times[0])), float(max(t_src[-1], frame_times[-1])), max(len(t_src), 512))
    # Extend source by endpoint hold if needed for retarded-time coverage.
    t_r, r = solve_retarded_time_path(t_obs, t_src, traj["x"], traj["y"], c=c)
    valid = np.isfinite(t_r) & np.isfinite(r) & (r > 1e-9)

    idx = np.clip(np.searchsorted(t_src, np.nan_to_num(t_r, nan=t_src[0]), side="left"), 0, len(t_src) - 1)
    r_xy = np.maximum(np.sqrt(traj["x"][idx] ** 2 + traj["y"][idx] ** 2), 1e-9)
    v_r = (traj["x"][idx] * traj["vx"][idx] + traj["y"][idx] * traj["vy"][idx]) / r_xy
    f_obs = params["f0"] * float(c) / (float(c) + v_r)
    f_obs = np.where(valid, f_obs, np.nan)
    A = np.where(valid, 1.0 / np.maximum(r, 1e-9), 0.0)

    f_fill = float(params["f0"])
    f_dense = np.where(np.isfinite(f_obs), f_obs, f_fill)
    f_hat = np.interp(frame_times, t_obs, f_dense)
    A_raw = np.interp(frame_times, t_obs, A)
    peak = float(np.max(A_raw)) if np.any(A_raw > 0) else 1.0
    A_hat = A_raw / max(peak, 1e-12)

    x_i = np.interp(frame_times, t_src, traj["x"], left=traj["x"][0], right=traj["x"][-1])
    y_i = np.interp(frame_times, t_src, traj["y"], left=traj["y"][0], right=traj["y"][-1])
    xy = np.column_stack([x_i, y_i])
    return f_hat, A_hat, xy


def init_params_from_ridges(
    feats: RidgeFeatures,
    *,
    family: FamilyName = "straight",
    c: float = SPEED_OF_SOUND,
) -> dict[str, float]:
    """Coarse method-of-moments init (audio-only)."""
    t = feats.frame_times
    f = feats.f_obs_hz
    A = feats.A_env
    q = feats.quality

    a_rel = A / max(float(np.max(A)), 1e-12)
    # Reject pad / dropout locks (e.g. tracker falling to ~80 Hz).
    f_med = float(np.median(f[q > 0.5])) if np.any(q > 0.5) else float(np.median(f))
    mask = (q > 0.65) & (a_rel > 0.12) & (np.abs(f - f_med) < 0.35 * max(f_med, 1.0))
    if mask.sum() < 8:
        mask = (q > 0.5) & (a_rel > 0.05)
    if mask.sum() < 5:
        mask = np.ones_like(q, dtype=bool)

    t_cpa = float(t[int(np.argmax(A))])
    near = mask & (np.abs(t - t_cpa) < 0.5)
    if near.sum() < 3:
        near = mask
    f0 = float(np.median(f[near]))

    f_hi = float(np.percentile(f[mask], 92))
    f_lo = float(np.percentile(f[mask], 8))
    # Symmetric Doppler: (f_hi - f_lo) / (f_hi + f_lo) ≈ v/c for small v/c.
    v = float(np.clip(((f_hi - f_lo) / max(f_hi + f_lo, 1e-6)) * c, 5.0, 50.0))

    # Crossing time of half-swing ≈ 2 h / v  →  h ≈ 0.5 v τ_half
    half = 0.25 * (f_hi - f_lo)
    active = mask & (np.abs(f - f0) > max(half, 3.0))
    if active.sum() >= 2:
        tau = float(t[active].max() - t[active].min())
    else:
        tau = float(t[mask].max() - t[mask].min())
    h = float(np.clip(0.35 * v * max(tau, 0.4), 3.0, 120.0))

    duration = float(t[-1] - t[0]) if len(t) > 1 else 4.0
    half_length = float(np.clip(0.5 * v * max(duration - 0.5, 1.0), 20.0, 200.0))

    params: dict[str, float] = {
        "v": v,
        "h": h,
        "t_cpa": t_cpa,
        "f0": f0,
        "half_length": half_length,
    }
    if family == "arc":
        params["radius"] = float(np.clip(h + 40.0, 25.0, 200.0))
        params["sweep"] = float(np.deg2rad(60.0))
    return params


def _pack(family: str, params: dict[str, float]) -> np.ndarray:
    if family == "straight":
        keys = ("v", "h", "t_cpa", "f0")
    else:
        keys = ("v", "h", "t_cpa", "f0", "radius", "sweep")
    return np.array([params[k] for k in keys], dtype=np.float64)


def _unpack(family: str, x: np.ndarray, base: dict[str, float]) -> dict[str, float]:
    out = dict(base)
    if family == "straight":
        keys = ("v", "h", "t_cpa", "f0")
    else:
        keys = ("v", "h", "t_cpa", "f0", "radius", "sweep")
    for k, val in zip(keys, x):
        out[k] = float(val)
    return out


def _bounds(family: str, t_lo: float, t_hi: float) -> tuple[np.ndarray, np.ndarray]:
    lo = [3.0, 2.0, t_lo, 50.0]
    hi = [60.0, 200.0, t_hi, 4000.0]
    if family == "arc":
        lo += [15.0, np.deg2rad(20.0)]
        hi += [300.0, np.deg2rad(150.0)]
    return np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)


def fit_parametric_orbit(
    feats: RidgeFeatures,
    *,
    family: FamilyName = "straight",
    use_amplitude: bool = True,
    w_f: float = 1.0,
    w_A: float = 0.5,
    pad_s: float = 0.25,
    c: float = SPEED_OF_SOUND,
    gt_xy: np.ndarray | None = None,
    gt_speed_mps: float | None = None,
    gt_cpa_distance_m: float | None = None,
) -> FitResult:
    """Nonlinear least squares fit of straight/arc params to ridge features."""
    init0 = init_params_from_ridges(feats, family=family, c=c)
    t = feats.frame_times
    f_obs = feats.f_obs_hz
    A_obs = feats.A_env / max(float(np.max(feats.A_env)), 1e-12)
    f_med = float(np.median(f_obs[feats.quality > 0.5])) if np.any(feats.quality > 0.5) else float(
        np.median(f_obs)
    )
    w = (
        np.clip(feats.quality, 0.0, 1.0)
        * np.clip(A_obs, 0.0, 1.0)
        * (np.abs(f_obs - f_med) < 0.35 * max(f_med, 1.0)).astype(np.float64)
    )
    w = np.maximum(w, 1e-3)
    f_scale = max(float(np.std(f_obs[w > 0.05])), 10.0) if np.any(w > 0.05) else 10.0
    lo, hi = _bounds(family, float(t[0]), float(t[-1]))

    # Multi-start over CPA distance (frequency alone has flat directions).
    h_grid = np.unique(
        np.clip(
            np.array(
                [init0["h"] * s for s in (0.4, 0.7, 1.0, 1.5)]
                + [8.0, 15.0, 25.0, 40.0],
                dtype=np.float64,
            ),
            lo[1] + 1e-3,
            hi[1] - 1e-3,
        )
    )

    def residual(x: np.ndarray) -> np.ndarray:
        params = _with_derived_length(family, _unpack(family, x, init0), t, pad_s=pad_s)
        f_hat, A_hat, _ = _forward_observables(
            family, params, t, sr=feats.sr, pad_s=pad_s, c=c
        )
        rf = w * (f_hat - f_obs) / f_scale
        if use_amplitude:
            ra = w * (A_hat - A_obs)
            return np.concatenate([np.sqrt(w_f) * rf, np.sqrt(w_A) * ra])
        return np.sqrt(w_f) * rf

    best = None
    for h0 in h_grid:
        init = dict(init0)
        init["h"] = float(h0)
        if family == "arc":
            init["radius"] = float(max(init.get("radius", h0 + 40.0), h0 + 5.0))
        x0 = _pack(family, init)
        x0 = np.minimum(np.maximum(x0, lo + 1e-6), hi - 1e-6)
        result = least_squares(
            residual,
            x0,
            bounds=(lo, hi),
            method="trf",
            max_nfev=120,
            ftol=1e-9,
            xtol=1e-9,
        )
        cost = float(result.cost)
        if best is None or cost < best[0]:
            best = (cost, result, init)

    assert best is not None
    _, result, init = best
    params = _with_derived_length(family, _unpack(family, result.x, init), t, pad_s=pad_s)
    f_hat, A_hat, xy = _forward_observables(
        family, params, t, sr=feats.sr, pad_s=pad_s, c=c
    )
    rf = float(np.sqrt(np.mean((w * (f_hat - f_obs)) ** 2)))
    ra = float(np.sqrt(np.mean((w * (A_hat - A_obs)) ** 2)))

    orbit = None
    metrics: dict[str, float] | None = None
    if gt_xy is not None:
        n = min(len(xy), len(gt_xy))
        orbit = orbit_align(xy[:n], gt_xy[:n])
        metrics = {
            "orbit_rms_m": orbit.rms,
            "pred_speed_mps": params["v"],
            "pred_cpa_distance_m": params["h"],
        }
        if gt_speed_mps is not None:
            metrics["speed_abs_err_mps"] = abs(params["v"] - float(gt_speed_mps))
            metrics["speed_rel_err"] = abs(params["v"] - float(gt_speed_mps)) / max(
                float(gt_speed_mps), 1e-6
            )
        if gt_cpa_distance_m is not None:
            metrics["cpa_abs_err_m"] = abs(params["h"] - float(gt_cpa_distance_m))
            metrics["cpa_rel_err"] = abs(params["h"] - float(gt_cpa_distance_m)) / max(
                float(gt_cpa_distance_m), 1e-6
            )

    return FitResult(
        family=family,
        params=params,
        xy_pred=xy,
        frame_times=t.copy(),
        f_hat_hz=f_hat,
        A_hat=A_hat,
        residual_rms_f=rf,
        residual_rms_A=ra,
        success=bool(result.success) or best[0] < 1e6,
        message=str(result.message),
        orbit=orbit,
        metrics=metrics,
    )


def fit_orbit_from_audio(
    *,
    wav_path: Path | str | None = None,
    stft_db: np.ndarray | None = None,
    audio: np.ndarray | None = None,
    sr: int | None = None,
    family: FamilyName = "straight",
    use_amplitude: bool = True,
    gt_xy: np.ndarray | None = None,
    gt_speed_mps: float | None = None,
    gt_cpa_distance_m: float | None = None,
    n_harmonics: int = 1,
) -> FitResult:
    """End-to-end: audio/STFT → ridges → parametric orbit (no metadata)."""
    kwargs: dict[str, Any] = {"n_harmonics": n_harmonics}
    if stft_db is not None:
        kwargs["stft_db"] = stft_db
        if sr is not None:
            kwargs["sr"] = sr
    elif audio is not None:
        kwargs["audio"] = audio
        if sr is not None:
            kwargs["sr"] = sr
    elif wav_path is not None:
        kwargs["wav_path"] = wav_path
    else:
        raise ValueError("provide wav_path, audio, or stft_db")
    feats = extract_ridges(**kwargs)
    return fit_parametric_orbit(
        feats,
        family=family,
        use_amplitude=use_amplitude,
        gt_xy=gt_xy,
        gt_speed_mps=gt_speed_mps,
        gt_cpa_distance_m=gt_cpa_distance_m,
    )


def plot_fit_overlay(
    fit: FitResult,
    out_path: Path | str,
    *,
    gt_xy: np.ndarray | None = None,
    title: str = "Parametric orbit fit",
) -> Path:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(fit.frame_times, fit.f_hat_hz, label="f_hat", lw=2)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Hz")
    ax.set_title("Frequency fit")
    ax.legend()

    ax = axes[1]
    ax.plot(fit.xy_pred[:, 0], fit.xy_pred[:, 1], label="pred", lw=2)
    if gt_xy is not None:
        n = min(len(fit.xy_pred), len(gt_xy))
        aligned = orbit_align(fit.xy_pred[:n], gt_xy[:n]).aligned_pred
        ax.plot(gt_xy[:n, 0], gt_xy[:n, 1], "--", label="gt", alpha=0.8)
        ax.plot(aligned[:, 0], aligned[:, 1], ":", label="pred aligned", alpha=0.9)
    ax.scatter([0.0], [0.0], c="red", marker="x", label="mic")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
