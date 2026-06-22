"""Hand-crafted physics features for vehicle length estimation (Phase A)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from length_estimation.config import SUBBAND_EDGES
from length_estimation.src.preprocess import ClipRecord, align_and_crop, load_audio
from length_estimation.src.spectrograms import (
    ReassignedAtoms,
    SpectrogramBundle,
    build_spectrogram_bundle,
)


def _rms_envelope(y: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    frame = hop
    n_frames = 1 + max(0, (len(y) - frame) // hop)
    if n_frames <= 0:
        t = np.array([0.0])
        return np.array([np.sqrt(np.mean(y**2) + 1e-12)]), t
    env = np.array(
        [np.sqrt(np.mean(y[i * hop : i * hop + frame] ** 2) + 1e-12) for i in range(n_frames)]
    )
    times = np.arange(n_frames) * hop / sr
    return env, times


def envelope_features(
    y: np.ndarray,
    t_rel: np.ndarray,
    sr: int,
    speed_mps: float,
    db_thresholds: tuple[float, ...] = (-3.0, -10.0),
) -> dict[str, float]:
    env, env_t = _rms_envelope(y, sr)
    env_db = 20.0 * np.log10(env + 1e-12)
    peak_idx = int(np.argmax(env_db))
    peak_t = env_t[peak_idx]
    peak_db = env_db[peak_idx]

    out: dict[str, float] = {"env_peak_time_s": float(peak_t)}

    for thr in db_thresholds:
        label = f"env_{int(abs(thr))}db"
        above = env_db >= (peak_db + thr)
        if not np.any(above):
            out[f"{label}_width_s"] = 0.0
            out[f"{label}_width_x_speed_m"] = 0.0
            out[f"{label}_rise_s"] = 0.0
            out[f"{label}_fall_s"] = 0.0
            out[f"{label}_asymmetry"] = 1.0
            continue

        idx = np.where(above)[0]
        t_rise = env_t[idx[0]]
        t_fall = env_t[idx[-1]]
        width = t_fall - t_rise
        rise = peak_t - t_rise
        fall = t_fall - peak_t
        asym = fall / rise if rise > 1e-6 else 1.0

        out[f"{label}_width_s"] = float(width)
        out[f"{label}_width_x_speed_m"] = float(width * speed_mps)
        out[f"{label}_rise_s"] = float(rise)
        out[f"{label}_fall_s"] = float(fall)
        out[f"{label}_asymmetry"] = float(asym)

    return out


def _band_energy_envelope(
    power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    f_lo: float,
    f_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(mask):
        return np.zeros(len(times)), times
    band_power = power[mask].sum(axis=0)
    return band_power, times


def subband_crosscorr_features(
    power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    speed_mps: float,
    band_edges: tuple[float, ...] = SUBBAND_EDGES,
) -> dict[str, float]:
    bands: list[tuple[str, np.ndarray]] = []
    for i in range(len(band_edges) - 1):
        f_lo, f_hi = band_edges[i], band_edges[i + 1]
        env, _ = _band_energy_envelope(power, freqs, times, f_lo, f_hi)
        bands.append((f"{int(f_lo)}_{int(f_hi)}", env))

    out: dict[str, float] = {}
    for i, (name_a, env_a) in enumerate(bands):
        for j, (name_b, env_b) in enumerate(bands):
            if j <= i:
                continue
            env_a = env_a - env_a.mean()
            env_b = env_b - env_b.mean()
            if np.std(env_a) < 1e-12 or np.std(env_b) < 1e-12:
                lag_s = 0.0
                peak_corr = 0.0
            else:
                corr = signal.correlate(env_a, env_b, mode="full")
                lags = signal.correlation_lags(len(env_a), len(env_b), mode="full")
                peak_idx = int(np.argmax(corr))
                lag_frames = lags[peak_idx]
                dt = times[1] - times[0] if len(times) > 1 else 0.0
                lag_s = float(lag_frames * dt)
                peak_corr = float(corr[peak_idx] / (np.linalg.norm(env_a) * np.linalg.norm(env_b) + 1e-12))

            key = f"xcorr_lag_{name_a}_{name_b}_s"
            out[key] = lag_s
            out[f"{key}_x_speed_m"] = abs(lag_s) * speed_mps
            out[f"xcorr_peak_{name_a}_{name_b}"] = peak_corr

    return out


def spectral_centroid_features(
    power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    speed_mps: float,
) -> dict[str, float]:
    denom = power.sum(axis=0) + 1e-12
    centroid = (freqs[:, None] * power).sum(axis=0) / denom

    pre_mask = times <= 0
    post_mask = times >= 0

    def _extremum(mask: np.ndarray, fn: Any) -> float:
        if not np.any(mask):
            return 0.0
        return float(fn(centroid[mask], initial=centroid[mask][0]))

    t_pre = times[pre_mask][int(np.argmax(centroid[pre_mask]))] if np.any(pre_mask) else 0.0
    t_post = times[post_mask][int(np.argmin(centroid[post_mask]))] if np.any(post_mask) else 0.0
    delta_t = t_post - t_pre

    return {
        "centroid_delta_t_s": float(delta_t),
        "centroid_delta_t_x_speed_m": float(abs(delta_t) * speed_mps),
        "centroid_mean_hz": float(np.mean(centroid)),
        "centroid_std_hz": float(np.std(centroid)),
        "centroid_skew": float(_skew(centroid)),
        "centroid_span_hz": float(np.max(centroid) - np.min(centroid)),
    }


def _skew(x: np.ndarray) -> float:
    x = x - x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean((x / s) ** 3))


def _track_dominant_ridge(
    power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    peak_bins = np.argmax(power, axis=0)
    ridge_f = freqs[peak_bins]
    return times, ridge_f


def _track_reassigned_ridge(
    atoms: ReassignedAtoms,
    t_window: tuple[float, float] = (-2.0, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    t_flat = atoms.times.ravel()
    f_flat = atoms.freqs.ravel()
    m_flat = atoms.mags.ravel()
    valid = np.isfinite(t_flat) & np.isfinite(f_flat) & (m_flat > 0)
    t_flat = t_flat[valid]
    f_flat = f_flat[valid]
    m_flat = m_flat[valid]

    if len(t_flat) == 0:
        return np.array([0.0]), np.array([0.0])

    mask = (t_flat >= t_window[0]) & (t_flat <= t_window[1])
    t_flat = t_flat[mask]
    f_flat = f_flat[mask]
    m_flat = m_flat[mask]
    if len(t_flat) == 0:
        return np.array([0.0]), np.array([0.0])

    # Per-frame: strongest reassigned atom near each STFT hop centre
    hop_t = np.arange(int(t_window[0] * atoms.sr), int(t_window[1] * atoms.sr), atoms.hop_length) / atoms.sr
    ridge_t: list[float] = []
    ridge_f: list[float] = []
    half_hop = atoms.hop_length / atoms.sr / 2
    for tc in hop_t:
        sel = (t_flat >= tc - half_hop) & (t_flat < tc + half_hop)
        if not np.any(sel):
            continue
        idx = np.argmax(m_flat[sel])
        sel_t = t_flat[sel]
        sel_f = f_flat[sel]
        ridge_t.append(float(sel_t[idx]))
        ridge_f.append(float(sel_f[idx]))

    if not ridge_t:
        return np.array([0.0]), np.array([0.0])
    return np.asarray(ridge_t), np.asarray(ridge_f)


def doppler_transition_features(
    ridge_t: np.ndarray,
    ridge_f: np.ndarray,
    speed_mps: float,
) -> dict[str, float]:
    if len(ridge_t) < 5:
        return {
            "doppler_transition_width_s": 0.0,
            "doppler_transition_width_x_speed_m": 0.0,
            "doppler_pre_slope_hz_s": 0.0,
            "doppler_post_slope_hz_s": 0.0,
            "doppler_slope_ratio": 1.0,
        }

    order = np.argsort(ridge_t)
    t = ridge_t[order]
    f = ridge_f[order]

    cpa_idx = int(np.argmin(np.abs(t)))
    f_cpa = f[cpa_idx]

    # Normalised sigmoid fit proxy: 10–90% frequency transition width around CPA
    f_range = np.percentile(f, 95) - np.percentile(f, 5)
    if f_range < 1.0:
        f_range = max(np.max(f) - np.min(f), 1.0)

    f_lo = f_cpa - 0.4 * f_range
    f_hi = f_cpa + 0.4 * f_range
    cross_lo = t[f <= f_lo]
    cross_hi = t[f >= f_hi]
    width = float(cross_hi[0] - cross_lo[-1]) if len(cross_lo) and len(cross_hi) else 0.0
    width = max(width, 0.0)

    pre = t < 0
    post = t > 0
    pre_slope = _linear_slope(t[pre], f[pre]) if np.sum(pre) >= 2 else 0.0
    post_slope = _linear_slope(t[post], f[post]) if np.sum(post) >= 2 else 0.0
    ratio = abs(post_slope / pre_slope) if abs(pre_slope) > 1e-6 else 1.0

    return {
        "doppler_transition_width_s": width,
        "doppler_transition_width_x_speed_m": width * speed_mps,
        "doppler_pre_slope_hz_s": float(pre_slope),
        "doppler_post_slope_hz_s": float(post_slope),
        "doppler_slope_ratio": float(ratio),
    }


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    coef = np.polyfit(x, y, 1)
    return float(coef[0])


def extract_features_from_bundle(
    bundle: SpectrogramBundle,
    y: np.ndarray,
    t_rel: np.ndarray,
    sr: int,
    speed_mps: float,
    *,
    db_thresholds: tuple[float, ...] = (-3.0, -10.0),
    band_edges: tuple[float, ...] = SUBBAND_EDGES,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(envelope_features(y, t_rel, sr, speed_mps, db_thresholds))

    feats.update(
        subband_crosscorr_features(
            bundle.stft_power, bundle.stft_freqs, bundle.stft_times, speed_mps, band_edges
        )
    )
    feats.update(
        spectral_centroid_features(bundle.stft_power, bundle.stft_freqs, bundle.stft_times, speed_mps)
    )

    if bundle.ssq_power is not None and bundle.ssq_freqs is not None and bundle.ssq_times is not None:
        feats.update(
            {
                f"ssq_{k}": v
                for k, v in subband_crosscorr_features(
                    bundle.ssq_power, bundle.ssq_freqs, bundle.ssq_times, speed_mps, band_edges
                ).items()
            }
        )

    # Doppler transition: reassigned preferred, STFT fallback
    if bundle.reassigned is not None:
        rt, rf = _track_reassigned_ridge(bundle.reassigned)
        prefix = "reassigned_"
    else:
        rt, rf = _track_dominant_ridge(bundle.stft_power, bundle.stft_freqs, bundle.stft_times)
        prefix = "stft_"

    doppler = doppler_transition_features(rt, rf, speed_mps)
    feats.update({f"{prefix}{k}": v for k, v in doppler.items()})

    return feats


def extract_clip_features(
    record: ClipRecord,
    *,
    include_reassigned: bool = True,
    include_ssq: bool = True,
) -> dict[str, float | str]:
    y, sr = load_audio(record.wav_path)
    y_crop, t_rel = align_and_crop(y, sr, record.cpa_time_s)
    speed_mps = record.speed_kmh / 3.6

    bundle = build_spectrogram_bundle(
        y_crop,
        t_rel,
        sr,
        include_reassigned=include_reassigned,
        include_ssq=include_ssq,
        include_mel=False,
    )
    feats = extract_features_from_bundle(bundle, y_crop, t_rel, sr, speed_mps)
    feats["clip_id"] = record.clip_id
    feats["vehicle"] = record.vehicle
    feats["speed_kmh"] = record.speed_kmh
    feats["speed_mps"] = speed_mps
    feats["length_m"] = record.length_m
    feats["wheelbase_m"] = record.wheelbase_m
    feats["power_kw"] = record.power_kw
    return feats
