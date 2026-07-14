"""Kinematic physics features for IDMT direction (L2R vs R2L)."""

from __future__ import annotations

import numpy as np

from idmt_experiments.config import PhysicsConfig
from idmt_experiments.physics.spectrogram import (
    clip_center_time_s,
    compute_stft_bundle,
    envelope_peak_time_s,
    load_mono_for_physics,
)
from idmt_experiments.src.preprocess import ClipRecord

KINEMATIC_V2_FEATURES: tuple[str, ...] = (
    # Antisymmetric under y[::-1] with clip_center CPA (no symmetric width).
    "env_energy_half_asymmetry",
    "env_3db_log_asymmetry",
    "env_3db_rise_minus_fall_s",
    "env_10db_log_asymmetry",
    "env_10db_rise_minus_fall_s",
    "stft_doppler_slope_diff_hz_s",
    "stft_doppler_log_signed_slope_ratio",
    "centroid_time_asymmetry_s",
)

# v3: strictly antisymmetric under y[::-1] (each feature negates). Drops the two
# non-flipping Doppler terms from v2 (slope_diff is *invariant*, log_signed_ratio is
# noisy) and adds slope_sum, which negates correctly. Paired with a flip-enforcing
# classifier (with_mean=False + fit_intercept=False), this guarantees the model's
# direction call reverses when the waveform is reversed.
KINEMATIC_V3_FEATURES: tuple[str, ...] = (
    "env_energy_half_asymmetry",
    "env_3db_log_asymmetry",
    "env_3db_rise_minus_fall_s",
    "env_10db_log_asymmetry",
    "env_10db_rise_minus_fall_s",
    "stft_doppler_slope_sum_hz_s",
    "centroid_time_asymmetry_s",
)

# kinematic_full: accuracy-first union of *every* kinematic scalar the pipeline
# already computes (raw v1 envelope/Doppler/centroid + the antisym-derived terms).
# Nothing is dropped for time-reverse symmetry — this set is meant to be paired with a
# nonlinear head (gbt/mlp) to maximise direction accuracy. Speed-scaled ``_x_speed_m``
# terms are excluded because IDMT speed is frequently UNK (would inject 0-fill noise).
KINEMATIC_FULL_FEATURES: tuple[str, ...] = (
    # Envelope shape (clip-center anchored)
    "env_peak_offset_s",
    "env_3db_width_s",
    "env_3db_rise_s",
    "env_3db_fall_s",
    "env_3db_asymmetry",
    "env_10db_width_s",
    "env_10db_rise_s",
    "env_10db_fall_s",
    "env_10db_asymmetry",
    # Spectral centroid trajectory
    "centroid_delta_t_s",
    "centroid_mean_hz",
    "centroid_std_hz",
    "centroid_skew",
    "centroid_span_hz",
    # Doppler ridge transition
    "stft_doppler_transition_width_s",
    "stft_doppler_pre_slope_hz_s",
    "stft_doppler_post_slope_hz_s",
    "stft_doppler_slope_ratio",
    # Antisymmetric-derived (kept for their accuracy contribution too)
    "env_energy_half_asymmetry",
    "env_3db_log_asymmetry",
    "env_3db_rise_minus_fall_s",
    "env_10db_log_asymmetry",
    "env_10db_rise_minus_fall_s",
    "stft_doppler_slope_diff_hz_s",
    "stft_doppler_slope_sum_hz_s",
    "stft_doppler_log_signed_slope_ratio",
    "centroid_time_asymmetry_s",
)


def _env_energy_half_asymmetry(y: np.ndarray, sr: int) -> float:
    """(E_first_half - E_second_half) / E_total — negates under time-reverse."""
    env, _ = _rms_envelope(y, sr)
    mid = max(1, len(env) // 2)
    e1 = float(env[:mid].sum())
    e2 = float(env[mid:].sum())
    return float((e1 - e2) / (e1 + e2 + 1e-12))


def _centroid_time_asymmetry(power: np.ndarray, times_rel: np.ndarray) -> float:
    """Weighted mean STFT time (relative to CPA origin) — negates under time-reverse when origin is clip center."""
    weights = power.sum(axis=0)
    if weights.sum() < 1e-12:
        return 0.0
    return float(np.average(times_rel, weights=weights))


def _resolve_cpa_time_s(y: np.ndarray, sr: int, cpa_mode: str) -> float:
    if cpa_mode == "clip_center":
        return clip_center_time_s(y, sr)
    if cpa_mode == "envelope_peak":
        return envelope_peak_time_s(y, sr)
    raise ValueError(f"Unknown cpa_mode: {cpa_mode!r} (expected clip_center|envelope_peak)")


def _effective_cpa_mode(cfg: PhysicsConfig) -> str:
    if cfg.feature_set in ("kinematic_v2", "kinematic_v3", "kinematic_full"):
        return "clip_center"
    return cfg.cpa_mode


def _log_asymmetry(asymmetry: float) -> float:
    return float(np.log(max(asymmetry, 1e-6)))


def _signed_log_slope_ratio(pre_slope: float, post_slope: float) -> float:
    if abs(pre_slope) < 1e-6:
        return 0.0
    return float(np.sign(pre_slope) * np.log(max(abs(post_slope / pre_slope), 1e-6)))


def _derive_kinematic_antisym(
    raw: dict[str, float],
    *,
    y: np.ndarray,
    sr: int,
    bundle_times: np.ndarray,
    power: np.ndarray,
) -> dict[str, float]:
    """Superset of antisymmetric-oriented derived features (v2 + v3 select from this).

    ``slope_diff`` is *invariant* under y[::-1] (pre/post swap and both negate); keep it
    only for the v2 accuracy comparison. ``slope_sum`` negates correctly and is the v3
    Doppler term.
    """
    pre = raw.get("stft_doppler_pre_slope_hz_s", 0.0)
    post = raw.get("stft_doppler_post_slope_hz_s", 0.0)
    return {
        "env_energy_half_asymmetry": _env_energy_half_asymmetry(y, sr),
        "env_3db_log_asymmetry": _log_asymmetry(raw.get("env_3db_asymmetry", 1.0)),
        "env_3db_rise_minus_fall_s": float(
            raw.get("env_3db_rise_s", 0.0) - raw.get("env_3db_fall_s", 0.0)
        ),
        "env_10db_log_asymmetry": _log_asymmetry(raw.get("env_10db_asymmetry", 1.0)),
        "env_10db_rise_minus_fall_s": float(
            raw.get("env_10db_rise_s", 0.0) - raw.get("env_10db_fall_s", 0.0)
        ),
        "stft_doppler_slope_diff_hz_s": float(pre - post),
        "stft_doppler_slope_sum_hz_s": float(pre + post),
        "stft_doppler_log_signed_slope_ratio": _signed_log_slope_ratio(pre, post),
        "centroid_time_asymmetry_s": _centroid_time_asymmetry(power, bundle_times),
    }


KINEMATIC_V1_FEATURES: tuple[str, ...] = (
    "env_3db_asymmetry",
    "env_3db_rise_s",
    "env_3db_fall_s",
    "env_3db_width_s",
    "env_10db_asymmetry",
    "env_peak_offset_s",
    "stft_doppler_pre_slope_hz_s",
    "stft_doppler_post_slope_hz_s",
    "stft_doppler_slope_ratio",
    "stft_doppler_transition_width_s",
    "centroid_delta_t_s",
    "centroid_skew",
    "centroid_span_hz",
    "centroid_mean_hz",
)


def parse_speed_mps(speed_kmh: str) -> float | None:
    raw = str(speed_kmh).strip().upper()
    if not raw or raw == "UNK":
        return None
    try:
        return float(raw) / 3.6
    except ValueError:
        return None


def _rms_envelope(y: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    frame = hop
    n_frames = 1 + max(0, (len(y) - frame) // hop)
    if n_frames <= 0:
        return np.array([np.sqrt(np.mean(y**2) + 1e-12)]), np.array([0.0])
    env = np.array(
        [np.sqrt(np.mean(y[i * hop : i * hop + frame] ** 2) + 1e-12) for i in range(n_frames)]
    )
    times = np.arange(n_frames) * hop / sr
    return env, times


def _envelope_features(
    y: np.ndarray,
    sr: int,
    t_ref_s: float,
    speed_mps: float | None,
    db_thresholds: tuple[float, ...] = (-3.0, -10.0),
) -> dict[str, float]:
    """Envelope duration cues relative to ``t_ref_s`` (clip center or envelope peak).

    -3 dB contour is still anchored to the detected envelope peak level; only rise/fall
  durations are measured vs ``t_ref_s`` so clip-center mode flips under time-reverse.
    """
    env, env_t = _rms_envelope(y, sr)
    env_db = 20.0 * np.log10(env + 1e-12)
    peak_idx = int(np.argmax(env_db))
    peak_t = env_t[min(peak_idx, len(env_t) - 1)]
    peak_db = env_db[peak_idx]
    out: dict[str, float] = {"env_peak_offset_s": float(peak_t - t_ref_s)}

    for thr in db_thresholds:
        label = f"env_{int(abs(thr))}db"
        above = env_db >= (peak_db + thr)
        if not np.any(above):
            out[f"{label}_width_s"] = 0.0
            out[f"{label}_rise_s"] = 0.0
            out[f"{label}_fall_s"] = 0.0
            out[f"{label}_asymmetry"] = 1.0
            continue
        idx = np.where(above)[0]
        t_rise = env_t[idx[0]]
        t_fall = env_t[idx[-1]]
        rise = t_ref_s - t_rise
        fall = t_fall - t_ref_s
        out[f"{label}_width_s"] = float(t_fall - t_rise)
        out[f"{label}_rise_s"] = float(rise)
        out[f"{label}_fall_s"] = float(fall)
        out[f"{label}_asymmetry"] = float(fall / rise) if rise > 1e-6 else 1.0
        if speed_mps is not None:
            out[f"{label}_width_x_speed_m"] = float(out[f"{label}_width_s"] * speed_mps)
    return out


def _skew(x: np.ndarray) -> float:
    x = x - x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean((x / s) ** 3))


def _spectral_centroid_features(
    power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    speed_mps: float | None,
) -> dict[str, float]:
    denom = power.sum(axis=0) + 1e-12
    centroid = (freqs[:, None] * power).sum(axis=0) / denom
    pre_mask = times <= 0
    post_mask = times >= 0
    t_pre = times[pre_mask][int(np.argmax(centroid[pre_mask]))] if np.any(pre_mask) else 0.0
    t_post = times[post_mask][int(np.argmin(centroid[post_mask]))] if np.any(post_mask) else 0.0
    delta_t = t_post - t_pre
    out = {
        "centroid_delta_t_s": float(delta_t),
        "centroid_mean_hz": float(np.mean(centroid)),
        "centroid_std_hz": float(np.std(centroid)),
        "centroid_skew": float(_skew(centroid)),
        "centroid_span_hz": float(np.max(centroid) - np.min(centroid)),
    }
    if speed_mps is not None:
        out["centroid_delta_t_x_speed_m"] = float(abs(delta_t) * speed_mps)
    return out


def _track_dominant_ridge(power: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peak_bins = np.argmax(power, axis=0)
    return times, freqs[peak_bins]


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _doppler_transition_features(
    ridge_t: np.ndarray,
    ridge_f: np.ndarray,
    speed_mps: float | None,
) -> dict[str, float]:
    if len(ridge_t) < 5:
        return {
            "stft_doppler_transition_width_s": 0.0,
            "stft_doppler_pre_slope_hz_s": 0.0,
            "stft_doppler_post_slope_hz_s": 0.0,
            "stft_doppler_slope_ratio": 1.0,
        }
    order = np.argsort(ridge_t)
    t = ridge_t[order]
    f = ridge_f[order]
    cpa_idx = int(np.argmin(np.abs(t)))
    f_cpa = f[cpa_idx]
    f_range = np.percentile(f, 95) - np.percentile(f, 5)
    if f_range < 1.0:
        f_range = max(float(np.max(f) - np.min(f)), 1.0)
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
    out = {
        "stft_doppler_transition_width_s": width,
        "stft_doppler_pre_slope_hz_s": float(pre_slope),
        "stft_doppler_post_slope_hz_s": float(post_slope),
        "stft_doppler_slope_ratio": float(ratio),
    }
    if speed_mps is not None:
        out["stft_doppler_transition_width_x_speed_m"] = width * speed_mps
    return out


def feature_names(cfg: PhysicsConfig) -> tuple[str, ...]:
    if cfg.feature_set == "kinematic_v1":
        names = list(KINEMATIC_V1_FEATURES)
        if not cfg.use_speed_scaled_features:
            names = [n for n in names if not n.endswith("_x_speed_m")]
        return tuple(names)
    if cfg.feature_set == "kinematic_v2":
        return KINEMATIC_V2_FEATURES
    if cfg.feature_set == "kinematic_v3":
        return KINEMATIC_V3_FEATURES
    if cfg.feature_set == "kinematic_full":
        return KINEMATIC_FULL_FEATURES
    raise ValueError(f"Unknown feature_set: {cfg.feature_set!r}")


def _extract_raw_kinematic(
    y: np.ndarray,
    sr: int,
    speed_mps: float | None,
    *,
    cpa_mode: str,
) -> dict[str, float]:
    t_origin = _resolve_cpa_time_s(y, sr, cpa_mode)
    bundle = compute_stft_bundle(y, sr, t_peak_s=t_origin)
    feats: dict[str, float] = {}
    feats.update(_envelope_features(y, sr, t_origin, speed_mps))
    feats.update(_spectral_centroid_features(bundle.power, bundle.freqs, bundle.times, speed_mps))
    rt, rf = _track_dominant_ridge(bundle.power, bundle.freqs, bundle.times)
    feats.update(_doppler_transition_features(rt, rf, speed_mps))
    feats["_power"] = bundle.power
    feats["_bundle_times"] = bundle.times
    feats["_cpa_mode"] = cpa_mode
    return feats


def extract_physics_features(
    record: ClipRecord,
    cfg: PhysicsConfig,
    *,
    mono_source: str | None = None,
    time_reverse: bool = False,
) -> dict[str, float]:
    if record.is_background and not cfg.include_no_vehicle:
        raise ValueError("Background clip excluded when include_no_vehicle=False")

    mono = mono_source or cfg.mono_source
    y, sr = load_mono_for_physics(record.wav_path, mono)
    speed_mps = parse_speed_mps(record.speed_kmh) if cfg.use_speed_scaled_features else None
    if time_reverse:
        y = y[::-1].copy()

    cpa_mode = _effective_cpa_mode(cfg)
    raw = _extract_raw_kinematic(y, sr, speed_mps, cpa_mode=cpa_mode)
    if cfg.feature_set in ("kinematic_v2", "kinematic_v3"):
        power = raw.pop("_power")
        bundle_times = raw.pop("_bundle_times")
        raw.pop("_cpa_mode", None)
        raw = _derive_kinematic_antisym(raw, y=y, sr=sr, bundle_times=bundle_times, power=power)
    elif cfg.feature_set == "kinematic_full":
        # Union: keep the raw v1 envelope/Doppler/centroid features AND add the
        # antisym-derived terms on top (accuracy-first, nothing dropped).
        power = raw.pop("_power")
        bundle_times = raw.pop("_bundle_times")
        raw.pop("_cpa_mode", None)
        raw.update(
            _derive_kinematic_antisym(raw, y=y, sr=sr, bundle_times=bundle_times, power=power)
        )
    else:
        raw.pop("_power", None)
        raw.pop("_bundle_times", None)
        raw.pop("_cpa_mode", None)

    names = feature_names(cfg)
    return {k: float(raw.get(k, 0.0)) for k in names}


def physics_label_index(record: ClipRecord) -> int:
    if record.travel_direction == "L2R":
        return 0
    if record.travel_direction == "R2L":
        return 1
    raise ValueError(f"No direction label for clip {record.clip_id}")
