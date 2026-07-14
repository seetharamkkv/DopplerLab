"""Univariate feature separability audit (Phase 2.5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from idmt_experiments.config import PHYSICS_DIRECTION_LABELS, PhysicsConfig
from idmt_experiments.physics.dataset import build_feature_batch
from idmt_experiments.src.preprocess import ClipRecord


def univariate_separability(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    labels: tuple[str, ...] = PHYSICS_DIRECTION_LABELS,
) -> dict:
    """Per-feature Mann-Whitney U + Cohen's d between L2R (0) and R2L (1)."""
    out: dict = {"features": {}, "n_samples": int(len(y)), "labels": list(labels)}
    mask0 = y == 0
    mask1 = y == 1
    for j, name in enumerate(feature_names):
        a = X[mask0, j]
        b = X[mask1, j]
        if len(a) < 2 or len(b) < 2:
            out["features"][name] = {"u_stat": None, "p_value": None, "cohens_d": None}
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = _cohens_d(a, b)
        out["features"][name] = {
            "u_stat": float(u),
            "p_value": float(p),
            "cohens_d": d,
            "mean_L2R": float(a.mean()),
            "mean_R2L": float(b.mean()),
        }
    ranked = sorted(
        out["features"].items(),
        key=lambda kv: abs(kv[1].get("cohens_d") or 0.0),
        reverse=True,
    )
    out["ranked_by_effect_size"] = [k for k, _ in ranked]
    return out


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled_std)


def time_reverse_feature_audit(
    X_fwd: np.ndarray,
    X_rev: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict:
    """Per-feature behavior under time-reverse (mono channel only).

    Kinematic direction cues should anticorrelate or flip sign vs forward clip.
    """
    if len(X_fwd) != len(X_rev):
        raise ValueError(f"Forward/reverse row mismatch: {len(X_fwd)} vs {len(X_rev)}")
    mask0 = y == 0
    mask1 = y == 1
    features: dict[str, dict] = {}
    for j, name in enumerate(feature_names):
        f = X_fwd[:, j]
        r = X_rev[:, j]
        nonzero = (np.abs(f) > 1e-9) & (np.abs(r) > 1e-9)
        sign_flip_rate = float(np.mean(np.sign(f[nonzero]) != np.sign(r[nonzero]))) if nonzero.any() else None
        corr = float(np.corrcoef(f, r)[0, 1]) if len(f) > 1 and f.std() > 1e-12 and r.std() > 1e-12 else None
        d_fwd = _cohens_d(f[mask0], f[mask1])
        d_rev = _cohens_d(r[mask0], r[mask1])
        separation_sign_flips = None
        if d_fwd is not None and d_rev is not None and abs(d_fwd) > 1e-6 and abs(d_rev) > 1e-6:
            separation_sign_flips = bool(np.sign(d_fwd) != np.sign(d_rev))
        features[name] = {
            "corr_fwd_rev": corr,
            "sign_flip_rate": sign_flip_rate,
            "cohens_d_fwd": d_fwd,
            "cohens_d_rev": d_rev,
            "separation_sign_flips": separation_sign_flips,
            "abs_cohens_d_fwd": abs(d_fwd) if d_fwd is not None else None,
            "abs_cohens_d_rev": abs(d_rev) if d_rev is not None else None,
        }
    ranked_anticorr = sorted(
        features.items(),
        key=lambda kv: kv[1].get("corr_fwd_rev") if kv[1].get("corr_fwd_rev") is not None else 1.0,
    )
    ranked_sep_flip = [
        k
        for k, v in features.items()
        if v.get("separation_sign_flips") and (v.get("abs_cohens_d_fwd") or 0) > 0.1
    ]
    return {
        "n_samples": int(len(y)),
        "mono_only": True,
        "test": "time_reverse_same_channel",
        "features": features,
        "ranked_by_anticorrelation": [k for k, _ in ranked_anticorr],
        "separation_flips_with_effect": ranked_sep_flip,
    }


def run_feature_audit(
    records: list[ClipRecord],
    cfg: PhysicsConfig,
    out_path: Path | None = None,
    *,
    flip_out_path: Path | None = None,
) -> dict:
    batch = build_feature_batch(records, cfg, show_progress=True)
    report = univariate_separability(batch.X, batch.y, batch.feature_names)
    report["mono_source"] = cfg.mono_source
    report["mono_only"] = True
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    flip_report = None
    if flip_out_path is not None:
        print("  Extracting time-reversed features (same mono channel)...", flush=True)
        batch_rev = build_feature_batch(records, cfg, time_reverse=True, show_progress=True)
        flip_report = time_reverse_feature_audit(
            batch.X, batch_rev.X, batch.y, batch.feature_names
        )
        flip_report["mono_source"] = cfg.mono_source
        flip_out_path = Path(flip_out_path)
        flip_out_path.parent.mkdir(parents=True, exist_ok=True)
        flip_out_path.write_text(json.dumps(flip_report, indent=2), encoding="utf-8")

    if flip_report is not None:
        report["time_reverse_audit_path"] = str(flip_out_path)
    return report
