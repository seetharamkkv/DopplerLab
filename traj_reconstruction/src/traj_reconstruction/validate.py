"""Phase 4 tiered validation harness (orbit RMS scorecard)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from traj_reconstruction.flexible import fit_flexible_from_audio
from traj_reconstruction.kinematics import (
    canonical_state_frames,
    interpolate_state,
    stft_frame_times,
    stft_n_frames,
)
from traj_reconstruction.orbit import xy_from_state
from traj_reconstruction.parametric import fit_orbit_from_audio, plot_fit_overlay
from traj_reconstruction.path_families import (
    make_arc,
    make_s_curve,
    make_straight,
)
from traj_reconstruction.tiers import (
    synthesize_tier1,
    synthesize_tier2_harmonics_rpm,
    synthesize_tier3_noise,
    synthesize_tier4_multipath,
    synthesize_tier5_directivity,
)


MethodName = Literal["parametric_straight", "flexible"]


@dataclass(frozen=True)
class CaseResult:
    tier: str
    family: str
    method: str
    orbit_rms: float
    speed_rel_err: float | None
    cpa_rel_err: float | None
    residual_rms_f: float
    success: bool
    tag: str
    extra: dict[str, Any]


def _gt_xy(synth: dict[str, Any]) -> np.ndarray:
    traj = synth["trajectory"]
    n = len(synth["audio"])
    times = stft_frame_times(stft_n_frames(n))
    state = interpolate_state(traj["t"], traj["state"], times)
    can, _ = canonical_state_frames(state)
    return xy_from_state(can)


def _speed_from_state(synth: dict[str, Any]) -> float:
    vx = synth["trajectory"]["vx"]
    vy = synth["trajectory"]["vy"]
    spd = np.sqrt(vx * vx + vy * vy)
    moving = spd > 1e-6
    return float(np.mean(spd[moving])) if np.any(moving) else float(np.mean(spd))


def _eval_case(
    synth: dict[str, Any],
    *,
    family: str,
    method: MethodName,
    tier: str,
    tag: str,
) -> CaseResult:
    gt = _gt_xy(synth)
    gt_speed = _speed_from_state(synth)
    gt_cpa = float(synth["cpa_distance_m"])
    if method == "parametric_straight":
        fit = fit_orbit_from_audio(
            audio=synth["audio"],
            sr=synth["sr"],
            family="straight",
            use_amplitude=True,
            gt_xy=gt,
            gt_speed_mps=gt_speed,
            gt_cpa_distance_m=gt_cpa,
        )
    else:
        fit = fit_flexible_from_audio(
            audio=synth["audio"],
            sr=synth["sr"],
            gt_xy=gt,
            gt_speed_mps=gt_speed,
            gt_cpa_distance_m=gt_cpa,
            n_modes=3,
        )
    metrics = fit.metrics or {}
    return CaseResult(
        tier=tier,
        family=family,
        method=method,
        orbit_rms=float(fit.orbit.rms if fit.orbit is not None else np.nan),
        speed_rel_err=metrics.get("speed_rel_err"),
        cpa_rel_err=metrics.get("cpa_rel_err"),
        residual_rms_f=float(fit.residual_rms_f),
        success=bool(fit.success),
        tag=tag,
        extra={
            "pred_h": fit.params.get("h"),
            "pred_v": fit.params.get("v"),
            "gt_h": gt_cpa,
            "gt_v": gt_speed,
            "scale_ambiguous": bool(synth.get("scale_ambiguous", False)),
            "fit": fit,
            "gt_xy": gt,
        },
    )


def _paths() -> dict[str, np.ndarray]:
    return {
        "straight": make_straight(cpa_distance_m=15.0, half_length_m=65.0, heading_rad=0.25),
        "arc": make_arc(
            cpa_distance_m=16.0, radius_m=50.0, sweep_rad=np.deg2rad(65.0), heading_rad=0.1
        ),
        "s_curve": make_s_curve(
            cpa_distance_m=15.0, half_length_m=70.0, amplitude_m=9.0, heading_rad=0.2
        ),
    }


def run_tier1(speed_mps: float = 20.0) -> list[CaseResult]:
    rows: list[CaseResult] = []
    for fam, xy in _paths().items():
        synth = synthesize_tier1(xy, speed_mps=speed_mps, f0_hz=500.0)
        for method in ("parametric_straight", "flexible"):
            if method == "parametric_straight" and fam not in ("straight", "arc"):
                # Still evaluate straight model as baseline on freehand.
                pass
            rows.append(
                _eval_case(
                    synth,
                    family=fam,
                    method=method,  # type: ignore[arg-type]
                    tier="tier1",
                    tag=f"tier1_{fam}_{method}",
                )
            )
    return rows


def run_tier2(speed_mps: float = 20.0) -> list[CaseResult]:
    xy = _paths()["straight"]
    synth = synthesize_tier2_harmonics_rpm(xy, speed_mps=speed_mps, gear_shift=True)
    rows = []
    for method in ("flexible", "parametric_straight"):
        case = _eval_case(
            synth,
            family="straight",
            method=method,  # type: ignore[arg-type]
            tier="tier2",
            tag=f"tier2_harmonics_rpm_{method}",
        )
        # Rebuild with extra tags (frozen dataclass).
        rows.append(
            CaseResult(
                tier=case.tier,
                family=case.family,
                method=case.method,
                orbit_rms=case.orbit_rms,
                speed_rel_err=case.speed_rel_err,
                cpa_rel_err=case.cpa_rel_err,
                residual_rms_f=case.residual_rms_f,
                success=case.success,
                tag=case.tag,
                extra={
                    **{k: v for k, v in case.extra.items()},
                    "gear_shift": True,
                    "shift_time_sec": synth.get("shift_time_sec"),
                },
            )
        )
    return rows


def run_tier3(
    snr_grid_db: list[float] | None = None,
    speed_mps: float = 20.0,
    rng_seed: int = 0,
) -> list[CaseResult]:
    snr_grid_db = snr_grid_db or [30.0, 20.0, 10.0, 5.0, 0.0]
    xy = _paths()["straight"]
    rng = np.random.default_rng(rng_seed)
    rows: list[CaseResult] = []
    for snr in snr_grid_db:
        synth = synthesize_tier3_noise(
            xy, speed_mps=speed_mps, snr_db=snr, rng=rng
        )
        for method in ("flexible", "parametric_straight"):
            case = _eval_case(
                synth,
                family="straight",
                method=method,  # type: ignore[arg-type]
                tier="tier3",
                tag=f"tier3_snr{snr:g}_{method}",
            )
            rows.append(
                CaseResult(
                    tier=case.tier,
                    family=case.family,
                    method=case.method,
                    orbit_rms=case.orbit_rms,
                    speed_rel_err=case.speed_rel_err,
                    cpa_rel_err=case.cpa_rel_err,
                    residual_rms_f=case.residual_rms_f,
                    success=case.success,
                    tag=case.tag,
                    extra={**case.extra, "snr_db": float(snr)},
                )
            )
    return rows


def run_tier4(speed_mps: float = 20.0) -> list[CaseResult]:
    xy = _paths()["straight"]
    synth = synthesize_tier4_multipath(xy, speed_mps=speed_mps)
    rows = []
    for method in ("flexible", "parametric_straight"):
        case = _eval_case(
            synth,
            family="straight",
            method=method,  # type: ignore[arg-type]
            tier="tier4",
            tag=f"tier4_multipath_{method}",
        )
        rows.append(
            CaseResult(
                tier=case.tier,
                family=case.family,
                method=case.method,
                orbit_rms=case.orbit_rms,
                speed_rel_err=case.speed_rel_err,
                cpa_rel_err=case.cpa_rel_err,
                residual_rms_f=case.residual_rms_f,
                success=case.success,
                tag=case.tag,
                extra={
                    **case.extra,
                    "wrong_trace_proxy": case.residual_rms_f > 40.0,
                },
            )
        )
    return rows


def run_tier5(speed_mps: float = 20.0) -> list[CaseResult]:
    xy = _paths()["straight"]
    synth = synthesize_tier5_directivity(xy, speed_mps=speed_mps, directivity=0.7)
    rows = []
    for method in ("flexible", "parametric_straight"):
        case = _eval_case(
            synth,
            family="straight",
            method=method,  # type: ignore[arg-type]
            tier="tier5",
            tag=f"tier5_directivity_{method}",
        )
        rows.append(
            CaseResult(
                tier=case.tier,
                family=case.family,
                method=case.method,
                orbit_rms=case.orbit_rms,
                speed_rel_err=case.speed_rel_err,
                cpa_rel_err=case.cpa_rel_err,
                residual_rms_f=case.residual_rms_f,
                success=case.success,
                tag=case.tag,
                extra={**case.extra, "scale_ambiguous": True},
            )
        )
    return rows


def _summarize(rows: list[CaseResult]) -> dict[str, Any]:
    by_tier: dict[str, list[CaseResult]] = {}
    for r in rows:
        by_tier.setdefault(r.tier, []).append(r)

    summary: dict[str, Any] = {}
    for tier, cases in by_tier.items():
        rms = [c.orbit_rms for c in cases if np.isfinite(c.orbit_rms)]
        summary[tier] = {
            "n": len(cases),
            "orbit_rms_mean": float(np.mean(rms)) if rms else None,
            "orbit_rms_median": float(np.median(rms)) if rms else None,
            "orbit_rms_max": float(np.max(rms)) if rms else None,
            "methods": sorted({c.method for c in cases}),
            "families": sorted({c.family for c in cases}),
        }
    return summary


def _gates(rows: list[CaseResult]) -> dict[str, dict[str, Any]]:
    """Go/no-go notes — honest about what is / isn't solid."""
    def tier_rows(t: str) -> list[CaseResult]:
        return [r for r in rows if r.tier == t]

    t1 = tier_rows("tier1")
    t1_flex_s = [r for r in t1 if r.family == "straight" and r.method == "flexible"]
    t1_para_s = [r for r in t1 if r.family == "straight" and r.method == "parametric_straight"]
    t3 = tier_rows("tier3")
    t4 = tier_rows("tier4")
    t5 = tier_rows("tier5")

    def mean_rms(cs: list[CaseResult]) -> float | None:
        vals = [c.orbit_rms for c in cs if np.isfinite(c.orbit_rms)]
        return float(np.mean(vals)) if vals else None

    gates = {
        "tier1": {
            "go": bool(t1_flex_s and mean_rms(t1_flex_s) is not None and mean_rms(t1_flex_s) < 8.0),
            "note": "Clean tone free-field; straight/flexible should be strong.",
            "orbit_rms_flexible_straight": mean_rms(t1_flex_s),
            "orbit_rms_parametric_straight": mean_rms(t1_para_s),
        },
        "tier2": {
            "go": True,
            "note": "Harmonics+RPM: expect higher f residual around gear shift; treat as confound stress test.",
            "orbit_rms_mean": mean_rms(tier_rows("tier2")),
        },
        "tier3": {
            "go": bool(t3) and mean_rms([r for r in t3 if r.extra.get("snr_db") == 20.0]) is not None,
            "note": "SNR sweep published; do not claim low-SNR deployment without quoting the curve.",
            "snr_curve": sorted(
                [
                    {
                        "snr_db": r.extra.get("snr_db"),
                        "method": r.method,
                        "orbit_rms": r.orbit_rms,
                    }
                    for r in t3
                ],
                key=lambda d: (d["method"], d["snr_db"] or 0),
            ),
        },
        "tier4": {
            "go": bool(t4) and mean_rms(t4) is not None and mean_rms(t4) < 40.0,
            "waiver": mean_rms(t4) is not None and mean_rms(t4) >= 40.0,
            "note": "Multipath ghosts can inflate residuals; wrong_trace_proxy flags hard failures.",
            "orbit_rms_mean": mean_rms(t4),
            "wrong_trace_proxies": [r.extra.get("wrong_trace_proxy") for r in t4],
        },
        "tier5": {
            "go": True,
            "note": "Directivity biases amplitude→distance; always set scale_ambiguous=true for this tier.",
            "orbit_rms_mean": mean_rms(t5),
            "cpa_rel_errs": [r.cpa_rel_err for r in t5],
            "scale_ambiguous": True,
        },
    }
    return gates


def run_all_tiers(
    *,
    out_dir: Path | str,
    snr_grid_db: list[float] | None = None,
    save_worst: int = 3,
) -> dict[str, Any]:
    """Run Tier 1–5, write JSON + Markdown report + SNR plot + worst overlays."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worst_dir = out_dir / "worst"
    worst_dir.mkdir(exist_ok=True)

    rows = []
    rows.extend(run_tier1())
    rows.extend(run_tier2())
    rows.extend(run_tier3(snr_grid_db=snr_grid_db))
    rows.extend(run_tier4())
    rows.extend(run_tier5())

    # Drop non-serializable fit objects for JSON.
    serializable = []
    for r in rows:
        d = {
            "tier": r.tier,
            "family": r.family,
            "method": r.method,
            "orbit_rms": r.orbit_rms,
            "speed_rel_err": r.speed_rel_err,
            "cpa_rel_err": r.cpa_rel_err,
            "residual_rms_f": r.residual_rms_f,
            "success": r.success,
            "tag": r.tag,
            "extra": {
                k: v
                for k, v in r.extra.items()
                if k not in ("fit", "gt_xy") and _jsonable(v)
            },
        }
        serializable.append(d)

    summary = _summarize(rows)
    gates = _gates(rows)

    # Worst-N by orbit RMS
    ranked = sorted(rows, key=lambda r: r.orbit_rms if np.isfinite(r.orbit_rms) else 1e9, reverse=True)
    for i, r in enumerate(ranked[: int(save_worst)]):
        fit = r.extra.get("fit")
        gt = r.extra.get("gt_xy")
        if fit is not None:
            plot_fit_overlay(
                fit,
                worst_dir / f"worst_{i+1}_{r.tag}.png",
                gt_xy=gt,
                title=f"WORST {i+1}: {r.tag} orbit_rms={r.orbit_rms:.2f}",
            )

    # SNR curve figure
    snr_png = out_dir / "snr_curve.png"
    _plot_snr_curve([r for r in rows if r.tier == "tier3"], snr_png)

    report = {
        "data_scope": "simulated_only",
        "headline_metric": "orbit_rms (Procrustes rotation ± reflection about mic)",
        "summary": summary,
        "gates": gates,
        "cases": serializable,
        "artifacts": {
            "snr_curve": str(snr_png),
            "worst_dir": str(worst_dir),
        },
    }
    (out_dir / "tiered_validation.json").write_text(json.dumps(report, indent=2))
    (out_dir / "tiered_validation.md").write_text(_render_markdown(report))
    return report


def _jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), list, dict))


def _plot_snr_curve(rows: list[CaseResult], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in sorted({r.method for r in rows}):
        pts = sorted(
            [r for r in rows if r.method == method],
            key=lambda r: float(r.extra.get("snr_db", 0.0)),
        )
        if not pts:
            continue
        ax.plot(
            [r.extra.get("snr_db") for r in pts],
            [r.orbit_rms for r in pts],
            marker="o",
            label=method,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Orbit RMS (m)")
    ax.set_title("Tier 3 — orbit error vs SNR (simulated)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Tiered validation report (simulated)",
        "",
        f"**Headline metric:** `{report['headline_metric']}`",
        "",
        "Absolute world heading is not scored. Paths are compared after orbit alignment.",
        "",
        "## Summary by tier",
        "",
        "| Tier | N | mean orbit RMS | median | max |",
        "|------|---|----------------|--------|-----|",
    ]
    for tier, s in report["summary"].items():
        lines.append(
            f"| {tier} | {s['n']} | {_fmt(s['orbit_rms_mean'])} | "
            f"{_fmt(s['orbit_rms_median'])} | {_fmt(s['orbit_rms_max'])} |"
        )
    lines += ["", "## Gates / honesty notes", ""]
    for tier, g in report["gates"].items():
        status = "GO" if g.get("go") else ("WAIVER" if g.get("waiver") else "NO-GO")
        lines.append(f"### {tier} — **{status}**")
        lines.append(f"- {g.get('note', '')}")
        if tier == "tier3":
            lines.append("- See `snr_curve.png` for the accuracy-vs-SNR curve.")
        if tier == "tier5":
            lines.append("- `scale_ambiguous=true` required when quoting CPA distance.")
        lines.append("")
    lines += [
        "## Artifacts",
        "",
        f"- JSON: `tiered_validation.json`",
        f"- SNR curve: `{report['artifacts']['snr_curve']}`",
        f"- Worst overlays: `{report['artifacts']['worst_dir']}`",
        "",
    ]
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)
