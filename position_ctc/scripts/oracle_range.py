"""Exact GT range into a geometric decoder. No audio, no learned weights.

From the DopplerLab root:

    position_ctc/.venv/bin/python position_ctc/scripts/oracle_range.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.data import discover_clips, split_by_vehicle, trim_stationary  # noqa: E402
from position_ctc.features import DATA_HOP, DATA_SR  # noqa: E402
from position_ctc.metrics import aggregate, range_m, score_path  # noqa: E402
from position_ctc.paths import ENV_PATH2D_CLIPS, resolve_audio_clips  # noqa: E402
from position_ctc.oracle_range import (  # noqa: E402
    branch_at_index,
    constant_speed,
    first_heading_cut,
    fit_straight_pass,
    flip_sign_path,
    heading_jump_count,
    recover_with_constant_sign,
    recover_with_true_signs,
    recover_without_speed,
    score_decoder,
)

DT = DATA_HOP / float(DATA_SR)
OUT = ROOT / "position_ctc" / "outputs" / "oracle_range"
DECODERS = (
    "straight_family",
    "range_rdot_speed_true_sign",
    "range_rdot_speed_constant_sign",
    "range_rdot_no_speed",
    "global_sign_flip",
    "corner_branch",
)


def _load_state(clip) -> dict:
    state = np.load(clip.sample_dir / "metadata" / "canonical_state_frames.npy").astype(np.float64)
    dummy = np.zeros((1, state.shape[0]), dtype=np.float32)
    _, state = trim_stationary(dummy, state.astype(np.float32))
    state = np.asarray(state, dtype=np.float64)
    xy = np.stack([state[:, 0], state[:, 2]], axis=1)
    vel = np.stack([state[:, 1], state[:, 3]], axis=1)
    return {
        "clip_id": clip.clip_id,
        "vehicle": clip.vehicle,
        "xy": xy,
        "vel": vel,
        "radius": range_m(xy),
        "speed": constant_speed(vel),
        "corners": heading_jump_count(vel, min_deg=12.0),
    }


def _branch_cut(row: dict) -> int:
    cut = first_heading_cut(row["vel"], min_deg=12.0)
    if cut is not None:
        return cut
    return int(np.argmin(row["radius"]))


def _decode(row: dict) -> dict[str, np.ndarray]:
    xy, vel, radius = row["xy"], row["vel"], row["radius"]
    plus = recover_with_constant_sign(xy, vel, DT, sign=1.0)
    minus = recover_with_constant_sign(xy, vel, DT, sign=-1.0)
    constant = plus if score_path(plus, xy)["orbit_rms_m"] <= score_path(minus, xy)["orbit_rms_m"] else minus
    return {
        "straight_family": fit_straight_pass(radius, DT)["xy"],
        "range_rdot_speed_true_sign": recover_with_true_signs(xy, vel, DT),
        "range_rdot_speed_constant_sign": constant,
        "range_rdot_no_speed": recover_without_speed(xy, vel, DT),
        "global_sign_flip": flip_sign_path(xy, vel, DT),
        "corner_branch": branch_at_index(xy, vel, DT, _branch_cut(row)),
    }


def _plot(row: dict, decoded: dict[str, np.ndarray], path: Path) -> None:
    gt = row["xy"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].plot(gt[:, 0], gt[:, 1], color="#1f4e79", lw=2.0, label="ground truth")
    axes[0].plot(
        decoded["straight_family"][:, 0],
        decoded["straight_family"][:, 1],
        color="#d35400",
        ls="--",
        lw=1.6,
        label="straight family from R",
    )
    axes[0].plot(
        decoded["corner_branch"][:, 0],
        decoded["corner_branch"][:, 1],
        color="#0e7c66",
        ls=":",
        lw=1.8,
        label="same R, sign switch at corner/CPA",
    )
    axes[0].scatter([0.0], [0.0], c="#c0392b", s=36, zorder=5, label="mic")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].legend(loc="best", fontsize=8)
    t = np.arange(len(gt)) * DT
    axes[1].plot(t, row["radius"], color="#1f4e79", lw=2.0, label="R(t)")
    axes[1].plot(t, range_m(decoded["straight_family"]), color="#d35400", ls="--", lw=1.4, label="straight R")
    axes[1].plot(t, range_m(decoded["corner_branch"]), color="#0e7c66", ls=":", lw=1.6, label="branched R")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("range (m)")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{row['clip_id']}  corners={row['corners']}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(value: float) -> str:
    return f"{value:.2f} m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Score only the vehicle-holdout test split. Default is every clip: this decoder does not train.",
    )
    args = parser.parse_args()
    data = resolve_audio_clips(args.data, env=ENV_PATH2D_CLIPS)
    OUT.mkdir(parents=True, exist_ok=True)
    clips = discover_clips(data)
    if args.holdout:
        splits = split_by_vehicle(clips, seed=0)
        chosen = [clips[i] for i in splits["test"]]
    else:
        chosen = clips
    test = [_load_state(clip) for clip in chosen]
    print(f"data {ENV_PATH2D_CLIPS}  clips {len(clips)}  scored {len(test)}", flush=True)
    by_name: dict[str, list[dict]] = {name: [] for name in DECODERS}
    extras = []
    for row in test:
        decoded = _decode(row)
        item = {
            "clip_id": row["clip_id"],
            "vehicle": row["vehicle"],
            "corners": row["corners"],
            "speed_mps": row["speed"],
            "straight_residual_rmse_m": float(fit_straight_pass(row["radius"], DT)["residual_rmse_m"]),
        }
        for name in DECODERS:
            scores = score_decoder(decoded[name], row["xy"])
            item[name] = scores
            by_name[name].append(scores)
        extras.append(item)

    summary = {name: aggregate(rows) for name, rows in by_name.items()}
    n_turn = sum(1 for row in extras if row["corners"] > 0)
    branch_orbit = float(np.mean([row["corner_branch"]["orbit_rms_m"] for row in extras]))
    turn_orbit = float(
        np.mean([row["corner_branch"]["orbit_rms_m"] for row in extras if row["corners"] > 0])
        if n_turn
        else branch_orbit
    )
    straight_orbit = float(
        np.mean(
            [row["straight_family"]["orbit_rms_m"] for row in extras if row["corners"] == 0]
            or [0.0]
        )
    )
    payload = {
        "file_purpose": "Oracle-range geometric decoder. Exact GT R(t), no audio.",
        "n_test": len(test),
        "n_with_heading_jumps": n_turn,
        "dt_s": DT,
        "coordinate_frame": "canonical meters, microphone at the origin, observer-time centerline",
        "dataset": "$DOPPLER_PATH2D_CLIPS",
        "dataset_env": ENV_PATH2D_CLIPS,
        "n_discovered": len(clips),
        "wav_not_used": True,
        "summary": summary,
        "clips": extras,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2))

    ranked = sorted(extras, key=lambda row: row["corner_branch"]["orbit_rms_m"])
    show = []
    if ranked:
        show.append(ranked[len(ranked) // 2])
        turning = [row for row in ranked if row["corners"] > 0]
        if turning:
            show.append(turning[-1])
        show.append(max(extras, key=lambda row: row["straight_family"]["orbit_rms_m"]))
    seen = set()
    for item in show:
        if item["clip_id"] in seen:
            continue
        seen.add(item["clip_id"])
        row = next(r for r in test if r["clip_id"] == item["clip_id"])
        _plot(row, _decode(row), OUT / f"{item['clip_id']}.png")

    lines = [
        "# Oracle range decoder",
        "",
        "Exact ground-truth R(t) is the only observation. There is no audio and no learned model.",
        "The path family is stated next to each row. Orbit alignment is a global rotation and",
        "reflection about the microphone, with no translation, scale, or time warp.",
        "",
        f"Dataset: `$DOPPLER_PATH2D_CLIPS`. Discovered clips: {len(clips)}. Scored clips: {len(test)}"
        + (" (vehicle holdout test)." if args.holdout else " (full set; decoder does not train)."),
        f"Clips with a heading jump of at least 12 degrees: {n_turn}.",
        "Labels are the observer-time centerline in `canonical_state_frames.npy`.",
        "",
        "| Decoder | Motion assumption | Orbit | Range MAE | CPA |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "straight_family": "Constant-speed straight pass",
        "range_rdot_speed_true_sign": "Known speed, GT sign of θ̇",
        "range_rdot_speed_constant_sign": "Known speed, one constant sign",
        "range_rdot_no_speed": "Unknown speed, GT sign of θ̇",
        "global_sign_flip": "Known speed, global sign flip of theta-dot",
        "corner_branch": "Known speed, sign switch after first corner or CPA",
    }
    for name in DECODERS:
        block = summary[name]
        lines.append(
            f"| `{name}` | {labels[name]} | {_fmt(block['orbit_rms_m_mean'])} | "
            f"{_fmt(block['range_mae_m_mean'])} | {_fmt(block['cpa_abs_m_mean'])} |"
        )
    lines.extend(
        [
            "",
            f"On clips with no heading jump, the straight family orbit is {_fmt(straight_orbit)}.",
            f"Switching the sign of theta-dot after the first 12-degree heading jump, or at CPA if none, leaves orbit {_fmt(branch_orbit)}.",
            "Those branched paths share the radial profile. A global sign flip is a gauge and is removed by orbit alignment;",
            "a sign switch at a corner is a different shape.",
            "",
            "The reconstruction that is given the true sign of theta-dot and the true speed is the",
            "identifiability ceiling for a range-and-rate decoder. Everything else is a missing motion assumption.",
            "",
            "Figures: ground truth, the straight fit from R(t), and the same range with a sign switch at the first corner or CPA.",
            "",
        ]
    )
    for clip_id in seen:
        lines.append(f"![{clip_id}]({clip_id}.png)")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines))
    print(json.dumps({name: {k: summary[name][k] for k in ("orbit_rms_m_mean", "range_mae_m_mean", "cpa_abs_m_mean")} for name in DECODERS}, indent=2))
    print(f"wrote {OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
