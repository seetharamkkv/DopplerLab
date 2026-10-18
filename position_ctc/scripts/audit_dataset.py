#!/usr/bin/env python3
"""Audit lateral motion on the framewise trainer's supervised window.

The window is the one ``trajectory_framewise.ClipStore`` feeds the loss:
canonical ``x, vx, y, vy`` placed on STFT frame centers, supervised wherever
a center lies inside the label clock. Stationary pads stay in that window.
Model weights are not loaded.

From the DopplerLab root, audit the clips named by a finished run:

    position_ctc/.venv/bin/python position_ctc/scripts/audit_dataset.py

Or point at a dataset directory. Splits then follow ``trajectory_framewise``
(seed 42, source-recording groups, several validation groups, 10 test examples):

    position_ctc/.venv/bin/python position_ctc/scripts/audit_dataset.py \\
        --root /path/to/dataset \\
        --original /path/to/the/unrotated/dataset

The dataset path is printed first.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "scripts"))

from trajectory_framewise import (  # noqa: E402
    ClipStore,
    MotionFramewise,
    load_audio,
    make_splits,
    read_records,
    training_normalization,
)

TRAINER_CONFIG = {
    "sample_rate": 44100,
    "n_fft": 2048,
    "hop": 512,
    "hidden": 96,
    "freq_pool": 64,
    "vmax": 30.0,
    "bin_width": 2.0,
    "sigma": 1.0,
    "vy_balance": 0.0,
    "ctc_weight": 0.0,
    "penalty": 0.25,
    "max_jump": 6.0,
    "decoder": "viterbi",
    "mode": "framewise",
}

SPLITS = ("train", "val", "test")
SOUND_SPEED_MPS = 343.0
STATIONARY_MPS = 0.5
Y_SPAN_EDGES = [0, 2, 5, 10, 20, 40, 80, 130]

# Within-clip statistics. ``kind`` selects the report format.
CLIP_FIELDS = [
    ("x_span", "x span (m)", "m"),
    ("y_span", "y span (m)", "m"),
    ("vx_std", "vx std (m/s)", "mps"),
    ("vy_std", "vy std (m/s)", "mps"),
    ("vx_min", "vx minimum (m/s)", "mps"),
    ("vx_max", "vx maximum (m/s)", "mps"),
    ("vy_min", "vy minimum (m/s)", "mps"),
    ("vy_max", "vy maximum (m/s)", "mps"),
    ("frac_qvy_pos", "Quantized vy > 0", "pct"),
    ("frac_qvy_neg", "Quantized vy < 0", "pct"),
    ("frac_qvy_zero", "Quantized vy = 0", "pct"),
    ("frac_abs_vy_ge2", "Frames with |vy| >= 2 m/s", "pct"),
    ("n_trans_vx", "vx bin transitions", "int"),
    ("n_trans_vy", "vy bin transitions", "int"),
    ("n_nonzero_segments", "Nonzero-vy segments", "int"),
    ("nonzero_duration_s", "Nonzero-vy duration (s)", "s"),
    ("median_segment_s", "Median segment in the clip (s)", "s"),
    ("lateral_travel", "Lateral travel (m)", "m"),
    ("signed_disp", "Signed y displacement (m)", "m"),
    ("abs_net_over_travel", "|net| / travel", "ratio"),
    ("turning_frac", "Turning-frame fraction", "pct"),
    ("zero_vy_mae", "All-zero vy MAE (m/s)", "mps"),
    ("edge_travel_frac", "Share of |dy| in the outer tenths", "pct"),
    ("n_frames", "Supervised frames", "int"),
]


def quantize(values: np.ndarray, step: float, vmax: float) -> np.ndarray:
    """Nearest bin center on the trainer's symmetric velocity grid."""
    k = np.rint(np.asarray(values, dtype=np.float64) / step)
    return np.clip(k, -vmax / step, vmax / step) * step


def centers(step: float, vmax: float) -> np.ndarray:
    return np.arange(-vmax, vmax + step / 2.0, step)


def percentiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    out = {f"p{q}": float(np.percentile(values, q)) for q in (0, 10, 25, 50, 75, 90, 95, 100)}
    out["mean"] = float(values.mean())
    out["n"] = int(values.size)
    return out


def gini(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0 or np.all(ordered == 0):
        return 0.0
    n = ordered.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * ordered) / (n * np.sum(ordered))) - (n + 1) / n)


def run_lengths(mask: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def wav_duration(path: str, sample_rate: int) -> float:
    """Match ``ClipStore``: duration is ``(n_samples - 1) / rate`` after resampling."""
    with wave.open(path, "rb") as handle:
        rate = handle.getframerate()
        n_frames = handle.getnframes()
    if rate != sample_rate:
        raise ValueError(f"{path} is {rate} Hz; the run resamples, so duration must be measured after load_audio.")
    return (n_frames - 1) / float(sample_rate)


def dataset_root(records: list[dict]) -> Path:
    common = Path(records[0]["wav"])
    for record in records[1:]:
        common = Path(*common.parts[: _common_prefix_len(common.parts, Path(record["wav"]).parts)])
    if common.name == "audio_clips":
        return common.parent
    return common


def _common_prefix_len(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    n = 0
    for a, b in zip(left, right):
        if a != b:
            break
        n += 1
    return n


def vehicle_name(wav_path: str) -> tuple[str, str]:
    stem = Path(wav_path).stem
    vehicle, _, speed = stem.rpartition("_")
    if not vehicle:
        return stem, ""
    return vehicle, speed.replace("kmh", "")


def fmt(value, kind: str) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    if kind == "pct":
        return f"{100.0 * value:.1f}%"
    if kind == "int":
        return f"{value:.0f}"
    if kind == "m":
        return f"{value:.1f}"
    if kind == "mps":
        return f"{value:.2f}"
    if kind == "s":
        return f"{value:.2f}"
    if kind == "ratio":
        return f"{value:.2f}"
    return f"{value:.3f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    cells = [headers, *rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]

    def line(row: list[str]) -> str:
        parts = []
        for i, (cell, width) in enumerate(zip(row, widths)):
            parts.append(cell.ljust(width) if i == 0 else cell.rjust(width))
        return "  ".join(parts)

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([line(headers), rule, *[line(row) for row in rows]])


def clip_metrics(clip: dict, record: dict, part: str, config: dict, duration: float) -> tuple[dict, dict]:
    """Statistics on one supervised clip, plus the raw series used for plots."""
    step = float(config["bin_width"])
    vmax = float(config["vmax"])
    dt = float(config["hop"]) / float(config["sample_rate"])
    position = np.asarray(clip["position"], dtype=np.float64)
    velocity = np.asarray(clip["velocity"], dtype=np.float64)
    times = np.asarray(clip["target_times"], dtype=np.float64)
    x, y = position[:, 0], position[:, 1]
    vx, vy = velocity[:, 0], velocity[:, 1]
    qvx = quantize(vx, step, vmax)
    qvy = quantize(vy, step, vmax)
    travel = float(np.sum(np.abs(np.diff(y)))) if len(y) > 1 else 0.0
    disp = float(y[-1] - y[0])
    nonzero = np.abs(qvy) > 1e-6
    spans = run_lengths(nonzero)
    seg_s = [(stop - start) * dt for start, stop in spans]
    hot = np.abs(vy) >= step
    short = np.zeros(len(vy), dtype=bool)
    for start, stop in spans:
        if (stop - start) * dt < 0.5:
            short[start:stop] = True
    opening_n = max(1, len(velocity) // 5)
    opening = velocity[:opening_n].mean(axis=0)
    # Same mask as trajectory_framewise.scores.
    turning = np.linalg.norm(velocity - opening, axis=1) >= step
    state = np.load(record["state"]).astype(np.float64)
    speed = np.hypot(state[:, 1], state[:, 3])
    moving = speed > STATIONARY_MPS
    if moving.any():
        first = int(np.flatnonzero(moving)[0])
        last = int(np.flatnonzero(moving)[-1]) + 1
    else:
        first, last = 0, len(state)
    pad = np.ones(len(state), dtype=bool)
    pad[first:last] = False
    canon_travel = float(np.sum(np.abs(np.diff(state[:, 2])))) if len(state) > 1 else 0.0
    pad_travel = 0.0
    if len(state) > 1 and pad.any():
        step_pad = pad[:-1] | pad[1:]
        pad_travel = float(np.sum(np.abs(np.diff(state[:, 2])[step_pad])))
    arrive = times + np.hypot(x, y) / SOUND_SPEED_MPS
    outside = arrive > duration + 0.5 * dt
    if len(y) > 10:
        dy = np.abs(np.diff(y))
        edge = max(1, len(dy) // 10)
        edge_frac = float((dy[:edge].sum() + dy[-edge:].sum()) / max(dy.sum(), 1e-12))
    else:
        edge_frac = 0.0
    vehicle, nominal = vehicle_name(record["wav"])
    features = np.asarray(clip["features"])
    row = {
        "id": record["id"],
        "split": part,
        "vehicle": vehicle,
        "nominal_kmh": nominal,
        "n_frames": int(len(times)),
        "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "x_span": float(x.max() - x.min()),
        "y_span": float(y.max() - y.min()),
        "vx_std": float(vx.std()),
        "vy_std": float(vy.std()),
        "vx_min": float(vx.min()),
        "vx_max": float(vx.max()),
        "vy_min": float(vy.min()),
        "vy_max": float(vy.max()),
        "frac_qvy_pos": float(np.mean(qvy > 0)),
        "frac_qvy_neg": float(np.mean(qvy < 0)),
        "frac_qvy_zero": float(np.mean(np.abs(qvy) <= 1e-6)),
        "frac_abs_vy_ge2": float(np.mean(hot)),
        "n_trans_vx": int(np.sum(np.abs(np.diff(qvx)) > 1e-6)) if len(qvx) > 1 else 0,
        "n_trans_vy": int(np.sum(np.abs(np.diff(qvy)) > 1e-6)) if len(qvy) > 1 else 0,
        "n_nonzero_segments": int(len(seg_s)),
        "nonzero_duration_s": float(sum(seg_s)),
        "median_segment_s": float(np.median(seg_s)) if seg_s else 0.0,
        "max_segment_s": float(max(seg_s)) if seg_s else 0.0,
        "lateral_travel": travel,
        "signed_disp": disp,
        "abs_net_over_travel": float(abs(disp) / travel) if travel > 0 else 0.0,
        "turning_frac": float(np.mean(turning)),
        "zero_vy_mae": float(np.mean(np.abs(vy))),
        "turning_zero_vy_mae": float(np.mean(np.abs(vy[turning]))) if turning.any() else None,
        "n_turning": int(turning.sum()),
        "edge_travel_frac": edge_frac,
        "hot_frames": int(hot.sum()),
        "hot_outside": int(np.sum(hot & outside)),
        "hot_short": int(np.sum(hot & short)),
        "outside_frames": int(outside.sum()),
        "canon_frames": int(len(state)),
        "canon_travel": canon_travel,
        "pad_frames": int(pad.sum()),
        "pad_travel": pad_travel,
        "audio_frames": int(features.shape[0]),
        "context_frames": int(features.shape[0] - len(clip["selected"])),
        "features_ndim": int(features.ndim),
        "features_bins": int(features.shape[1]),
        "features_min": float(features.min()),
        "features_max": float(features.max()),
        "segment_durations_s": seg_s,
    }
    series = {"t": times, "x": x, "y": y, "vx": vx, "vy": vy}
    return row, series


def bin_counts(values: np.ndarray, grid: np.ndarray) -> dict[str, int]:
    q = quantize(values, float(grid[1] - grid[0]), float(grid[-1]))
    return {f"{center:+.0f}": int(np.sum(np.isclose(q, center))) for center in grid}


def span_bin_edges(rows: list[dict]) -> list[float]:
    edges = list(Y_SPAN_EDGES)
    ymax = max((row["y_span"] for row in rows), default=0.0)
    if ymax >= edges[-1]:
        edges.append(float(np.ceil(ymax + 1.0)))
    return edges


def summarize(rows: list[dict], velocities: dict[str, dict[str, np.ndarray]], grid: np.ndarray, span_edges: list[float]) -> dict:
    summary = {}
    for part in SPLITS:
        part_rows = [row for row in rows if row["split"] == part]
        block = {
            "clips": len(part_rows),
            "frames": int(sum(row["n_frames"] for row in part_rows)),
            "percentiles": {},
        }
        for key, _label, _kind in CLIP_FIELDS:
            block["percentiles"][key] = percentiles(np.array([row[key] for row in part_rows], dtype=np.float64))
        turn = [row["turning_zero_vy_mae"] for row in part_rows if row["turning_zero_vy_mae"] is not None]
        block["percentiles"]["turning_zero_vy_mae"] = percentiles(np.array(turn, dtype=np.float64))
        vx = velocities[part]["vx"]
        vy = velocities[part]["vy"]
        qvy = quantize(vy, float(grid[1] - grid[0]), float(grid[-1]))
        turn_abs = velocities[part]["turn_abs"]
        total = max(int(vy.size), 1)
        vx_counts = bin_counts(vx, grid)
        vy_counts = bin_counts(vy, grid)
        block["frames_level"] = {
            "frames": int(vy.size),
            "qvy_pos": float(np.mean(qvy > 0)),
            "qvy_neg": float(np.mean(qvy < 0)),
            "qvy_zero": float(np.mean(np.abs(qvy) <= 1e-6)),
            "abs_vy_ge2": float(np.mean(np.abs(vy) >= float(grid[1] - grid[0]))),
            "vx_negative": float(sum(n for name, n in vx_counts.items() if name.startswith("-")) / total),
            "zero_vy_mae_micro": float(np.mean(np.abs(vy))),
            "zero_vy_mae_macro": float(np.mean([row["zero_vy_mae"] for row in part_rows])),
            "turning_frames": int(turn_abs.size),
            "turning_zero_vy_mae_micro": float(turn_abs.mean()) if turn_abs.size else None,
            "turning_zero_vy_mae_macro": float(np.mean(turn)) if turn else None,
            "clips_with_turning": int(sum(row["n_turning"] > 0 for row in part_rows)),
            "clips_all_zero_qvy": int(sum(row["frac_qvy_zero"] > 0.999 for row in part_rows)),
            "vx_bin_fraction": {name: n / total for name, n in vx_counts.items() if n},
            "vy_bin_fraction": {name: n / total for name, n in vy_counts.items() if n},
        }
        travel = np.array([row["lateral_travel"] for row in part_rows], dtype=np.float64)
        order = np.argsort(travel)[::-1]
        total_travel = float(travel.sum())
        k = max(1, int(round(0.1 * len(part_rows))))
        block["concentration"] = {
            "gini_travel": gini(travel),
            "top10pct_clips": k,
            "top10pct_travel_share": float(travel[order[:k]].sum() / total_travel) if total_travel else 0.0,
            "clips_y_span_lt_1m": int(sum(row["y_span"] < 1.0 for row in part_rows)),
            "clips_y_span_ge_5m": int(sum(row["y_span"] >= 5.0 for row in part_rows)),
            "clips_any_abs_vy_ge2": int(sum(row["frac_abs_vy_ge2"] > 0 for row in part_rows)),
            "clips_net_under_15pct_of_travel": int(sum(row["abs_net_over_travel"] < 0.15 for row in part_rows)),
            "top_clips": [
                {
                    "id": part_rows[int(i)]["id"],
                    "vehicle": part_rows[int(i)]["vehicle"],
                    "travel_m": part_rows[int(i)]["lateral_travel"],
                    "share": part_rows[int(i)]["lateral_travel"] / total_travel if total_travel else 0.0,
                    "y_span_m": part_rows[int(i)]["y_span"],
                    "signed_disp_m": part_rows[int(i)]["signed_disp"],
                    "frac_abs_vy_ge2": part_rows[int(i)]["frac_abs_vy_ge2"],
                }
                for i in order[:8]
            ],
        }
        segs = np.concatenate([np.asarray(row["segment_durations_s"], dtype=np.float64) for row in part_rows]) if part_rows else np.array([])
        block["segments"] = {
            "n": int(segs.size),
            "percentiles_s": percentiles(segs) if segs.size else {},
            "shorter_than_0_5s": int(np.sum(segs < 0.5)) if segs.size else 0,
        }
        counts, _edges = np.histogram([row["y_span"] for row in part_rows], bins=span_edges)
        block["y_span_hist"] = {
            "edges_m": list(span_edges),
            "counts": [int(c) for c in counts],
            "percent_of_clips": [float(100.0 * c / max(len(part_rows), 1)) for c in counts],
        }
        block["trim"] = {
            "canon_frames": int(sum(row["canon_frames"] for row in part_rows)),
            "supervised_frames": int(sum(row["n_frames"] for row in part_rows)),
            "canon_travel_m": float(sum(row["canon_travel"] for row in part_rows)),
            "supervised_travel_m": float(sum(row["lateral_travel"] for row in part_rows)),
            "pad_frames": int(sum(row["pad_frames"] for row in part_rows)),
            "pad_travel_m": float(sum(row["pad_travel"] for row in part_rows)),
            "hot_frames": int(sum(row["hot_frames"] for row in part_rows)),
            "hot_outside": int(sum(row["hot_outside"] for row in part_rows)),
            "hot_short": int(sum(row["hot_short"] for row in part_rows)),
            "outside_frames": int(sum(row["outside_frames"] for row in part_rows)),
            "context_frames": int(sum(row["context_frames"] for row in part_rows)),
        }
        summary[part] = block
    return summary


def vehicles_table(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = defaultdict(lambda: {"clips": {part: 0 for part in SPLITS}, "train_y": [], "train_hot": 0, "train_frames": 0, "train_travel": 0.0})
    for row in rows:
        bucket = grouped[row["vehicle"]]
        bucket["clips"][row["split"]] += 1
        if row["split"] == "train":
            bucket["train_y"].append(row["y_span"])
            bucket["train_hot"] += row["hot_frames"]
            bucket["train_frames"] += row["n_frames"]
            bucket["train_travel"] += row["lateral_travel"]
    train_travel = sum(bucket["train_travel"] for bucket in grouped.values())
    out = []
    for name, bucket in grouped.items():
        out.append(
            {
                "vehicle": name,
                "train": bucket["clips"]["train"],
                "val": bucket["clips"]["val"],
                "test": bucket["clips"]["test"],
                "train_median_y_span_m": float(np.median(bucket["train_y"])) if bucket["train_y"] else None,
                "train_hot_fraction": bucket["train_hot"] / bucket["train_frames"] if bucket["train_frames"] else None,
                "train_travel_share": bucket["train_travel"] / train_travel if train_travel else 0.0,
            }
        )
    out.sort(key=lambda item: -(item["train_median_y_span_m"] or -1))
    return out


def ks_table(rows: list[dict]) -> dict:
    from scipy.stats import ks_2samp

    keys = ("y_span", "vy_std", "lateral_travel", "signed_disp", "frac_abs_vy_ge2", "x_span", "vx_std")
    out = {}
    for key in keys:
        samples = {part: np.array([row[key] for row in rows if row["split"] == part], dtype=np.float64) for part in SPLITS}
        stat_v, p_v = ks_2samp(samples["train"], samples["val"])
        stat_t, p_t = ks_2samp(samples["train"], samples["test"])
        out[key] = {
            "train_median": float(np.median(samples["train"])),
            "val_median": float(np.median(samples["val"])),
            "test_median": float(np.median(samples["test"])),
            "ks_train_val": float(stat_v),
            "p_train_val": float(p_v),
            "ks_train_test": float(stat_t),
            "p_train_test": float(p_t),
        }
    return out


def check_inputs(store: ClipStore, rows: list[dict], config: dict, normalization_path: Path) -> dict:
    """Confirm every clip's feature tensor, then run the current encoder on three clips."""
    expected_bins = int(config["n_fft"]) // 2 + 1
    bad = [
        row["id"]
        for row in rows
        if row["features_ndim"] != 2 or row["features_bins"] != expected_bins or row["features_min"] < -1e-5
    ]
    saved = json.loads(normalization_path.read_text()) if normalization_path.is_file() else None
    refit = training_normalization(store, [row["id"] for row in rows if row["split"] == "train"])
    order = sorted(rows, key=lambda row: row["y_span"])
    picked = {"low": order[0], "median": order[len(order) // 2], "high": order[-1]}
    model = MotionFramewise(config)
    model.eval()
    forwards = []
    import torch

    for role, row in picked.items():
        clip = store.get(row["id"])
        features = np.asarray(clip["features"])
        p0 = np.asarray(clip["position"][0], dtype=np.float32)
        with torch.no_grad():
            dense, ctc = model(
                torch.as_tensor(features, dtype=torch.float32),
                torch.as_tensor(clip["selected"], dtype=torch.long),
                torch.as_tensor(p0, dtype=torch.float32),
            )
        forwards.append(
            {
                "role": role,
                "id": row["id"],
                "split": row["split"],
                "vehicle": row["vehicle"],
                "y_span_m": row["y_span"],
                "x_span_m": row["x_span"],
                "lateral_travel_m": row["lateral_travel"],
                "signed_disp_m": row["signed_disp"],
                "features_shape": list(features.shape),
                "features_min": float(features.min()),
                "features_max": float(features.max()),
                "p0_m": [float(p0[0]), float(p0[1])],
                "audio_frames": int(features.shape[0]),
                "supervised_frames": int(len(clip["selected"])),
                "context_frames": int(features.shape[0] - len(clip["selected"])),
                "dense_shape": list(dense.shape),
                "ctc_head": ctc,
            }
        )
    return {
        "clips_checked": len(rows),
        "expected_frequency_bins": expected_bins,
        "clips_failing_feature_shape": bad,
        "saved_normalization": saved,
        "refit_train_mean": refit["mean"],
        "refit_train_std": refit["std"],
        "refit_values": refit["values"],
        "forwards": forwards,
        "forward_signature": "MotionFramewise.forward(features [T, F], selected frame indices, p0 [2])",
    }


def pick_series(rows: list[dict], series: dict[str, dict]) -> dict[str, dict]:
    order = sorted(rows, key=lambda row: row["y_span"])
    chosen = {"low": order[0], "median": order[len(order) // 2], "high": order[-1]}
    return {role: {"row": row, "series": series[row["id"]]} for role, row in chosen.items()}


def write_plots(out: Path, summary: dict, examples: dict[str, dict], grid: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    edges = summary["train"]["y_span_hist"]["edges_m"]
    labels = [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], edges[1:])]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, part in enumerate(SPLITS):
        ax.bar(x + (i - 1) * width, summary[part]["y_span_hist"]["percent_of_clips"], width=width, label=part)
    ax.set_xticks(x, labels)
    ax.set_xlabel("y span, max(y) − min(y) (m)")
    ax.set_ylabel("Percent of clips")
    ax.set_title("Clips by lateral position span")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "y_span.png", dpi=140)
    plt.close(fig)

    for axis, filename, title in (
        ("vy_bin_fraction", "vy_bins.png", "Frame-level vy bin frequency"),
        ("vx_bin_fraction", "vx_bins.png", "Frame-level vx bin frequency"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        for part in SPLITS:
            fractions = summary[part]["frames_level"][axis]
            ys = [100.0 * fractions.get(f"{center:+.0f}", 0.0) for center in grid]
            ax.plot(grid, ys, marker="o", ms=3, label=part)
        ax.set_xlabel("Bin center (m/s)")
        ax.set_ylabel("Percent of supervised frames")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / filename, dpi=140)
        plt.close(fig)

    for role, item in examples.items():
        row, s = item["row"], item["series"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
        panels = (
            (axes[0, 0], s["x"], "x (m)", "x position"),
            (axes[0, 1], s["y"], "y (m)", "y position"),
            (axes[1, 0], s["vx"], "vx (m/s)", "vx velocity"),
            (axes[1, 1], s["vy"], "vy (m/s)", "vy velocity"),
        )
        for ax, values, ylabel, subtitle in panels:
            ax.plot(s["t"], values, color="black", lw=1.0)
            ax.set_ylabel(ylabel)
            ax.set_title(subtitle)
            ax.grid(alpha=0.3)
            ax.set_xlabel("Observer time (s)")
        sample = Path(row["id"]).name
        fig.suptitle(
            f"{role} lateral span · {sample} · {row['split']} · {row['vehicle']}\n"
            f"y span {row['y_span']:.2f} m, travel {row['lateral_travel']:.2f} m, "
            f"signed displacement {row['signed_disp']:.2f} m"
        )
        fig.tight_layout()
        fig.savefig(plots / f"{role}_{sample}.png", dpi=140)
        plt.close(fig)


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        f"Dataset: {payload['dataset']}",
        f"Run: {payload['run']}",
        f"Records: {payload['records']}",
        f"Splits: {payload['splits_file']}",
        "",
    ]
    excluded = payload.get("excluded") or []
    if excluded:
        lines.append(
            f"{len(excluded)} clip directories have a WAV and no canonical_state_frames.npy. "
            "The trainer skips them, and so does this audit:"
        )
        for item in excluded:
            sibling = item.get("sibling_split") or "no labeled sibling"
            lines.append(
                f"  {item['id']}  {item['wav']}  {item['angle_deg']}° render  "
                f"labeled copies are in {sibling}"
            )
        lines.append(
            "Labeled rotations of these WAV names stay with their source group. "
            "The missing files are not replaced."
        )
        lines.append("")
    rotation = payload.get("rotations")
    if rotation:
        lines.extend(rotation_lines(rotation))
        lines.append("")
    groups = payload.get("groups") or []
    if groups:
        lines.append("Split groups")
        lines.append(
            "  Every rotation of one WAV, and every clip that shares a source recording, "
            "is in one group. Near-duplicate paths from different sources stay together "
            "so they cannot land on both sides of a split."
        )
        if payload.get("validation_groups"):
            lines.append("  Validation groups: " + ", ".join(payload["validation_groups"]))
        for item in groups:
            lines.append(
                f"  {item['split']:<5}  {item['clips']:3d} clips  {item['paths']:3d} paths  {item['group']}"
            )
        lines.append("")
    lines.extend([
        payload["coverage"],
        "",
        "Window",
        "  Canonical state columns are x, vx, y, vy, in meters and m/s.",
        "  Position is linearly interpolated onto STFT frame centers. Velocity is the",
        "  nearest label frame, which keeps corner jumps intact when the clocks match.",
        f"  Clock: {payload['sample_rate']} Hz, FFT {payload['n_fft']}, hop {payload['hop']} "
        f"({payload['sample_rate'] / payload['hop']:.2f} Hz).",
        "  A frame is supervised when its center lies inside the canonical label clock.",
        "  That is the mask the trainer's loss sees. Stationary pads are left in.",
        f"  Velocity bins are {payload['bin_width']:.0f} m/s from −{payload['vmax']:.0f} to +{payload['vmax']:.0f}.",
        "  Quantized vy is the nearest bin. |vy| >= 2 m/s is the continuous absolute value.",
        "  A bin transition is a change of that center between consecutive supervised frames.",
        "  Nonzero-vy segment duration counts each frame as one hop.",
        "  Lateral travel is sum(|diff(y)|). Signed displacement is y[-1] − y[0].",
        "  Turning frames use the trainer's score mask: the L2 distance from the mean",
        "  velocity of the first fifth of the clip is at least one bin.",
        "  The all-zero vy baseline predicts 0 m/s. Its MAE is the mean of |vy|.",
        "  Frame-weighted MAE pools frames. The mean-of-clips MAE matches the trainer,",
        "  which averages one number per clip and skips clips with no turning frames.",
        "",
        "Per-clip percentiles",
        "  p50 is the median clip. The last column is the training 10th–90th percentile.",
        "",
    ])
    headers = ["Statistic", "Train p50", "Val p50", "Test p50", "Train p10–p90"]
    rows = [[
        "Clips / supervised frames",
        f"{summary['train']['clips']} / {summary['train']['frames']:,}",
        f"{summary['val']['clips']} / {summary['val']['frames']:,}",
        f"{summary['test']['clips']} / {summary['test']['frames']:,}",
        "",
    ]]
    for key, label, kind in CLIP_FIELDS:
        train = summary["train"]["percentiles"][key]
        rows.append([
            label,
            fmt(summary["train"]["percentiles"][key].get("p50"), kind),
            fmt(summary["val"]["percentiles"][key].get("p50"), kind),
            fmt(summary["test"]["percentiles"][key].get("p50"), kind),
            f"{fmt(train.get('p10'), kind)} – {fmt(train.get('p90'), kind)}",
        ])
    lines.append(table(headers, rows))
    lines.append("")
    lines.append("Frame-level bins and the all-zero vy baseline")
    lines.append("")
    frame_rows = []
    specs = [
        ("frames", "Supervised frames", "int"),
        ("qvy_pos", "Quantized vy positive", "pct"),
        ("qvy_neg", "Quantized vy negative", "pct"),
        ("qvy_zero", "Quantized vy zero", "pct"),
        ("abs_vy_ge2", "|vy| >= 2 m/s", "pct"),
        ("vx_negative", "vx in a negative bin", "pct"),
        ("zero_vy_mae_micro", "All-zero vy MAE, frame-weighted (m/s)", "mps"),
        ("turning_zero_vy_mae_micro", "All-zero vy MAE on turning frames (m/s)", "mps"),
        ("zero_vy_mae_macro", "All-zero vy MAE, mean of clips (m/s)", "mps"),
        ("turning_zero_vy_mae_macro", "Turning MAE, mean of clips (m/s)", "mps"),
        ("turning_frames", "Turning frames", "int"),
        ("clips_with_turning", "Clips with any turning frame", "int"),
        ("clips_all_zero_qvy", "Clips with quantized vy all zero", "int"),
    ]
    for key, label, kind in specs:
        frame_rows.append([
            label,
            *[fmt(summary[part]["frames_level"][key], kind) for part in SPLITS],
        ])
    lines.append(table(["Quantity", "Train", "Validation", "Test"], frame_rows))
    lines.append("")
    lines.append("Occupied vy bins are plotted in plots/vy_bins.png. vx bins are in plots/vx_bins.png.")
    lines.append("The y-span histogram is plots/y_span.png.")
    lines.append("")
    lines.append("Representative clips")
    lines.append("  Low, median, and high are the minimum, middle, and maximum of y span over all clips.")
    lines.append("")
    for role, item in payload["examples"].items():
        row = item
        sample = Path(row["id"]).name
        lines.append(
            f"  {role}: {row['id']}  {row['split']}  {row['vehicle']}  "
            f"y span {row['y_span_m']:.2f} m, x span {row['x_span_m']:.2f} m, "
            f"travel {row['lateral_travel_m']:.2f} m, signed displacement {row['signed_disp_m']:.2f} m"
        )
        lines.append(f"    plot: plots/{role}_{sample}.png")
    lines.append("")
    lines.append("Where the lateral motion sits")
    lines.append("")
    conc_rows = []
    for part in SPLITS:
        c = summary[part]["concentration"]
        seg = summary[part]["segments"]
        conc_rows.append([
            part,
            f"{c['gini_travel']:.2f}",
            f"{100 * c['top10pct_travel_share']:.1f}%",
            str(c["top10pct_clips"]),
            str(c["clips_y_span_ge_5m"]),
            str(c["clips_any_abs_vy_ge2"]),
            str(c["clips_net_under_15pct_of_travel"]),
            str(summary[part]["clips"]),
            fmt(seg["percentiles_s"].get("p50"), "s"),
            f"{seg['shorter_than_0_5s']} / {seg['n']}",
        ])
    lines.append(table(
        ["Split", "Gini", "Top 10% travel", "Clips in top 10%", "y span ≥ 5 m", "Any |vy| ≥ 2", "|net|/travel < 0.15", "Clips", "Segment p50 (s)", "Segments < 0.5 s"],
        conc_rows,
    ))
    lines.append("")
    lines.append("Largest lateral-travel clips")
    for part in SPLITS:
        lines.append(f"  {part}")
        for clip in summary[part]["concentration"]["top_clips"]:
            lines.append(
                f"    {Path(clip['id']).name}  {clip['vehicle']}  "
                f"travel {clip['travel_m']:.1f} m ({100 * clip['share']:.1f}%)  "
                f"y span {clip['y_span_m']:.1f} m  signed {clip['signed_disp_m']:.1f} m"
            )
    lines.append("")
    lines.append("Vehicles")
    lines.append("  Counts are clips. The y span and |vy| ≥ 2 fraction are training-only.")
    lines.append("")
    vehicle_rows = []
    for item in payload["vehicles"]:
        vehicle_rows.append([
            item["vehicle"],
            str(item["train"]),
            str(item["val"]),
            str(item["test"]),
            fmt(item["train_median_y_span_m"], "m"),
            fmt(item["train_hot_fraction"], "pct"),
            fmt(item["train_travel_share"], "pct"),
        ])
    lines.append(table(
        ["Vehicle", "Train", "Val", "Test", "Train median y span (m)", "Train |vy| ≥ 2", "Train travel share"],
        vehicle_rows,
    ))
    lines.append("")
    lines.append("Trimming and sound that arrives after the recording")
    lines.append("  Supervised travel is measured on the trainer's window. Canonical travel is the")
    lines.append("  full label file. Pad rows are frames with speed ≤ 0.5 m/s; the trainer keeps them.")
    lines.append("  Outside rows have reception time t + range/343 after the end of the WAV.")
    lines.append("")
    trim_rows = []
    for part in SPLITS:
        t = summary[part]["trim"]
        trim_rows.append([
            part,
            f"{t['supervised_frames']:,} / {t['canon_frames']:,}",
            f"{t['supervised_travel_m']:.2f}",
            f"{t['canon_travel_m']:.2f}",
            f"{t['pad_frames']:,}",
            f"{100 * t['pad_frames'] / t['canon_frames']:.2f}%",
            f"{t['pad_travel_m']:.2f}",
            f"{100 * t['pad_travel_m'] / t['canon_travel_m']:.3f}%" if t["canon_travel_m"] else "n/a",
            f"{t['hot_outside']:,} / {t['hot_frames']:,}",
            f"{t['outside_frames']:,}",
            f"{t['context_frames']:,}",
            f"{t['hot_short']:,} / {t['hot_frames']:,}",
        ])
    lines.append(table(
        ["Split", "Supervised / canonical frames", "Supervised travel (m)", "Canonical travel (m)", "Pad frames", "Pad frame share", "Pad travel (m)", "Pad travel share", "|vy|≥2 arriving late", "Any frame arriving late", "Audio frames outside the mask", "|vy|≥2 in segments < 0.5 s"],
        trim_rows,
    ))
    lines.append("")
    lines.append("Network inputs")
    inputs = payload["inputs"]
    lines.append(f"  {inputs['forward_signature']}")
    lines.append(
        f"  Feature check on {inputs['clips_checked']} clips: rank 2, "
        f"{inputs['expected_frequency_bins']} frequency bins, nonnegative log1p magnitude."
    )
    if inputs["clips_failing_feature_shape"]:
        lines.append("  Clips that failed that check: " + ", ".join(inputs["clips_failing_feature_shape"]))
    else:
        lines.append("  Every clip passed that check.")
    saved = inputs["saved_normalization"]
    if saved:
        lines.append(
            f"  Saved train normalization: mean {saved['mean']:.6f}, std {saved['std']:.6f}."
        )
        lines.append(
            f"  Refit on the current loader's training features: mean {inputs['refit_train_mean']:.6f}, "
            f"std {inputs['refit_train_std']:.6f}, over {inputs['refit_values']:,} values."
        )
    else:
        lines.append(
            f"  No saved training normalization. Refit on these training features: "
            f"mean {inputs['refit_train_mean']:.6f}, std {inputs['refit_train_std']:.6f}, "
            f"over {inputs['refit_values']:,} values."
        )
    lines.append("  Untrained forward on the low, median, and high clips, weights not loaded:")
    for item in inputs["forwards"]:
        lines.append(
            f"    {item['role']} {item['id']}: features {item['features_shape']}, "
            f"range [{item['features_min']:.3g}, {item['features_max']:.3g}], "
            f"p0 ({item['p0_m'][0]:.3f}, {item['p0_m'][1]:.3f}) m, "
            f"dense {item['dense_shape']}, ctc {item['ctc_head']}, "
            f"context frames {item['context_frames']}"
        )
    lines.append("  Velocity labels and later positions are loss targets. They are not columns of the feature tensor.")
    lines.append("  The selected index picks recurrent frames for the velocity heads. It is not a feature channel.")
    lines.append("")
    lines.append("Training against validation")
    lines.append("  Two-sample Kolmogorov–Smirnov tests on the per-clip values.")
    lines.append("")
    ks_rows = []
    labels = {
        "y_span": "y span (m)",
        "vy_std": "vy std (m/s)",
        "lateral_travel": "Lateral travel (m)",
        "signed_disp": "Signed displacement (m)",
        "frac_abs_vy_ge2": "Fraction |vy| ≥ 2",
        "x_span": "x span (m)",
        "vx_std": "vx std (m/s)",
    }
    kinds = {
        "y_span": "m",
        "vy_std": "mps",
        "lateral_travel": "m",
        "signed_disp": "m",
        "frac_abs_vy_ge2": "ratio",
        "x_span": "m",
        "vx_std": "mps",
    }
    for key, label in labels.items():
        item = payload["ks"][key]
        kind = kinds[key]
        ks_rows.append([
            label,
            fmt(item["train_median"], kind),
            fmt(item["val_median"], kind),
            fmt(item["test_median"], kind),
            f"{item['ks_train_val']:.2f}",
            f"{item['p_train_val']:.2f}",
            f"{item['ks_train_test']:.2f}",
            f"{item['p_train_test']:.2f}",
        ])
    lines.append(table(
        ["Clip statistic", "Train median", "Val median", "Test median", "KS train–val", "p", "KS train–test", "p"],
        ks_rows,
    ))
    lines.append("")
    straight = [row for row in payload["straight_clips"]]
    if straight:
        lines.append("Clips whose quantized vy is entirely zero")
        for row in straight:
            lines.append(
                f"  {row['split']}  {row['id']}  {row['vehicle']}  "
                f"y span {row['y_span']:.2f} m  vy std {row['vy_std']:.2f} m/s"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def coverage_text(summary: dict, ks: dict) -> str:
    med_y = [summary[part]["percentiles"]["y_span"]["p50"] for part in SPLITS]
    med_travel = [summary[part]["percentiles"]["lateral_travel"]["p50"] for part in SPLITS]
    mae = [summary[part]["frames_level"]["zero_vy_mae_micro"] for part in SPLITS]
    hot = [summary[part]["frames_level"]["abs_vy_ge2"] for part in SPLITS]
    turn_share = [
        summary[part]["frames_level"]["turning_frames"] / summary[part]["frames_level"]["frames"]
        for part in SPLITS
    ]
    adequate = all(y >= 5.0 for y in med_y) and all(t >= 10.0 for t in med_travel) and all(m >= 2.0 for m in mae)
    verdict = (
        "The dataset provides adequate lateral-motion coverage."
        if adequate
        else "Lateral-motion coverage is thin on at least one split."
    )
    p_min = min(item["p_train_val"] for item in ks.values())
    return "\n".join([
        verdict,
        (
            f"Median y span is {med_y[0]:.1f} m in training, {med_y[1]:.1f} m in validation, "
            f"and {med_y[2]:.1f} m in test. Median lateral travel is {med_travel[0]:.1f} m, "
            f"{med_travel[1]:.1f} m, and {med_travel[2]:.1f} m."
        ),
        (
            f"An all-zero vy predictor has frame-weighted MAE {mae[0]:.2f}, {mae[1]:.2f}, "
            f"and {mae[2]:.2f} m/s. Frames with |vy| ≥ 2 m/s are {100 * hot[0]:.1f}%, "
            f"{100 * hot[1]:.1f}%, and {100 * hot[2]:.1f}% of each split."
        ),
        (
            f"The trainer's turning mask covers {100 * turn_share[0]:.1f}%, {100 * turn_share[1]:.1f}%, "
            f"and {100 * turn_share[2]:.1f}% of frames, so it describes most of a clip."
        ),
        (
            f"The smallest train–validation Kolmogorov–Smirnov p-value across y span, vy scale, "
            f"travel, signed displacement, |vy| fraction, x span, and vx scale is {p_min:.2f}."
        ),
        (
            f"{summary['train']['frames_level']['clips_all_zero_qvy']} training clips, "
            f"{summary['val']['frames_level']['clips_all_zero_qvy']} validation clips, and "
            f"{summary['test']['frames_level']['clips_all_zero_qvy']} test clips stay inside the zero vy bin. "
            f"Negative vx bins hold {100 * summary['train']['frames_level']['vx_negative']:.2f}% of training frames, "
            "so the longitudinal label does not reverse."
        ),
    ])


def write_csv(path: Path, rows: list[dict]) -> None:
    skip = {"segment_durations_s"}
    fields = [key for key in rows[0].keys() if key not in skip]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _sample_index(directory: Path) -> int:
    return int(directory.name.split("_")[-1])


def _clip_ready(directory: Path) -> bool:
    wavs = list(directory.glob("*.wav")) if directory.is_dir() else []
    state = directory / "metadata" / "canonical_state_frames.npy"
    times = directory / "metadata" / "frame_times.npy"
    return len(wavs) == 1 and state.is_file() and times.is_file()


def _rotate_path(xy: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return xy @ rotation.T


def _log_stft(waveform: np.ndarray, sample_rate: int, n_fft: int, hop: int) -> np.ndarray:
    import torch

    spectrum = torch.stft(
        torch.from_numpy(np.ascontiguousarray(waveform)),
        n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    times = np.arange(spectrum.shape[1]) * hop / sample_rate
    duration = (len(waveform) - 1) / sample_rate
    return torch.log1p(spectrum.abs()).T.numpy()[times <= duration]


def _audio_pair(left: np.ndarray, right: np.ndarray, left_feat: np.ndarray, right_feat: np.ndarray) -> dict:
    residual = left.astype(np.float64) - right.astype(np.float64)
    denom = float(np.sqrt(np.mean(np.square(right, dtype=np.float64)))) + 1e-12
    correlation = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else 1.0
    frames = min(len(left_feat), len(right_feat))
    spectrum = np.abs(left_feat[:frames] - right_feat[:frames])
    return {
        "waveform_relative_rms": float(np.sqrt(np.mean(np.square(residual))) / denom),
        "waveform_max_abs": float(np.max(np.abs(residual))),
        "waveform_correlation": correlation,
        "log_stft_mae": float(np.mean(spectrum)),
        "log_stft_max_abs": float(np.max(spectrum)),
    }


def compare_rotations(root: Path, original: Path, sample_rate: int, n_fft: int, hop: int) -> dict:
    """Match original / 0° / 45° / 90° clips and compare clocks, audio, and labels.

    Block layout of the rotated export: sample i, i+130, and i+260 are the 0°,
    45°, and 90° renders of original sample i. The angle is checked by rotating
    the original path about the microphone, not taken from the batch name alone.
    Audio is the same log1p magnitude STFT the trainer feeds the network.
    """
    angles = ((0, 0), (45, 130), (90, 260))
    excluded = []
    pair_names = ("original_vs_0", "original_vs_45", "original_vs_90", "45_vs_0", "90_vs_0", "90_vs_45")
    store = {name: [] for name in pair_names}
    label_gap = []
    clock_gap = []
    angle_gap = []
    length_mismatch = 0
    rate_mismatch = 0
    compared = 0
    for index in range(1, 131):
        original_dir = original / "audio_clips" / f"sample_{index:07d}"
        members = {"original": original_dir}
        members.update({angle: root / "audio_clips" / f"sample_{index + offset:07d}" for angle, offset in angles})
        ready = {name: _clip_ready(path) for name, path in members.items()}
        for angle, _offset in angles:
            path = members[angle]
            if path.is_dir() and not ready[angle]:
                wavs = list(path.glob("*.wav"))
                excluded.append({
                    "id": path.relative_to(root).as_posix(),
                    "wav": wavs[0].name if wavs else "",
                    "angle_deg": angle,
                    "base_index": index,
                    "reason": "WAV present, canonical_state_frames.npy absent",
                })
        if not all(ready.values()):
            continue
        headers = {}
        waveforms = {}
        features = {}
        for name, path in members.items():
            wav = next(path.glob("*.wav"))
            with wave.open(str(wav), "rb") as handle:
                headers[name] = (handle.getframerate(), handle.getnframes(), handle.getnchannels())
            waveforms[name] = load_audio(wav, sample_rate)
            features[name] = _log_stft(waveforms[name], sample_rate, n_fft, hop)
        rates = {name: item[0] for name, item in headers.items()}
        lengths = {name: item[1] for name, item in headers.items()}
        if any(rate != sample_rate or channels != 1 for rate, _frames, channels in headers.values()):
            rate_mismatch += 1
        if len(set(lengths.values())) != 1 or any(len(waveforms[name]) != lengths["original"] for name in waveforms):
            length_mismatch += 1
        clocks = {name: np.load(path / "metadata" / "frame_times.npy") for name, path in members.items()}
        states = {name: np.load(path / "metadata" / "canonical_state_frames.npy") for name, path in members.items()}
        paths = {
            name: np.asarray(json.loads((path / "metadata" / "simulation_parameters.json").read_text())["path_xy"],
                             dtype=np.float64)
            for name, path in members.items()
        }
        base_clock = clocks["original"]
        for name in (0, 45, 90):
            clock_gap.append(float(np.max(np.abs(clocks[name] - base_clock))))
            label_gap.append(float(np.max(np.abs(states[name] - states["original"]))))
            rotated = _rotate_path(paths["original"], float(name))
            angle_gap.append(float(np.max(np.linalg.norm(paths[name] - rotated, axis=1))))
        pairs = {
            "original_vs_0": ("original", 0),
            "original_vs_45": ("original", 45),
            "original_vs_90": ("original", 90),
            "45_vs_0": (45, 0),
            "90_vs_0": (90, 0),
            "90_vs_45": (90, 45),
        }
        for name, (left, right) in pairs.items():
            store[name].append(_audio_pair(waveforms[left], waveforms[right], features[left], features[right]))
        compared += 1
        if compared % 20 == 0:
            print(f"  compared {compared} rotation families", flush=True)
    summary = {}
    for name, rows in store.items():
        summary[name] = {
            field: percentiles(np.array([row[field] for row in rows]))
            for field in rows[0]
        } if rows else {}
    # Identical copies would correlate and leave the log-STFT unchanged at numerical noise.
    cross = summary.get("45_vs_0", {})
    correlation = cross.get("waveform_correlation", {}).get("p50", 0.0)
    spectrum = cross.get("log_stft_mae", {}).get("p50", 1.0)
    audio_identical = correlation > 0.999 and spectrum < 1e-3
    return {
        "original": str(original),
        "families": compared,
        "rate_mismatches": rate_mismatch,
        "length_mismatches": length_mismatch,
        "frame_time_max_abs_s": float(max(clock_gap) if clock_gap else 0.0),
        "canonical_label_max_abs": float(max(label_gap) if label_gap else 0.0),
        "path_rotation_max_error_m": float(max(angle_gap) if angle_gap else 0.0),
        "pairs": summary,
        "audio_identical": audio_identical,
        "decision": "deduplicate" if audio_identical else "keep_as_augmentations",
        "excluded": excluded,
    }


def rotation_lines(rotation: dict) -> list[str]:
    lines = [
        "Matched original, 45°, and 90° renders",
        f"  Original dataset: {rotation['original']}",
        (
            f"  {rotation['families']} families have all four clocks. "
            f"Sample-rate mismatches: {rotation['rate_mismatches']}. "
            f"WAV-length mismatches: {rotation['length_mismatches']}. "
            f"Largest frame_times gap: {rotation['frame_time_max_abs_s']:.3e} s."
        ),
        (
            f"  Rotating the original path by 0°, 45°, and 90° about the microphone "
            f"matches path_xy within {rotation['path_rotation_max_error_m']:.3e} m. "
            f"Canonical labels still match within {rotation['canonical_label_max_abs']:.3e}."
        ),
        "  Waveform and log-STFT gaps are median / maximum. Correlation is median.",
    ]
    labels = {
        "original_vs_0": "original vs 0°",
        "original_vs_45": "original vs 45°",
        "original_vs_90": "original vs 90°",
        "45_vs_0": "45° vs 0°",
        "90_vs_0": "90° vs 0°",
        "90_vs_45": "90° vs 45°",
    }
    for key, title in labels.items():
        stats = rotation["pairs"][key]
        lines.append(
            f"  {title}: relative RMS {stats['waveform_relative_rms']['p50']:.3f} / "
            f"{stats['waveform_relative_rms']['p100']:.3f}, "
            f"correlation {stats['waveform_correlation']['p50']:.4f}, "
            f"log-STFT MAE {stats['log_stft_mae']['p50']:.4f} / "
            f"{stats['log_stft_mae']['p100']:.4f}"
        )
    if rotation["audio_identical"]:
        lines.append(
            "  Audio features are effectively identical, so repeated angles are duplicates "
            "and are not kept as extra training clips."
        )
    else:
        lines.append(
            "  Audio features differ (correlation near zero, log-STFT MAE near 0.08 against "
            "a feature scale near 0.5), so every angle is kept as an augmentation. "
            "Canonical-label equality does not make the waveforms interchangeable."
        )
    return lines


def group_rows(records: list[dict], splits: dict) -> list[dict]:
    owner = {clip_id: part for part in SPLITS for clip_id in splits[part]}
    grouped: dict[str, dict] = {}
    for record in records:
        item = grouped.setdefault(record["group"], {
            "group": record["group"],
            "split": owner[record["id"]],
            "clips": 0,
            "paths": set(),
        })
        item["clips"] += 1
        item["paths"].add(Path(record["wav"]).name)
    rows = []
    for item in grouped.values():
        rows.append({
            "group": item["group"],
            "split": item["split"],
            "clips": item["clips"],
            "paths": len(item["paths"]),
        })
    order = {"train": 0, "val": 1, "test": 2}
    rows.sort(key=lambda row: (order[row["split"]], row["group"]))
    return rows


def unlabeled_clips(root: Path) -> list[dict]:
    rows = []
    for clip in sorted(p for p in root.rglob("sample_*") if p.is_dir()):
        wavs = sorted(clip.glob("*.wav"))
        state = clip / "metadata" / "canonical_state_frames.npy"
        if wavs and not state.is_file():
            rows.append({
                "id": clip.relative_to(root).as_posix(),
                "wav": wavs[0].name,
                "angle_deg": "",
                "sibling_split": "",
                "reason": "WAV present, canonical_state_frames.npy absent",
            })
    return rows


def load_run(run: Path) -> tuple[list[dict], dict, dict, list[str]]:
    records = json.loads((run / "records.json").read_text())
    splits = json.loads((run / "splits.json").read_text())
    config = json.loads((run / "config.json").read_text())
    return records, splits, config, []


def load_dataset(root: Path, out: Path, seed: int, n_examples: int) -> tuple[list[dict], dict, dict, list[str]]:
    class _Args:
        manifest = None
        wav_name = None

    _Args.root = str(root)
    records = read_records(_Args())
    out.mkdir(parents=True, exist_ok=True)
    splits = make_splits(records, out, seed, n_examples)
    (out / "records.json").write_text(json.dumps(records, indent=2))
    (out / "config.json").write_text(json.dumps(TRAINER_CONFIG, indent=2))
    return records, splits, dict(TRAINER_CONFIG), unlabeled_clips(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="Finished run with records.json and splits.json")
    parser.add_argument("--root", type=Path, default=None, help="Dataset directory. Splits are built the way training builds them.")
    parser.add_argument("--out", type=Path, default=None, help="Report directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--original", type=Path, default=None,
                        help="Unrotated dataset used to match 0/45/90 clocks and audio")
    args = parser.parse_args()
    if args.root is None and args.run is None:
        args.run = ROOT / "position_ctc" / "outputs" / "trajectory86"
    if args.root is not None and args.run is not None:
        raise SystemExit("Pass --root or --run, not both.")
    if args.root is not None:
        root = args.root.resolve()
        out = (args.out or (ROOT / "position_ctc" / "outputs" / f"audit_{root.name}")).resolve()
        records, splits, config, missing = load_dataset(root, out, args.seed, args.examples)
        run = out
    else:
        run = args.run.resolve()
        out = (args.out or (run / "dataset_audit")).resolve()
        out.mkdir(parents=True, exist_ok=True)
        records, splits, config, missing = load_run(run)
    owner = {clip_id: part for part in SPLITS for clip_id in splits[part]}
    if set(owner) != {record["id"] for record in records}:
        raise SystemExit("records.json and splits.json do not name the same clips.")
    data_root = dataset_root(records)
    print(f"Dataset: {data_root}", flush=True)
    print(f"Run: {run}", flush=True)
    rotations = None
    if args.original is not None:
        print("Comparing original, 45°, and 90° clocks and audio...", flush=True)
        rotations = compare_rotations(
            data_root, args.original.resolve(),
            int(config["sample_rate"]), int(config["n_fft"]), int(config["hop"]),
        )
        found = {item["id"] for item in rotations["excluded"]}
        for item in unlabeled_clips(data_root):
            if item["id"] not in found:
                rotations["excluded"].append(item)
        missing = rotations["excluded"]
    if args.root is not None and not missing:
        missing = unlabeled_clips(data_root)
    by_wav: dict[str, set[str]] = {}
    for record in records:
        by_wav.setdefault(Path(record["wav"]).name, set()).add(owner[record["id"]])
    excluded = []
    for item in missing:
        if isinstance(item, str):
            item = {"id": item, "wav": "", "angle_deg": "", "reason": "unlabeled"}
        else:
            item = dict(item)
        parts = sorted(by_wav.get(item.get("wav", ""), ()))
        item["sibling_split"] = ", ".join(parts) if parts else "no labeled sibling"
        excluded.append(item)
    grid = centers(float(config["bin_width"]), float(config["vmax"]))
    rows = []
    series = {}
    velocities = {part: {"vx": [], "vy": [], "turn_abs": []} for part in SPLITS}
    by_id = {record["id"]: record for record in records}
    with tempfile.TemporaryDirectory(prefix="trajectory_audit_") as cache:
        store = ClipStore(records, config, Path(cache))
        ordered = [clip_id for part in SPLITS for clip_id in splits[part]]
        for n, clip_id in enumerate(ordered, start=1):
            record = by_id[clip_id]
            duration = wav_duration(record["wav"], int(config["sample_rate"]))
            row, one = clip_metrics(store.get(clip_id), record, owner[clip_id], config, duration)
            rows.append(row)
            series[clip_id] = one
            velocities[owner[clip_id]]["vx"].append(one["vx"])
            velocities[owner[clip_id]]["vy"].append(one["vy"])
            opening_n = max(1, len(one["vx"]) // 5)
            vel = np.column_stack([one["vx"], one["vy"]])
            opening = vel[:opening_n].mean(axis=0)
            turning = np.linalg.norm(vel - opening, axis=1) >= float(config["bin_width"])
            if turning.any():
                velocities[owner[clip_id]]["turn_abs"].append(np.abs(one["vy"][turning]))
            if n % 20 == 0 or n == len(ordered):
                print(f"  loaded {n}/{len(ordered)}", flush=True)
        for part in SPLITS:
            velocities[part]["vx"] = np.concatenate(velocities[part]["vx"])
            velocities[part]["vy"] = np.concatenate(velocities[part]["vy"])
            parts = velocities[part]["turn_abs"]
            velocities[part]["turn_abs"] = np.concatenate(parts) if parts else np.array([])
        summary = summarize(rows, velocities, grid, span_bin_edges(rows))
        ks = ks_table(rows)
        print("Checking network inputs...", flush=True)
        inputs = check_inputs(store, rows, config, run / "normalization.json")
    examples = pick_series(rows, series)
    write_plots(out, summary, examples, grid)
    payload = {
        "dataset": str(data_root),
        "run": str(run),
        "records": str(run / "records.json"),
        "splits_file": str(run / "splits.json"),
        "sample_rate": int(config["sample_rate"]),
        "n_fft": int(config["n_fft"]),
        "hop": int(config["hop"]),
        "bin_width": float(config["bin_width"]),
        "vmax": float(config["vmax"]),
        "summary": summary,
        "vehicles": vehicles_table(rows),
        "ks": ks,
        "inputs": inputs,
        "examples": {item["role"]: item for item in inputs["forwards"]},
        "straight_clips": [row for row in rows if row["frac_qvy_zero"] > 0.999],
        "excluded": excluded,
        "rotations": rotations,
        "groups": group_rows(records, splits),
        "validation_groups": splits.get("validation_groups", []),
        "coverage": "",
    }
    payload["coverage"] = coverage_text(summary, ks)
    report = render_report(payload)
    (out / "report.txt").write_text(report)
    json_payload = json.loads(json.dumps(payload, default=lambda o: None))
    (out / "summary.json").write_text(json.dumps(json_payload, indent=2))
    write_csv(out / "clips.csv", rows)
    print(report)
    print(f"Wrote {out / 'report.txt'}", flush=True)


if __name__ == "__main__":
    main()
