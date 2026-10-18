"""Held-out softmax distributions for each frame-balance setting.

The three checkpoints differ only in the framewise cross-entropy weight:
ordinary mean, dominant-bin balance, and frequency balance. This script
averages the framewise velocity softmax on the vehicle-holdout test clips
and writes that distribution next to the label histogram.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.bins import CENTERS_MPS, velocity_bin_index  # noqa: E402
from position_ctc.data import discover_clips, load_arrays, split_by_vehicle, standardize_mag  # noqa: E402
from position_ctc.motion_model import MotionNet  # noqa: E402
from position_ctc.paths import resolve_audio_clips  # noqa: E402

MOTION = ROOT / "position_ctc" / "outputs" / "motion"
OUT = MOTION / "pred_distributions.png"
RUNS = (
    ("mean", "epoch 35, ordinary CE", MOTION / "motion.pt", "#1f77b4"),
    ("dominant", "epoch 19, dominant balance", MOTION / "balance_dominant" / "motion.pt", "#d62728"),
    ("frequency", "epoch 18, frequency balance", MOTION / "balance_frequency" / "motion.pt", "#2ca02c"),
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load(path: Path, device: torch.device) -> MotionNet:
    model = MotionNet(dropout=0.0).to(device)
    saved = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


@torch.no_grad()
def _softmax(model: MotionNet, feat: np.ndarray, origin: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    frames = int(feat.shape[-1])
    batch = torch.from_numpy(np.ascontiguousarray(feat)).unsqueeze(0).to(device)
    lengths = torch.tensor([frames], dtype=torch.long, device=device)
    start = torch.tensor(origin, dtype=torch.float32, device=device).view(1, 2)
    out = model(batch, lengths, start)
    px = out["vel_x"][0, :frames].softmax(-1).cpu().numpy()
    py = out["vel_y"][0, :frames].softmax(-1).cpu().numpy()
    return px, py


def _window(freq: np.ndarray, curves: list[np.ndarray], pad: int = 1) -> slice:
    occupied = freq > 0.001
    for curve in curves:
        occupied = occupied | (curve > 0.005)
    idx = np.flatnonzero(occupied)
    if idx.size == 0:
        return slice(0, freq.size)
    lo = max(int(idx[0]) - pad, 0)
    hi = min(int(idx[-1]) + pad + 1, freq.size)
    return slice(lo, hi)


def main() -> None:
    clips = discover_clips(resolve_audio_clips())
    test_idx = split_by_vehicle(clips, seed=0)["test"]
    device = _device()
    models = {name: _load(path, device) for name, _label, path, _color in RUNS}
    print(f"device {device}  test clips {len(test_idx)}", flush=True)

    label = {"vx": np.zeros(CENTERS_MPS.size), "vy": np.zeros(CENTERS_MPS.size)}
    mass = {name: {"vx": np.zeros(CENTERS_MPS.size), "vy": np.zeros(CENTERS_MPS.size)} for name, *_ in RUNS}
    confidence = {name: {"vx": [], "vy": []} for name, *_ in RUNS}
    frames = 0
    bin_frames = {"vx": np.zeros(CENTERS_MPS.size, dtype=np.int64), "vy": np.zeros(CENTERS_MPS.size, dtype=np.int64)}
    correct_mass = {
        name: {"vx": np.zeros(CENTERS_MPS.size), "vy": np.zeros(CENTERS_MPS.size)} for name, *_ in RUNS
    }

    for n_done, index in enumerate(test_idx, start=1):
        feat, state = load_arrays(clips[index])
        feat = standardize_mag(feat)
        n = min(int(feat.shape[1]), int(state.shape[0]))
        feat = feat[:, :n]
        state = state[:n]
        origin = state[0, [0, 2]]
        bx = velocity_bin_index(state[:, 1])
        by = velocity_bin_index(state[:, 3])
        label["vx"] += np.bincount(bx, minlength=CENTERS_MPS.size)
        label["vy"] += np.bincount(by, minlength=CENTERS_MPS.size)
        np.add.at(bin_frames["vx"], bx, 1)
        np.add.at(bin_frames["vy"], by, 1)
        frames += n
        for name, model in models.items():
            px, py = _softmax(model, feat, origin, device)
            mass[name]["vx"] += px.sum(axis=0)
            mass[name]["vy"] += py.sum(axis=0)
            confidence[name]["vx"].append(px.max(axis=1))
            confidence[name]["vy"].append(py.max(axis=1))
            np.add.at(correct_mass[name]["vx"], bx, px[np.arange(n), bx])
            np.add.at(correct_mass[name]["vy"], by, py[np.arange(n), by])
        if n_done % 25 == 0 or n_done == len(test_idx):
            print(f"scored {n_done}/{len(test_idx)}", flush=True)

    freq = {axis: label[axis] / frames for axis in ("vx", "vy")}
    mean_p = {
        name: {axis: mass[name][axis] / frames for axis in ("vx", "vy")} for name, *_ in RUNS
    }
    dominant_bin = {axis: int(np.argmax(label[axis])) for axis in ("vx", "vy")}

    summary = {
        "split": "vehicle holdout seed 0, test vehicles Kia Sportage and Renault Captur",
        "clips": len(test_idx),
        "frames": frames,
        "bin_step_mps": 2.0,
        "alphabet_mps": [float(CENTERS_MPS[0]), float(CENTERS_MPS[-1])],
        "dominant_bin_mps": {axis: float(CENTERS_MPS[dominant_bin[axis]]) for axis in ("vx", "vy")},
        "runs": {},
    }
    for name, label_text, path, _color in RUNS:
        summary["runs"][name] = {
            "checkpoint": str(path.relative_to(ROOT)),
            "label": label_text,
            "vx": _correct_stats_from_sums(correct_mass[name]["vx"], bin_frames["vx"], mean_p[name]["vx"], dominant_bin["vx"]),
            "vy": _correct_stats_from_sums(correct_mass[name]["vy"], bin_frames["vy"], mean_p[name]["vy"], dominant_bin["vy"]),
        }
        summary["runs"][name]["vx"]["confidence"] = _confidence_summary(confidence[name]["vx"])
        summary["runs"][name]["vy"]["confidence"] = _confidence_summary(confidence[name]["vy"])

    _figure(freq, mean_p, confidence, dominant_bin, summary)
    report_path = OUT.with_suffix(".json")
    report_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")


def _correct_stats_from_sums(
    correct_sum: np.ndarray,
    counts: np.ndarray,
    mean_probs: np.ndarray,
    dominant: int,
) -> dict[str, float]:
    total = int(counts.sum())
    minority = np.ones(counts.size, dtype=bool)
    minority[dominant] = False
    dom_n = int(counts[dominant])
    min_n = int(counts[minority].sum())
    entropy = float((-(mean_probs * np.log(np.clip(mean_probs, 1e-12, 1.0))).sum()))
    return {
        "mean_prob_on_true_bin": float(correct_sum.sum() / total),
        "dominant_prob_on_true_bin": float(correct_sum[dominant] / dom_n) if dom_n else float("nan"),
        "minority_prob_on_true_bin": float(correct_sum[minority].sum() / min_n) if min_n else float("nan"),
        "mean_distribution_entropy_nats": entropy,
        "dominant_bin_mps": float(CENTERS_MPS[dominant]),
        "dominant_frame_fraction": float(dom_n / total),
    }


def _confidence_summary(chunks: list[np.ndarray]) -> dict[str, float]:
    values = np.concatenate(chunks)
    return {
        "mean_max_probability": float(values.mean()),
        "median_max_probability": float(np.median(values)),
        "fraction_above_0.5": float(np.mean(values >= 0.5)),
        "fraction_above_0.9": float(np.mean(values >= 0.9)),
    }


def _figure(
    freq: dict[str, np.ndarray],
    mean_p: dict[str, dict[str, np.ndarray]],
    confidence: dict[str, dict[str, list[np.ndarray]]],
    dominant_bin: dict[str, int],
    summary: dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), gridspec_kw={"height_ratios": (1.25, 1.0)})
    curves = [mean_p[name] for name, *_ in RUNS]
    for col, axis in enumerate(("vx", "vy")):
        ax = axes[0, col]
        window = _window(freq[axis], [item[axis] for item in curves])
        centers = CENTERS_MPS[window]
        ax.bar(
            centers,
            freq[axis][window],
            width=1.55,
            color="0.75",
            label="label frequency",
            zorder=1,
        )
        for name, label_text, _path, color in RUNS:
            ax.plot(
                centers,
                mean_p[name][axis][window],
                color=color,
                lw=1.6,
                marker="o",
                ms=3.5,
                label=label_text,
                zorder=2,
            )
        peak = float(CENTERS_MPS[dominant_bin[axis]])
        ax.axvline(peak, color="0.35", lw=0.7, ls=":", zorder=0)
        ax.set_xlabel(f"{axis} bin center (m/s)")
        ax.set_ylabel("probability")
        ax.set_title(f"{axis}  mean softmax on the test set")
        ax.set_ylim(0.0, max(0.75, float(freq[axis][window].max()) * 1.18))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

        conf_ax = axes[1, col]
        bins = np.linspace(0.0, 1.0, 21)
        centers_c = 0.5 * (bins[:-1] + bins[1:])
        width = (bins[1] - bins[0]) * 0.28
        tallest = 0.0
        for offset, (name, label_text, _path, color) in zip((-width, 0.0, width), RUNS):
            values = np.concatenate(confidence[name][axis])
            hist, _ = np.histogram(values, bins=bins, density=False)
            share = hist / values.size
            tallest = max(tallest, float(share.max()))
            conf_ax.bar(
                centers_c + offset,
                share,
                width=width,
                color=color,
                label=label_text,
            )
        conf_ax.set_xlabel(f"max {axis} probability")
        conf_ax.set_ylabel("fraction of frames")
        conf_ax.set_xlim(0.0, 1.0)
        conf_ax.set_ylim(0.0, tallest * 1.18)
        conf_ax.set_title(f"{axis}  confidence of the chosen bin")
        conf_ax.grid(True, axis="y", alpha=0.3)
        conf_ax.legend(frameon=False, fontsize=8)

    clips = summary["clips"]
    frames = summary["frames"]
    fig.suptitle(
        f"Held-out velocity distributions  ·  {clips} test clips, {frames:,} frames",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.text(
        0.5,
        0.005,
        "Vehicle holdout, seed 0 (Kia Sportage, Renault Captur). "
        "Bars are the label histogram. Lines are the mean framewise softmax. "
        "Alphabet is 2 m/s bins from −30 to +30 m/s; empty tails are cropped.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.25",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
