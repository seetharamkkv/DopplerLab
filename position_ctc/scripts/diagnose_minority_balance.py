"""One short-tail clip, two losses, the same optimization budget.

Ordinary framewise cross-entropy is compared with per-axis dominant-bin
balancing. Architecture, initialization, optimizer, and the number of updates
match. CTC, the trajectory loss, and Viterbi stay out of this run.

A third run, balanced cross-entropy without the stride-2 pools, starts only
if balancing does not reproduce the non-dominant tail.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.bins import CENTERS_MPS, velocity_bin_index  # noqa: E402
from position_ctc.data import discover_clips, load_arrays, split_by_vehicle, standardize_mag  # noqa: E402
from position_ctc.features import DATA_HOP, DATA_SR, features_from_wav  # noqa: E402
from position_ctc.motion_loss import dominant_balanced_nll  # noqa: E402
from position_ctc.motion_model import MotionNet  # noqa: E402
from position_ctc.paths import resolve_audio_clips  # noqa: E402

OUT = ROOT / "position_ctc" / "outputs" / "motion" / "minority"
DT = DATA_HOP / float(DATA_SR)
DT_22050 = DATA_HOP / 22050.0
EPOCHS = 300
LR = 3e-3
SEED = 0
LOG_EPOCHS = (1, 10, 20, 40, 80, 120, 160, 200, 250, 300)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _wav_meta(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getframerate(), handle.getnframes()


def _clock_audit(clips: list) -> dict:
    """Compare label-frame counts with the 44.1 kHz hop and the 22.05 kHz hop."""
    gaps_86 = []
    gaps_43 = []
    rates = []
    for clip in clips:
        sr, n_samples = _wav_meta(clip.wav_path)
        rates.append(sr)
        state_n = int(np.load(clip.sample_dir / "metadata" / "canonical_state_frames.npy").shape[0])
        frames_86 = 1 + n_samples // DATA_HOP
        frames_43 = 1 + (n_samples // 2) // DATA_HOP
        gaps_86.append(abs(state_n - frames_86))
        gaps_43.append(abs(state_n - frames_43))
    rates = np.asarray(rates)
    return {
        "wav_sample_rates": [int(v) for v in np.unique(rates)],
        "clips": len(clips),
        "median_abs_frame_gap_512_over_44100": float(np.median(gaps_86)),
        "max_abs_frame_gap_512_over_44100": int(np.max(gaps_86)),
        "median_abs_frame_gap_512_over_22050": float(np.median(gaps_43)),
        "feature_dt_s": DT,
        "alternate_export_dt_s": DT_22050,
        "feature_rate_hz": float(DATA_SR / DATA_HOP),
        "alternate_export_hz": float(22050 / DATA_HOP),
    }


def _clock_on_clip(clip) -> dict:
    sr, n_samples = _wav_meta(clip.wav_path)
    feat = features_from_wav(clip.wav_path)
    state = np.load(clip.sample_dir / "metadata" / "canonical_state_frames.npy")
    loaded, loaded_state = load_arrays(clip)
    frames_86 = 1 + n_samples // DATA_HOP
    return {
        "clip_id": clip.clip_id,
        "wav_sr": sr,
        "n_samples": n_samples,
        "audio_duration_s": n_samples / float(sr),
        "feature_frames": int(feat.shape[1]),
        "frames_from_512_over_44100": frames_86,
        "feature_clock_matches_formula": int(feat.shape[1]) == frames_86 and sr == DATA_SR,
        "raw_state_frames": int(state.shape[0]),
        "trimmed_frames": int(loaded_state.shape[0]),
        "trimmed_feature_frames": int(loaded.shape[1]),
        "trimmed_lengths_match": int(loaded.shape[1]) == int(loaded_state.shape[0]),
        "raw_state_duration_at_feature_dt_s": float(state.shape[0] * DT),
        "raw_state_duration_at_22050_dt_s": float(state.shape[0] * DT_22050),
        "audio_minus_feature_grid_s": float(n_samples / float(sr) - frames_86 * DT),
    }


def _minority_run(bins: np.ndarray, dominant: int) -> tuple[float, float, float]:
    minority = bins != dominant
    count = int(minority.sum())
    if count == 0:
        return 0.0, 0.0, 0.5
    longest = 0
    current = 0
    end = 0
    for i, flag in enumerate(minority):
        if flag:
            current += 1
            if current > longest:
                longest = current
                end = i
        else:
            current = 0
    center = float(np.flatnonzero(minority).mean() / max(bins.size - 1, 1))
    return count / bins.size, longest / count, center


def _select_clip(rows: list[tuple]) -> dict:
    """A short contiguous non-dominant tail, not a clip whose turn is most frames."""
    ranked = []
    for clip_id, state in rows:
        for axis, column in (("vx", 1), ("vy", 3)):
            values = state[:, column]
            bins = velocity_bin_index(values)
            dominant = int(np.bincount(bins).argmax())
            fraction, concentration, center = _minority_run(bins, dominant)
            if not (0.05 <= fraction <= 0.20):
                continue
            if concentration < 0.60:
                continue
            if not (center <= 0.25 or center >= 0.75):
                continue
            minority = bins != dominant
            if int(minority.sum()) < 30:
                continue
            span = float(np.ptp(values[minority]))
            shift = float(abs(values[minority].mean() - CENTERS_MPS[dominant]))
            if span < 4.0 and shift < 4.0:
                continue
            ranked.append(
                {
                    "clip_id": clip_id,
                    "axis": axis,
                    "dominant_mps": float(CENTERS_MPS[dominant]),
                    "minority_fraction": fraction,
                    "run_concentration": concentration,
                    "run_center": center,
                    "minority_span_mps": span,
                    "minority_shift_mps": shift,
                    "frames": int(values.size),
                }
            )
    if not ranked:
        raise SystemExit("no short-tail clip matched the selection rule")
    ranked.sort(key=lambda item: (abs(item["minority_fraction"] - 0.10), -item["minority_shift_mps"]))
    return ranked[0], ranked[:8]


def _new_model(temporal_pool: bool) -> MotionNet:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return MotionNet(dropout=0.0, temporal_pool=temporal_pool)


def _group_metrics(probs: np.ndarray, truth: np.ndarray) -> dict:
    true_bins = velocity_bin_index(truth)
    pred_bins = np.argmax(probs, axis=1)
    dominant = int(np.bincount(true_bins).argmax())
    dominant_frames = true_bins == dominant
    minority = ~dominant_frames
    pred_mps = CENTERS_MPS[pred_bins]

    def _acc(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(pred_bins[mask] == true_bins[mask]))

    def _mae(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.abs(pred_mps[mask] - truth[mask])))

    if np.any(minority):
        correct_p = float(probs[np.flatnonzero(minority), true_bins[minority]].mean())
    else:
        correct_p = float("nan")
    hold = np.zeros(true_bins.size, dtype=bool)
    hold[1:] = (true_bins[1:] == true_bins[:-1]) & dominant_frames[1:] & dominant_frames[:-1]
    flips = np.zeros(true_bins.size, dtype=bool)
    flips[1:] = pred_bins[1:] != pred_bins[:-1]
    false_changes = int(np.sum(flips & hold))
    hold_count = int(np.sum(hold))
    true_changes = np.flatnonzero(true_bins[1:] != true_bins[:-1]) + 1
    pred_changes = np.flatnonzero(pred_bins[1:] != pred_bins[:-1]) + 1
    if true_changes.size and pred_changes.size:
        match = [float(np.min(np.abs(pred_changes - t)) * DT) for t in true_changes]
        median_match = float(np.median(match))
    else:
        median_match = float("nan")

    def _first(mask_bins: np.ndarray) -> float | None:
        hit = np.flatnonzero(mask_bins != dominant)
        if hit.size == 0:
            return None
        return float(hit[0] * DT)

    return {
        "dominant_mps": float(CENTERS_MPS[dominant]),
        "dominant_fraction": float(np.mean(dominant_frames)),
        "dominant_accuracy": _acc(dominant_frames),
        "minority_accuracy": _acc(minority),
        "dominant_mae_mps": _mae(dominant_frames),
        "minority_mae_mps": _mae(minority),
        "minority_correct_probability": correct_p,
        "false_changes_on_dominant_runs": false_changes,
        "dominant_hold_frames": hold_count,
        "false_change_rate": float(false_changes / max(hold_count, 1)),
        "true_transition_s": [float(t * DT) for t in true_changes[:12]],
        "pred_transition_s": [float(t * DT) for t in pred_changes[:12]],
        "true_transition_count": int(true_changes.size),
        "pred_transition_count": int(pred_changes.size),
        "median_transition_time_error_s": median_match,
        "true_first_leave_dominant_s": _first(true_bins),
        "pred_first_leave_dominant_s": _first(pred_bins),
        "pred_mps": pred_mps,
        "minority": minority,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key not in ("pred_mps", "minority")}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _train_one(
    feat: np.ndarray,
    state: np.ndarray,
    *,
    balanced: bool,
    temporal_pool: bool,
    device: torch.device,
) -> dict:
    model = _new_model(temporal_pool).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    frames = feat.shape[1]
    batch = torch.from_numpy(np.ascontiguousarray(feat)).unsqueeze(0).to(device)
    lengths = torch.tensor([frames], dtype=torch.long, device=device)
    origin = torch.tensor([[float(state[0, 0]), float(state[0, 2])]], dtype=torch.float32, device=device)
    bins_x = torch.from_numpy(velocity_bin_index(state[:, 1])).view(1, -1).to(device)
    bins_y = torch.from_numpy(velocity_bin_index(state[:, 3])).view(1, -1).to(device)
    valid = torch.ones(1, frames, dtype=torch.bool, device=device)
    history = []
    curves = []
    probe = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(batch, lengths, origin)
        nll_x = F.cross_entropy(out["vel_x"][0], bins_x[0], reduction="none").view(1, -1)
        nll_y = F.cross_entropy(out["vel_y"][0], bins_y[0], reduction="none").view(1, -1)
        if epoch == 1:
            dominant_y = int(np.bincount(velocity_bin_index(state[:, 3])).argmax())
            dominant_x = int(np.bincount(velocity_bin_index(state[:, 1])).argmax())
            terms = []
            counts = {}
            for name, nll, target, dominant in (
                ("vx", nll_x, bins_x, dominant_x),
                ("vy", nll_y, bins_y, dominant_y),
            ):
                minority = valid & (target != dominant)
                counts[name] = int(minority.sum())
                if counts[name] > 0:
                    terms.append(nll[minority].mean())
            if not terms:
                raise RuntimeError("clip has no non-dominant frames on either axis")
            torch.stack(terms).sum().backward()
            probe = {
                "minority_frames_in_loss_vx": counts["vx"],
                "minority_frames_in_loss_vy": counts["vy"],
                "label_minority_frames_vx": int(np.sum(velocity_bin_index(state[:, 1]) != dominant_x)),
                "label_minority_frames_vy": int(np.sum(velocity_bin_index(state[:, 3]) != dominant_y)),
                "vel_x_grad": float(model.vel_x.weight.grad.detach().norm()) if model.vel_x.weight.grad is not None else 0.0,
                "vel_y_grad": float(model.vel_y.weight.grad.detach().norm()) if model.vel_y.weight.grad is not None else 0.0,
                "conv_grad": float(model.conv[0].weight.grad.detach().norm()),
                "origin_grad": float(model.origin.weight.grad.detach().norm()),
                "encoder_frames": int(out["enc_lengths"][0]),
                "audio_frames": frames,
            }
            opt.zero_grad(set_to_none=True)
            out = model(batch, lengths, origin)
            nll_x = F.cross_entropy(out["vel_x"][0], bins_x[0], reduction="none").view(1, -1)
            nll_y = F.cross_entropy(out["vel_y"][0], bins_y[0], reduction="none").view(1, -1)
        ordinary = 0.5 * (nll_x.mean() + nll_y.mean())
        balanced_loss = 0.5 * (
            dominant_balanced_nll(nll_x, bins_x, valid) + dominant_balanced_nll(nll_y, bins_y, valid)
        )
        loss = balanced_loss if balanced else ordinary
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if epoch in LOG_EPOCHS or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                seen = model(batch, lengths, origin)
                px = seen["vel_x"][0].softmax(-1).cpu().numpy()
                py = seen["vel_y"][0].softmax(-1).cpu().numpy()
            mx = _group_metrics(px, state[:, 1])
            my = _group_metrics(py, state[:, 3])
            row = {
                "epoch": epoch,
                "ordinary_ce": float(ordinary.detach()),
                "balanced_ce": float(balanced_loss.detach()),
                "train_loss": float(loss.detach()),
                "vx": _jsonable(mx),
                "vy": _jsonable(my),
            }
            history.append(row)
            curves.append({"epoch": epoch, "vx": mx["pred_mps"], "vy": my["pred_mps"]})
            print(
                f"{'balanced' if balanced else 'ordinary':8s} pool {int(temporal_pool)}  "
                f"epoch {epoch:03d}  ordinary {row['ordinary_ce']:.3f}  balanced {row['balanced_ce']:.3f}  "
                f"vy minority acc {my['minority_accuracy']:.3f}  p {my['minority_correct_probability']:.3f}  "
                f"mae {my['minority_mae_mps']:.2f}  "
                f"vx minority acc {mx['minority_accuracy']:.3f}  p {mx['minority_correct_probability']:.3f}",
                flush=True,
            )
            model.train()
    return {
        "balanced": balanced,
        "temporal_pool": temporal_pool,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "updates": EPOCHS,
        "minority_gradient": probe,
        "history": history,
        "curves": curves,
    }


def _recovered(run: dict, axis: str) -> bool:
    final = run["history"][-1][axis]
    accuracy = final["minority_accuracy"]
    probability = final["minority_correct_probability"]
    return (
        accuracy is not None
        and probability is not None
        and accuracy >= 0.8
        and probability >= 0.5
    )


def _curve_figure(runs: list[dict], path: Path, axis: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    for run in runs:
        name = "balanced" if run["balanced"] else "ordinary"
        if not run["temporal_pool"]:
            name += ", full resolution"
        epochs = [row["epoch"] for row in run["history"]]
        axes[0].plot(epochs, [row["ordinary_ce"] for row in run["history"]], lw=1.4, label=name)
        axes[1].plot(epochs, [row[axis]["minority_accuracy"] for row in run["history"]], lw=1.4, label=name)
        axes[2].plot(
            epochs,
            [row[axis]["minority_correct_probability"] for row in run["history"]],
            lw=1.4,
            label=name,
        )
    axes[0].set_title("ordinary CE")
    axes[1].set_title(f"{axis} non-dominant accuracy")
    axes[2].set_title(f"{axis} P(correct bin | non-dominant)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _prediction_figure(state: np.ndarray, runs: list[dict], path: Path) -> None:
    time = np.arange(state.shape[0]) * DT
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)
    axes[0].plot(time, state[:, 1], color="black", lw=1.4, label="ground truth")
    axes[1].plot(time, state[:, 3], color="black", lw=1.4, label="ground truth")
    for run in runs:
        name = "balanced" if run["balanced"] else "ordinary"
        if not run["temporal_pool"]:
            name += ", full resolution"
        by_epoch = {item["epoch"]: item for item in run["curves"]}
        for epoch, style in ((80, ":"), (300, "-")):
            if epoch not in by_epoch:
                continue
            axes[0].plot(time, by_epoch[epoch]["vx"], lw=1.1, ls=style, label=f"{name} epoch {epoch}")
            axes[1].plot(time, by_epoch[epoch]["vy"], lw=1.1, ls=style, label=f"{name} epoch {epoch}")
    axes[0].set_ylabel(r"$v_x$ (m/s)")
    axes[1].set_ylabel(r"$v_y$ (m/s)")
    axes[1].set_xlabel("t (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = _device()
    clips = discover_clips(resolve_audio_clips())
    splits = split_by_vehicle(clips, seed=0)
    train = [clips[i] for i in splits["train"]]
    print(f"device {device}  feature dt {DT:.6f} s ({DATA_SR / DATA_HOP:.2f} Hz)", flush=True)
    clock = _clock_audit(train)
    print(
        f"clock  wav rates {clock['wav_sample_rates']}  "
        f"median |state - 44100 grid| {clock['median_abs_frame_gap_512_over_44100']:.1f} frames  "
        f"median |state - 22050 grid| {clock['median_abs_frame_gap_512_over_22050']:.1f} frames",
        flush=True,
    )
    rows = []
    for clip in train:
        state = np.load(clip.sample_dir / "metadata" / "canonical_state_frames.npy")
        speed = np.hypot(state[:, 1], state[:, 3])
        moving = np.flatnonzero(speed > 0.5)
        if moving.size == 0:
            continue
        rows.append((clip.clip_id, state[int(moving[0]) : int(moving[-1]) + 1]))
    chosen, nearby = _select_clip(rows)
    clip = next(item for item in train if item.clip_id == chosen["clip_id"])
    clip_clock = _clock_on_clip(clip)
    feat, state = load_arrays(clip)
    feat = standardize_mag(feat)
    n = min(feat.shape[1], state.shape[0])
    feat, state = feat[:, :n], state[:n]
    print(
        f"clip {clip.clip_id}  tail axis {chosen['axis']}  "
        f"dominant {chosen['dominant_mps']:+.0f} m/s  "
        f"minority {chosen['minority_fraction']:.1%}  "
        f"shift {chosen['minority_shift_mps']:.1f} m/s  frames {n}",
        flush=True,
    )
    print(
        f"clip clock  feature frames {clip_clock['feature_frames']} "
        f"formula {clip_clock['frames_from_512_over_44100']}  "
        f"match {clip_clock['feature_clock_matches_formula']}  "
        f"trimmed state/feature {clip_clock['trimmed_frames']}/{clip_clock['trimmed_feature_frames']}",
        flush=True,
    )
    ordinary = _train_one(feat, state, balanced=False, temporal_pool=True, device=device)
    balanced = _train_one(feat, state, balanced=True, temporal_pool=True, device=device)
    runs = [ordinary, balanced]
    axis = chosen["axis"]
    if not _recovered(balanced, axis):
        print(
            f"balanced CE did not recover the {axis} tail; repeating without temporal pooling",
            flush=True,
        )
        runs.append(_train_one(feat, state, balanced=True, temporal_pool=False, device=device))
    _curve_figure(runs, OUT / "learning_curves.png", axis)
    _prediction_figure(state, runs, OUT / "predictions.png")
    report = {
        "clock_train": clock,
        "clock_clip": clip_clock,
        "selection": chosen,
        "selection_nearby": nearby,
        "loss_note": (
            "Both runs use 0.5 * (L_x + L_y) so the learning rate matches the earlier "
            "300-epoch memorization. Inside L_a, ordinary CE is the frame mean and "
            "balanced CE gives the dominant bin and the other bins half each."
        ),
        "runs": [{key: value for key, value in run.items() if key != "curves"} for run in runs],
    }
    (OUT / "minority.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
