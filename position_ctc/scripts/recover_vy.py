"""Fine-tune epoch 38 so nonzero vy frames share the vy loss.

Two runs start from motion_epoch38.pt and see the same batches for the same
number of updates. One keeps ordinary framewise cross-entropy on both axes.
The other keeps ordinary cross-entropy on vx and splits the vy loss equally
between the zero bin and the other bins inside each clip. The acoustic CNN
stays frozen. CTC and the trajectory loss stay off. Existing checkpoints are
only read.

Run from the DopplerLab root:

    position_ctc/.venv/bin/python -u position_ctc/scripts/recover_vy.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.bins import CENTERS_MPS, velocity_bin_index  # noqa: E402
from position_ctc.data import discover_clips, load_arrays, split_by_vehicle, standardize_mag  # noqa: E402
from position_ctc.features import DATA_HOP, DATA_SR, N_MELS  # noqa: E402
from position_ctc.motion_loss import _bin_index, zero_bin_balanced_nll  # noqa: E402
from position_ctc.motion_model import _upsample_valid  # noqa: E402
from position_ctc.reconstruct import reconstruct  # noqa: E402
from position_ctc.transitions import a_max_for_jump, transition_matrix  # noqa: E402

DATA = Path("/Users/seetharamkkv/Desktop/Seetharam/CMU/projects/doppler/datasets/audio_clips")
MOTION = ROOT / "position_ctc" / "outputs" / "motion"
OUT = MOTION / "recovery"
PARENT = MOTION / "motion_epoch38.pt"
PROTECTED = (
    MOTION / "motion_epoch38.pt",
    MOTION / "motion.pt",
    MOTION / "motion_polar_epoch27.pt",
    MOTION / "motion_corr.pt",
    MOTION / "balance_dominant" / "motion.pt",
    MOTION / "balance_frequency" / "motion.pt",
)

DT = DATA_HOP / float(DATA_SR)
SPEED_OF_SOUND = 343.0
ZERO_BIN = int(velocity_bin_index(np.array([0.0]))[0])
TURN_MPS = 2.0
JUMP_MPS = 6.0
LAMBDA = 1.0
BASE_LR = 1e-3
LR = BASE_LR / 10.0
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0
EPOCHS = 20
BATCH = 8
SEED = 0
PLOT_TEST = ("sample_0000172", "sample_0000498")


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Epoch38Net(nn.Module):
    """The epoch-38 trunk: CNN, BiLSTM, separate vx/vy heads, unused pos/acc.

    No initial-position conditioning. The convolution is detached because this
    recovery keeps it frozen; the LSTM and the velocity heads still train.
    """

    def __init__(self, n_mels: int = N_MELS, n_vel: int = CENTERS_MPS.size, dropout: float = 0.1):
        super().__init__()
        self.time_stride = 4
        self.conv = nn.Sequential(
            nn.Conv1d(3 * n_mels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.LSTM(96, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.vel_x = nn.Linear(256, n_vel)
        self.vel_y = nn.Linear(256, n_vel)
        self.ctc_x = nn.Linear(256, n_vel + 1)
        self.ctc_y = nn.Linear(256, n_vel + 1)
        self.pos = nn.Linear(256, 2)
        self.acc = nn.Linear(256, 2)

    def forward(self, feat: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        lengths = lengths.to(device=feat.device, dtype=torch.long)
        hidden = self.conv(feat).detach().transpose(1, 2)
        enc_len = torch.clamp(lengths // self.time_stride, min=1)
        enc_len = torch.minimum(enc_len, torch.full_like(enc_len, hidden.size(1)))
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden, enc_len.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=hidden.size(1))
        frame = _upsample_valid(self.drop(hidden), enc_len, lengths)
        return {"vel_x": self.vel_x(frame), "vel_y": self.vel_y(frame)}


def _freeze(model: Epoch38Net) -> None:
    for name, param in model.named_parameters():
        train = name.startswith("rnn.") or name.startswith("vel_x.") or name.startswith("vel_y.")
        param.requires_grad = train


def _load_parent(device: torch.device) -> Epoch38Net:
    model = Epoch38Net().to(device)
    saved = torch.load(PARENT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(saved["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"epoch 38 load mismatch missing={missing} unexpected={unexpected}")
    _freeze(model)
    model.eval()
    return model


def _wav_duration(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes(), handle.getframerate()


def _audit(clips) -> dict:
    """Spectrogram clock against labels, and late-turn sound against the wav."""
    gaps = []
    time_err = []
    hot = 0
    hot_inside_wav = 0
    hot_inside_moving = 0
    sr_bad = []
    for clip in clips:
        n_wav, sr = _wav_duration(clip.wav_path)
        if sr != DATA_SR:
            sr_bad.append(clip.clip_id)
        state = np.load(clip.sample_dir / "metadata" / "canonical_state_frames.npy")
        times = np.load(clip.sample_dir / "metadata" / "frame_times.npy")
        n_feat = 1 + n_wav // DATA_HOP
        n = min(state.shape[0], times.shape[0])
        gaps.append(int(n_feat - state.shape[0]))
        time_err.append(float(np.max(np.abs(times[:n] - np.arange(n) * DT))))
        speed = np.hypot(state[:n, 1], state[:n, 3])
        moving = speed > 0.5
        spike = moving & (np.abs(state[:n, 3]) >= TURN_MPS)
        if not np.any(spike):
            continue
        delay = np.hypot(state[:n, 0], state[:n, 2]) / SPEED_OF_SOUND
        arrive = times[:n] + delay
        wav_end = n_wav / float(sr)
        hot += int(spike.sum())
        hot_inside_wav += int(np.sum(spike & (arrive <= wav_end + DT)))
        idx = np.flatnonzero(moving)
        t0, t1 = float(times[idx[0]]), float(times[idx[-1]])
        hot_inside_moving += int(np.sum(spike & (arrive >= t0 - DT) & (arrive <= t1 + DT)))
    gaps_a = np.asarray(gaps)
    report = {
        "clips": len(clips),
        "sample_rate_mismatches": sr_bad,
        "frame_count_gap_median": float(np.median(gaps_a)),
        "frame_count_gap_min": int(gaps_a.min()),
        "frame_count_gap_max": int(gaps_a.max()),
        "frame_time_error_max_s": float(np.max(time_err)),
        "frame_dt_s": DT,
        "speed_of_sound_mps": SPEED_OF_SOUND,
        "hot_vy_frames": hot,
        "hot_arrival_inside_wav": hot_inside_wav,
        "hot_arrival_inside_moving_window": hot_inside_moving,
        "clock_ok": (
            not sr_bad
            and abs(int(gaps_a.min())) <= 1
            and abs(int(gaps_a.max())) <= 1
            and float(np.max(time_err)) < 1e-3
        ),
        "late_turn_in_waveform": hot > 0 and hot_inside_wav == hot,
    }
    report["passed"] = bool(report["clock_ok"] and report["late_turn_in_waveform"])
    return report


def _prepare(clips, indices: list[int]) -> list[dict]:
    rows = []
    for count, index in enumerate(indices, start=1):
        clip = clips[index]
        feat, state = load_arrays(clip)
        feat = standardize_mag(feat)
        n = min(feat.shape[1], state.shape[0])
        rows.append(
            {
                "clip_id": clip.clip_id,
                "vehicle": clip.vehicle,
                "feat": np.ascontiguousarray(feat[:, :n]),
                "state": np.ascontiguousarray(state[:n].astype(np.float32)),
            }
        )
        if count % 100 == 0 or count == len(indices):
            print(f"  features {count}/{len(indices)}", flush=True)
    return rows


def _collate(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(int(row["feat"].shape[-1]) for row in rows)
    feat = torch.zeros(len(rows), rows[0]["feat"].shape[0], length)
    state = torch.zeros(len(rows), length, 4)
    lengths = torch.zeros(len(rows), dtype=torch.long)
    for i, row in enumerate(rows):
        n = int(row["feat"].shape[-1])
        feat[i, :, :n] = torch.from_numpy(row["feat"])
        state[i, :n] = torch.from_numpy(row["state"])
        lengths[i] = n
    return feat, state, lengths


def _turn_mask(velocity: np.ndarray) -> np.ndarray:
    n0 = max(int(velocity.shape[0]) // 5, 1)
    base = float(np.mean(velocity[:n0]))
    return np.abs(velocity - base) >= TURN_MPS


def _score_one(vx: np.ndarray, vy: np.ndarray, xy: np.ndarray, state: np.ndarray) -> dict[str, float]:
    n = min(vx.shape[0], state.shape[0])
    gt = state[:n]
    delta = xy[:n] - gt[:, [0, 2]]
    zero = velocity_bin_index(gt[:, 3]) == ZERO_BIN
    pred_bin = velocity_bin_index(vy[:n])
    false_motion = float(np.mean(pred_bin[zero] != ZERO_BIN)) if np.any(zero) else float("nan")
    turn = _turn_mask(gt[:, 3])
    turn_err = np.abs(vy[:n] - gt[:, 3])
    return {
        "vx_mae": float(np.mean(np.abs(vx[:n] - gt[:, 1]))),
        "vy_turn_mae": float(np.mean(turn_err[turn])) if np.any(turn) else float("nan"),
        "zero_false_motion": false_motion,
        "trajectory_rmse_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "anchor_error_m": float(np.hypot(xy[0, 0] - gt[0, 0], xy[0, 1] - gt[0, 2])),
    }


def _mean(rows: list[dict]) -> dict[str, float]:
    return {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}


@torch.no_grad()
def _evaluate(model: Epoch38Net, rows: list[dict], psi: np.ndarray, device: torch.device) -> tuple[list[dict], dict]:
    model.eval()
    scored = []
    for start in range(0, len(rows), 16):
        chunk = rows[start : start + 16]
        feat, state, lengths = _collate(chunk)
        out = model(feat.to(device), lengths.to(device))
        log_x = out["vel_x"].log_softmax(-1).cpu().numpy()
        log_y = out["vel_y"].log_softmax(-1).cpu().numpy()
        for i, row in enumerate(chunk):
            n = int(lengths[i])
            gt = row["state"][:n]
            traj = reconstruct(
                log_x[i, :n],
                log_y[i, :n],
                DT,
                x0=float(gt[0, 0]),
                y0=float(gt[0, 2]),
                psi=psi,
            )
            scored.append(_score_one(traj.vx, traj.vy, traj.xy, gt))
    return scored, _mean(scored)


def _batch_loss(out: dict, state: torch.Tensor, lengths: torch.Tensor, balance_vy: bool) -> tuple[torch.Tensor, dict]:
    valid = torch.arange(state.shape[1], device=state.device)[None, :] < lengths[:, None]
    bins_x = _bin_index(state[:, :, 1])
    bins_y = _bin_index(state[:, :, 3])
    nll_x = F.cross_entropy(out["vel_x"].transpose(1, 2), bins_x, reduction="none")
    nll_y = F.cross_entropy(out["vel_y"].transpose(1, 2), bins_y, reduction="none")
    lx = (nll_x * valid).sum() / valid.sum().clamp(min=1)
    if balance_vy:
        ly = zero_bin_balanced_nll(nll_y, bins_y, valid, ZERO_BIN)
    else:
        ly = (nll_y * valid).sum() / valid.sum().clamp(min=1)
    return lx + ly, {"vx": float(lx.detach()), "vy": float(ly.detach()), "total": float((lx + ly).detach())}


def _qualifies(metrics: dict, baseline: dict) -> bool:
    return bool(
        metrics["vy_turn_mae"] <= 0.90 * baseline["vy_turn_mae"]
        and metrics["trajectory_rmse_m"] < baseline["trajectory_rmse_m"]
        and metrics["vx_mae"] <= 1.05 * baseline["vx_mae"]
    )


def _snapshot(model: Epoch38Net, path: Path, payload: dict) -> None:
    if OUT not in path.resolve().parents and path.resolve() != OUT:
        raise RuntimeError(f"refusing to write outside {OUT}: {path}")
    body = dict(payload)
    body["model"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(body, path)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        number = value.item()
        if isinstance(number, float) and not np.isfinite(number):
            return None
        return number
    return value


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def _overlay(clip_id: str, vehicle: str, state: np.ndarray, curves: dict[str, tuple], path: Path) -> None:
    time = np.arange(state.shape[0]) * DT
    gt_xy = state[:, [0, 2]]
    names = list(curves)
    styles = {
        "ground truth": ("black", 1.6, "-"),
        "epoch 38": ("0.35", 1.3, "--"),
        "ordinary CE": ("#1f77b4", 1.35, "-"),
        "vy balance": ("#d62728", 1.35, "-"),
    }
    fig = plt.figure(figsize=(12.4, 7.2))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.15, 1.15, 1.0))
    ax_vx = fig.add_subplot(grid[0, 0])
    ax_vy = fig.add_subplot(grid[0, 1], sharex=ax_vx)
    ax_x = fig.add_subplot(grid[1, 0], sharex=ax_vx)
    ax_y = fig.add_subplot(grid[1, 1], sharex=ax_vx)
    ax_path = fig.add_subplot(grid[:, 2])

    def limits(series: list[np.ndarray]) -> tuple[float, float]:
        lo = min(float(np.min(item)) for item in series)
        hi = max(float(np.max(item)) for item in series)
        span = max(hi - lo, 1.0)
        return lo - 0.04 * span, hi + 0.04 * span

    vx_series = [state[:, 1]] + [curves[name][0] for name in names]
    vy_series = [state[:, 3]] + [curves[name][1] for name in names]
    x_series = [gt_xy[:, 0]] + [curves[name][2][:, 0] for name in names]
    y_series = [gt_xy[:, 1]] + [curves[name][2][:, 1] for name in names]
    for ax, truth, column, ylabel, ylim in (
        (ax_vx, state[:, 1], 0, r"$v_x$ (m/s)", limits(vx_series)),
        (ax_vy, state[:, 3], 1, r"$v_y$ (m/s)", limits(vy_series)),
    ):
        ax.plot(time, truth, color="black", lw=1.6, label="ground truth")
        for name in names:
            color, width, ls = styles[name]
            ax.plot(time, curves[name][column], color=color, lw=width, ls=ls, label=name)
        ax.set_xlim(float(time[0]), float(time[-1]))
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    for ax, column, ylabel, ylim in (
        (ax_x, 0, r"$x$ (m)", limits(x_series)),
        (ax_y, 1, r"$y$ (m)", limits(y_series)),
    ):
        ax.plot(time, gt_xy[:, column], color="black", lw=1.6)
        for name in names:
            color, width, ls = styles[name]
            ax.plot(time, curves[name][2][:, column], color=color, lw=width, ls=ls)
        ax.set_xlim(float(time[0]), float(time[-1]))
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("t (s)")
        ax.grid(True, alpha=0.3)
    ax_path.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", lw=1.6, label="ground truth")
    for name in names:
        color, width, ls = styles[name]
        xy = curves[name][2]
        ax_path.plot(xy[:, 0], xy[:, 1], color=color, lw=width, ls=ls, label=name)
    ax_path.scatter([gt_xy[0, 0]], [gt_xy[0, 1]], s=26, color="black", zorder=4)
    ax_path.scatter([0.0], [0.0], marker="+", s=40, color="tab:red", zorder=4, label="microphone")
    ax_path.set_xlim(*limits(x_series + [np.array([0.0])]))
    ax_path.set_ylim(*limits(y_series + [np.array([0.0])]))
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.grid(True, alpha=0.3)
    ax_vx.set_title(r"$v_x$")
    ax_vy.set_title(r"$v_y$")
    ax_x.set_title(r"$x$")
    ax_y.set_title(r"$y$")
    handles, labels = ax_path.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"{clip_id}  {vehicle}", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _turning_figure(history: dict[str, list[dict]], baseline: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, color in (("ordinary CE", "#1f77b4"), ("vy balance", "#d62728")):
        epochs = [row["epoch"] for row in history[name]]
        ax.plot(epochs, [row["train"]["vy_turn_mae"] for row in history[name]], color=color, ls="--", label=f"{name} train")
        ax.plot(epochs, [row["val"]["vy_turn_mae"] for row in history[name]], color=color, ls="-", label=f"{name} val")
    ax.axhline(baseline, color="0.35", ls=":", label="epoch 38 val")
    ax.set_xlabel("fine-tune epoch")
    ax.set_ylabel(r"turning $v_y$ MAE (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def _predict_rows(model: Epoch38Net, rows: list[dict], psi: np.ndarray, device: torch.device) -> dict[str, tuple]:
    model.eval()
    curves = {}
    for row in rows:
        feat, state, lengths = _collate([row])
        out = model(feat.to(device), lengths.to(device))
        n = int(lengths[0])
        gt = row["state"][:n]
        traj = reconstruct(
            out["vel_x"][0, :n].log_softmax(-1).cpu().numpy(),
            out["vel_y"][0, :n].log_softmax(-1).cpu().numpy(),
            DT,
            x0=float(gt[0, 0]),
            y0=float(gt[0, 2]),
            psi=psi,
        )
        curves[row["clip_id"]] = (traj.vx, traj.vy, traj.xy, gt)
    return curves


def _wrong_audio(model: Epoch38Net, rows: list[dict], psi: np.ndarray, device: torch.device) -> dict[str, float]:
    """Score each validation state against the next clip's spectrogram."""
    model.eval()
    scored = []
    order = list(range(len(rows)))
    for i, row in enumerate(rows):
        donor = rows[order[(i + 1) % len(rows)]]
        n = min(row["state"].shape[0], donor["feat"].shape[1], row["feat"].shape[1])
        paired = {
            "feat": donor["feat"][:, :n],
            "state": row["state"][:n],
            "clip_id": row["clip_id"],
            "vehicle": row["vehicle"],
        }
        correct = {"feat": row["feat"][:, :n], "state": row["state"][:n], "clip_id": row["clip_id"], "vehicle": row["vehicle"]}
        wrong_curve = _predict_rows(model, [paired], psi, device)[row["clip_id"]]
        right_curve = _predict_rows(model, [correct], psi, device)[row["clip_id"]]
        scored.append(
            {
                "wrong": _score_one(wrong_curve[0], wrong_curve[1], wrong_curve[2], paired["state"]),
                "right": _score_one(right_curve[0], right_curve[1], right_curve[2], correct["state"]),
            }
        )
    return {
        "correct_audio": _mean([item["right"] for item in scored]),
        "wrong_audio": _mean([item["wrong"] for item in scored]),
    }


def _metric_table(rows: list[list[str]]) -> str:
    headers = ["Model", "Split", r"Turning $v_y$ MAE", "Zero-bin false motion", r"$v_x$ MAE", "Trajectory RMSE"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = {str(path): _sha256(path) for path in PROTECTED}
    device = _device()
    clips = discover_clips(DATA)
    print("auditing timestamps and renderer delay", flush=True)
    audit = _audit(clips)
    print(
        f"  frame gap {audit['frame_count_gap_min']}..{audit['frame_count_gap_max']}  "
        f"time error {audit['frame_time_error_max_s']:.3e} s  "
        f"hot arrivals in wav {audit['hot_arrival_inside_wav']}/{audit['hot_vy_frames']}  "
        f"passed {audit['passed']}",
        flush=True,
    )
    if not audit["passed"]:
        (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
        raise SystemExit("timestamp or late-turn coverage check failed; training was not started")

    psi = transition_matrix(DT, a_max_for_jump(JUMP_MPS, DT), lam=LAMBDA)
    splits = split_by_vehicle(clips, seed=SEED)
    print(f"device {device}  lr {LR:.1e}  epochs {EPOCHS}  batch {BATCH}", flush=True)
    print("loading train and val features", flush=True)
    train_rows = _prepare(clips, splits["train"])
    val_rows = _prepare(clips, splits["val"])
    by_id = {row["clip_id"]: row for row in train_rows + val_rows}
    val_plot = []
    for vehicle in sorted({row["vehicle"] for row in val_rows}):
        ids = sorted(row["clip_id"] for row in val_rows if row["vehicle"] == vehicle)
        val_plot.append(ids[0])
    plot_ids = list(PLOT_TEST) + val_plot
    for clip_id in PLOT_TEST:
        if clip_id not in by_id:
            clip = next(item for item in clips if item.clip_id == clip_id)
            feat, state = load_arrays(clip)
            feat = standardize_mag(feat)
            n = min(feat.shape[1], state.shape[0])
            by_id[clip_id] = {
                "clip_id": clip_id,
                "vehicle": clip.vehicle,
                "feat": np.ascontiguousarray(feat[:, :n]),
                "state": np.ascontiguousarray(state[:n].astype(np.float32)),
            }

    both_groups = 0
    nonzero_frames = 0
    frames = 0
    for row in train_rows:
        bins = velocity_bin_index(row["state"][:, 3])
        nonzero_frames += int(np.sum(bins != ZERO_BIN))
        frames += int(bins.size)
        if np.any(bins == ZERO_BIN) and np.any(bins != ZERO_BIN):
            both_groups += 1
    print(
        f"train clips with both vy groups {both_groups}/{len(train_rows)}  "
        f"nonzero vy frames {nonzero_frames / frames:.3%}",
        flush=True,
    )

    torch.manual_seed(SEED)
    ordinary = _load_parent(device)
    balanced = _load_parent(device)
    frozen_conv = {key: value.detach().cpu().clone() for key, value in ordinary.conv.state_dict().items()}
    opt_ordinary = torch.optim.AdamW((p for p in ordinary.parameters() if p.requires_grad), lr=LR, weight_decay=WEIGHT_DECAY)
    opt_balanced = torch.optim.AdamW((p for p in balanced.parameters() if p.requires_grad), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)

    print("evaluating epoch 38", flush=True)
    _, baseline_val = _evaluate(ordinary, val_rows, psi, device)
    _, baseline_train = _evaluate(ordinary, train_rows, psi, device)
    print(
        f"  val turn vy {baseline_val['vy_turn_mae']:.3f}  false {baseline_val['zero_false_motion']:.3f}  "
        f"vx {baseline_val['vx_mae']:.3f}  path {baseline_val['trajectory_rmse_m']:.3f}  "
        f"anchor {baseline_val['anchor_error_m']:.3e}",
        flush=True,
    )
    if baseline_val["anchor_error_m"] > 1e-5 or baseline_train["anchor_error_m"] > 1e-5:
        raise SystemExit("integration anchor is not the supplied position")

    runs = {
        "ordinary CE": {"model": ordinary, "opt": opt_ordinary, "balance": False, "history": []},
        "vy balance": {"model": balanced, "opt": opt_balanced, "balance": True, "history": []},
    }
    kept: dict[str, dict] = {}

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_rows))
        totals = {name: [] for name in runs}
        for name, run in runs.items():
            run["model"].train()
        for start in range(0, len(order), BATCH):
            batch = [train_rows[int(i)] for i in order[start : start + BATCH]]
            feat, state, lengths = _collate(batch)
            feat, state, lengths = feat.to(device), state.to(device), lengths.to(device)
            for name, run in runs.items():
                run["opt"].zero_grad(set_to_none=True)
                out = run["model"](feat, lengths)
                loss, parts = _batch_loss(out, state, lengths, run["balance"])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss in {name}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_((p for p in run["model"].parameters() if p.requires_grad), GRAD_CLIP)
                run["opt"].step()
                totals[name].append(parts)
        print(f"epoch {epoch:02d} scored", end="", flush=True)
        for name, run in runs.items():
            _, train_metrics = _evaluate(run["model"], train_rows, psi, device)
            _, val_metrics = _evaluate(run["model"], val_rows, psi, device)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean([item["total"] for item in totals[name]])),
                "train": train_metrics,
                "val": val_metrics,
                "qualifies": _qualifies(val_metrics, baseline_val),
            }
            run["history"].append(row)
            slot = kept.setdefault(name, {})
            if row["qualifies"]:
                previous = slot.get("qualifying")
                if previous is None or val_metrics["trajectory_rmse_m"] < previous["val"]["trajectory_rmse_m"]:
                    slot["qualifying"] = row
                    _snapshot(
                        run["model"],
                        OUT / f"{'ordinary' if name == 'ordinary CE' else 'vy_balance'}_qualifying.pt",
                        {"run": name, "epoch": epoch, "val": val_metrics, "qualifies": True},
                    )
            previous_turn = slot.get("lowest_turn")
            if previous_turn is None or val_metrics["vy_turn_mae"] < previous_turn["val"]["vy_turn_mae"]:
                slot["lowest_turn"] = row
                _snapshot(
                    run["model"],
                    OUT / f"{'ordinary' if name == 'ordinary CE' else 'vy_balance'}_lowest_turn.pt",
                    {"run": name, "epoch": epoch, "val": val_metrics, "qualifies": row["qualifies"]},
                )
            print(
                f"  {name}: train turn {train_metrics['vy_turn_mae']:.3f}  "
                f"val turn {val_metrics['vy_turn_mae']:.3f}  "
                f"vx {val_metrics['vx_mae']:.3f}  path {val_metrics['trajectory_rmse_m']:.3f}  "
                f"gate {row['qualifies']}",
                flush=True,
            )

    for name, run in runs.items():
        for key, value in run["model"].conv.state_dict().items():
            if not torch.equal(value.detach().cpu(), frozen_conv[key]):
                raise RuntimeError(f"frozen CNN weights changed in {name}")

    def reported(name: str) -> tuple[dict, str]:
        slot = kept[name]
        if "qualifying" in slot:
            return slot["qualifying"], "qualifying"
        return slot["lowest_turn"], "lowest validation turning vy, did not qualify"

    choices = []
    for name in ("ordinary CE", "vy balance"):
        row, reason = reported(name)
        choices.append((name, row, reason))
    qualifying = [item for item in choices if item[1]["qualifies"]]
    if qualifying:
        selected_name, selected_row, _ = min(qualifying, key=lambda item: item[1]["val"]["trajectory_rmse_m"])
        selected_file = OUT / ("ordinary_qualifying.pt" if selected_name == "ordinary CE" else "vy_balance_qualifying.pt")
        selected_note = f"{selected_name}, fine-tune epoch {selected_row['epoch']}"
        target = OUT / "selected.pt"
        target.write_bytes(selected_file.read_bytes())
    else:
        selected_name = "epoch 38"
        selected_row = None
        selected_file = PARENT
        selected_note = "epoch 38 retained; neither fine-tune met the validation gate"
        if (OUT / "selected.pt").exists():
            (OUT / "selected.pt").unlink()

    def load_reported(name: str) -> Epoch38Net:
        row, reason = reported(name)
        stem = "ordinary" if name == "ordinary CE" else "vy_balance"
        path = OUT / f"{stem}_qualifying.pt" if reason == "qualifying" else OUT / f"{stem}_lowest_turn.pt"
        model = Epoch38Net().to(device)
        saved = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        return model

    parent = _load_parent(device)
    models = {
        "epoch 38": parent,
        "ordinary CE": load_reported("ordinary CE"),
        "vy balance": load_reported("vy balance"),
    }
    plot_rows = [by_id[clip_id] for clip_id in plot_ids]
    drawn = {name: _predict_rows(model, plot_rows, psi, device) for name, model in models.items()}
    for row in plot_rows:
        curves = {name: drawn[name][row["clip_id"]][:3] for name in models}
        _overlay(row["clip_id"], row["vehicle"], drawn["epoch 38"][row["clip_id"]][3], curves, OUT / f"{row['clip_id']}.png")
    _turning_figure({name: runs[name]["history"] for name in runs}, baseline_val["vy_turn_mae"], OUT / "turning_curves.png")

    print("wrong-audio ablation", flush=True)
    ablation = {name: _wrong_audio(model, val_rows, psi, device) for name, model in models.items()}

    after = {str(path): _sha256(path) for path in PROTECTED}
    if before != after:
        raise RuntimeError("an existing checkpoint hash changed")

    config = {
        "parent": str(PARENT),
        "architecture": "epoch 38 CNN + BiLSTM + separate vx/vy classifiers; pos and acc loaded and frozen; no origin input",
        "frozen": "conv, ctc_x, ctc_y, pos, acc",
        "trained": "rnn, vel_x, vel_y",
        "lr": LR,
        "original_base_lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "seed": SEED,
        "augmentation": "none",
        "crop": "none",
        "ctc": 0.0,
        "trajectory_loss": 0.0,
        "decoder": {"jump_mps": JUMP_MPS, "transition_lambda": LAMBDA, "dt_s": DT},
        "anchor": "state[0] x,y",
        "vy_loss": "0.5 mean CE on zero bin + 0.5 mean CE on other bins, per clip",
        "vx_loss": "ordinary mean CE on valid frames",
        "gate": "val turning vy MAE <= 90% of epoch 38, lower trajectory RMSE, vx MAE <= 105% of epoch 38",
        "plot_clips": plot_ids,
    }
    summary = {
        "audit": audit,
        "config": config,
        "baseline_val": baseline_val,
        "baseline_train": baseline_train,
        "history": {name: runs[name]["history"] for name in runs},
        "reported": {
            name: {"row": reported(name)[0], "reason": reported(name)[1]} for name in runs
        },
        "selected": selected_note,
        "selected_file": str(selected_file),
        "ablation": ablation,
        "protected_unchanged": True,
        "train_clips_with_both_vy_groups": both_groups,
        "train_nonzero_vy_frame_fraction": nonzero_frames / frames,
    }
    (OUT / "recovery.json").write_text(json.dumps(_jsonable(summary), indent=2))

    table_rows = [
        ["Epoch 38", "train", _fmt(baseline_train["vy_turn_mae"]), _fmt(baseline_train["zero_false_motion"]), _fmt(baseline_train["vx_mae"]), _fmt(baseline_train["trajectory_rmse_m"])],
        ["Epoch 38", "val", _fmt(baseline_val["vy_turn_mae"]), _fmt(baseline_val["zero_false_motion"]), _fmt(baseline_val["vx_mae"]), _fmt(baseline_val["trajectory_rmse_m"])],
    ]
    for name, _row, reason in choices:
        shown, _ = reported(name)
        label = f"{name} ep {shown['epoch']}"
        table_rows.append(
            [label, "train", _fmt(shown["train"]["vy_turn_mae"]), _fmt(shown["train"]["zero_false_motion"]), _fmt(shown["train"]["vx_mae"]), _fmt(shown["train"]["trajectory_rmse_m"])]
        )
        table_rows.append(
            [label + (" qualified" if shown["qualifies"] else " not qualified"), "val", _fmt(shown["val"]["vy_turn_mae"]), _fmt(shown["val"]["zero_false_motion"]), _fmt(shown["val"]["vx_mae"]), _fmt(shown["val"]["trajectory_rmse_m"])]
        )
    abl_rows = []
    for name in models:
        right, wrong = ablation[name]["correct_audio"], ablation[name]["wrong_audio"]
        abl_rows.append(
            [name, "correct", _fmt(right["vy_turn_mae"]), _fmt(right["zero_false_motion"]), _fmt(right["vx_mae"]), _fmt(right["trajectory_rmse_m"])]
        )
        abl_rows.append(
            [name, "wrong clip", _fmt(wrong["vy_turn_mae"]), _fmt(wrong["zero_false_motion"]), _fmt(wrong["vx_mae"]), _fmt(wrong["trajectory_rmse_m"])]
        )
    tail = audit["hot_vy_frames"] - audit["hot_arrival_inside_moving_window"]
    text = f"""# Vy recovery from epoch 38

Selected checkpoint: **{selected_note}**.

The acoustic CNN stayed frozen. The BiLSTM and the two velocity heads were fine-tuned for {EPOCHS} epochs at learning rate {LR:.1e}, one tenth of the original base rate {BASE_LR:.1e}. Both runs used the same full clips, the same shuffle, and the same {EPOCHS * int(np.ceil(len(train_rows) / BATCH))} updates. CTC and the anchored trajectory loss stayed at weight 0. Their code in `motion_loss.py` was not removed. Position, acceleration, and CTC heads were loaded from epoch 38 and left frozen. No origin input was added. The integrator still starts at the labeled `(x, y)`.

## Data checks, before training

Spectrogram frames are `1 + n_samples / 512` at 44.1 kHz. Label `frame_times` match `k * 512/44100` with a maximum error of {audit['frame_time_error_max_s']:.3e} s. Frame-count differences are between {audit['frame_count_gap_min']} and {audit['frame_count_gap_max']}.

Renderer delay uses 343 m/s, the speed of sound in the path synthesizer. Every moving frame with `|vy| >= 2` has its reception time `t + range/c` inside the waveform ({audit['hot_arrival_inside_wav']} / {audit['hot_vy_frames']}). {tail} of those frames arrive during the stationary tail, which the existing trim does not feed to the network. The waveform itself contains them, and the label clock already matches the spectrogram clock, so the pairing was left as exported.

{both_groups} of {len(train_rows)} training clips contain both a zero `vy` bin and some other `vy` bin. Nonzero `vy` frames are {100 * nonzero_frames / frames:.1f}% of training frames.

## Gate

On the validation vehicles, a fine-tune qualifies only if turning `vy` MAE is at most 90% of epoch 38, trajectory RMSE is lower, and `vx` MAE is at most 105% of epoch 38. The saved candidate is the qualifying epoch with the lowest validation trajectory RMSE. A run with no qualifying epoch is reported at its lowest validation turning-`vy` epoch and is not selected.

{selected_note[0].upper() + selected_note[1:]}.

## Comparison

Turning `vy` uses frames whose ground-truth `vy` has left the mean of the first fifth by at least 2 m/s. Zero-bin false motion is the fraction of frames labeled as the 0 m/s bin whose decoded bin is something else. Trajectory RMSE is the anchored Viterbi integral.

{_metric_table(table_rows)}

Train and validation turning `vy` over the fine-tune:

![turning curves](turning_curves.png)

## Wrong-audio ablation

Each validation state is scored twice on the same prefix: with its own spectrogram, and with the next validation clip's spectrogram. The anchor stays the true initial position.

{_metric_table(abl_rows)}

## Configuration

```json
{json.dumps(config, indent=2)}
```

Existing checkpoints were read only. Their hashes were unchanged at the end of the run.
"""
    (OUT / "RECOVERY.md").write_text(text)
    print(selected_note, flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
