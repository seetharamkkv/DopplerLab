"""Train one full motion model: frequency CNN, BiLSTM, CTC, and integration.

The acoustic CNN is trained. Trailing audio that contains the delayed sound of
the last supervised motion is kept as encoder context and is left out of the
losses and the reported trajectory. Existing checkpoints are only read.

Run from the DopplerLab root:

    position_ctc/.venv/bin/python -u position_ctc/scripts/train_full.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))
sys.path.insert(0, str(ROOT / "position_ctc" / "scripts"))

from position_ctc.bins import CENTERS_MPS, velocity_bin_index  # noqa: E402
from position_ctc.data import (  # noqa: E402
    SOUND_SPEED_MPS,
    discover_clips,
    load_motion_window,
    split_by_vehicle,
    standardize_mag,
)
from position_ctc.features import DATA_HOP, DATA_SR, N_MELS  # noqa: E402
from position_ctc.integrate import integrate_trapezoid  # noqa: E402
from position_ctc.motion_loss import FullObjective, FullWeights  # noqa: E402
from position_ctc.motion_model import FullMotionNet, load_epoch38  # noqa: E402
from position_ctc.transitions import a_max_for_jump, transition_matrix  # noqa: E402
from position_ctc.paths import resolve_audio_clips  # noqa: E402
from position_ctc.viterbi import viterbi  # noqa: E402
from recover_vy import Epoch38Net  # noqa: E402

MOTION = ROOT / "position_ctc" / "outputs" / "motion"
OUT = MOTION / "full"
PARENT = MOTION / "motion_epoch38.pt"
PROTECTED = (
    PARENT,
    MOTION / "motion.pt",
    MOTION / "motion_polar_epoch27.pt",
    MOTION / "motion_corr.pt",
)
DT = DATA_HOP / float(DATA_SR)
JUMP_MPS = 6.0
LAMBDA = 1.0
TURN_MPS = 2.0
EPOCHS = 40
BATCH = 8
SEED = 0
LR_PRETRAINED = 1e-4
LR_NEW = 1e-3
WEIGHTS = FullWeights()
PLOT_HELD_OUT = ("sample_0000172", "sample_0000498")
ZERO_BIN = int(velocity_bin_index(np.array([0.0]))[0])


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


def _stats(rows: list[dict], sl: slice) -> tuple[float, float]:
    total = 0.0
    square = 0.0
    count = 0
    for row in rows:
        values = row["feat"][sl].astype(np.float64)
        total += float(values.sum())
        square += float(np.square(values).sum())
        count += int(values.size)
    mean = total / count
    var = max(square / count - mean * mean, 1e-8)
    return mean, float(np.sqrt(var))


def _apply_norm(rows: list[dict], mag: tuple[float, float], phase: tuple[float, float]) -> None:
    for row in rows:
        feat = row["feat"]
        feat[:N_MELS] = ((feat[:N_MELS] - mag[0]) / mag[1]).astype(np.float32)
        feat[N_MELS:] = ((feat[N_MELS:] - phase[0]) / phase[1]).astype(np.float32)


def _scales(rows: list[dict]) -> dict[str, float]:
    vx, vy, dx, dy, ax, ay = [], [], [], [], [], []
    origins = []
    for row in rows:
        n = row["n_super"]
        state = row["state"][:n]
        origins.append(state[0, [0, 2]])
        vx.append(state[:, 1])
        vy.append(state[:, 3])
        dx.append(state[:, 0] - state[0, 0])
        dy.append(state[:, 2] - state[0, 2])
        if n < 2:
            continue
        dt = np.diff(row["times"][:n])
        step_x = np.diff(state[:, 1])
        step_y = np.diff(state[:, 3])
        ok = (np.abs(step_x) <= JUMP_MPS) & (np.abs(step_y) <= JUMP_MPS)
        ax.append(step_x[ok] / dt[ok])
        ay.append(step_y[ok] / dt[ok])

    def std(parts: list[np.ndarray], floor: float) -> float:
        values = np.concatenate(parts)
        return max(float(np.std(values)), floor)

    origin = np.stack(origins)
    return {
        "s_vx": std(vx, 1.0),
        "s_vy": std(vy, 1.0),
        "s_x": std(dx, 1.0),
        "s_y": std(dy, 1.0),
        "s_ax": std(ax, 0.5),
        "s_ay": std(ay, 0.5),
        "origin_scale": max(float(np.std(origin)), 1.0),
    }


def _coverage(rows: list[dict]) -> dict[str, float]:
    hot = 0
    covered = 0
    tails = []
    outside = 0
    for row in rows:
        n = row["n_super"]
        state = row["state"]
        times = row["times"]
        tails.append(row["n_audio"] - n)
        spike = np.abs(state[:n, 3]) >= TURN_MPS
        if not np.any(spike):
            continue
        arrive = times[:n] + np.hypot(state[:n, 0], state[:n, 2]) / SOUND_SPEED_MPS
        last = float(times[row["n_audio"] - 1])
        hot += int(spike.sum())
        covered += int(np.sum(spike & (arrive <= last + 0.5 * DT)))
        outside += int(np.sum(spike & (arrive > last + 0.5 * DT)))
    return {
        "clips": len(rows),
        "mean_tail_frames": float(np.mean(tails)),
        "hot_frames": hot,
        "hot_covered": covered,
        "hot_past_audio": outside,
    }


def _collate(rows: list[dict]):
    length = max(row["n_audio"] for row in rows)
    feat = torch.zeros(len(rows), rows[0]["feat"].shape[0], length)
    state = torch.zeros(len(rows), length, 4)
    times = torch.zeros(len(rows), length)
    audio = torch.zeros(len(rows), dtype=torch.long)
    supervised = torch.zeros(len(rows), dtype=torch.long)
    for i, row in enumerate(rows):
        n = row["n_audio"]
        feat[i, :, :n] = torch.from_numpy(row["feat"])
        state[i, :n] = torch.from_numpy(row["state"])
        times[i, :n] = torch.from_numpy(row["times"].astype(np.float32))
        audio[i] = n
        supervised[i] = row["n_super"]
    origin = torch.stack((state[:, 0, 0], state[:, 0, 2]), dim=-1)
    return feat, state, times, audio, supervised, origin


def _turn_mask(velocity: np.ndarray) -> np.ndarray:
    n0 = max(int(velocity.shape[0]) // 5, 1)
    return np.abs(velocity - float(np.mean(velocity[:n0]))) >= TURN_MPS


def _decode(log_x: np.ndarray, log_y: np.ndarray, times: np.ndarray, x0: float, y0: float, psi: np.ndarray):
    vx = CENTERS_MPS[viterbi(log_x, psi)]
    vy = CENTERS_MPS[viterbi(log_y, psi)]
    dt = np.diff(times)
    xy = integrate_trapezoid(vx, vy, dt, x0=x0, y0=y0)
    return vx, vy, xy, dt


def _score(vx: np.ndarray, vy: np.ndarray, xy: np.ndarray, state: np.ndarray, psi: np.ndarray) -> dict[str, float]:
    gt = state[: vx.shape[0]]
    delta = xy - gt[:, [0, 2]]
    zero = velocity_bin_index(gt[:, 3]) == ZERO_BIN
    pred_bin = velocity_bin_index(vy)
    turn = _turn_mask(gt[:, 3])
    forbid = 0
    if vx.shape[0] > 1:
        ix = velocity_bin_index(vx)
        iy = velocity_bin_index(vy)
        forbid = int(np.sum(~np.isfinite(psi[ix[:-1], ix[1:]])) + np.sum(~np.isfinite(psi[iy[:-1], iy[1:]])))
    return {
        "vx_mae": float(np.mean(np.abs(vx - gt[:, 1]))),
        "vy_mae": float(np.mean(np.abs(vy - gt[:, 3]))),
        "vy_turn_mae": float(np.mean(np.abs(vy - gt[:, 3])[turn])) if np.any(turn) else float("nan"),
        "zero_false_motion": float(np.mean(pred_bin[zero] != ZERO_BIN)) if np.any(zero) else float("nan"),
        "trajectory_rmse_m": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "max_lateral_m": float(np.max(np.abs(delta[:, 1]))),
        "anchor_error_m": float(np.hypot(xy[0, 0] - gt[0, 0], xy[0, 1] - gt[0, 2])),
        "forbidden_steps": float(forbid),
    }


def _mean(rows: list[dict]) -> dict[str, float]:
    out = {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0] if key != "forbidden_steps"}
    out["forbidden_steps"] = float(np.sum([row["forbidden_steps"] for row in rows]))
    return out


@torch.no_grad()
def _forward_super(model, row: dict, device: torch.device, *, legacy: bool):
    n_super = row["n_super"]
    if legacy:
        feat_np = standardize_mag(row["feat"][:, :n_super].copy())
        feat = torch.from_numpy(feat_np).unsqueeze(0).to(device)
        out = model(feat, torch.tensor([n_super], device=device))
    else:
        feat = torch.from_numpy(row["feat"]).unsqueeze(0).to(device)
        origin = torch.tensor([[float(row["state"][0, 0]), float(row["state"][0, 2])]], dtype=torch.float32, device=device)
        out = model(feat, torch.tensor([row["n_audio"]], device=device), origin)
    log_x = out["vel_x"][0, :n_super].log_softmax(-1).cpu().numpy()
    log_y = out["vel_y"][0, :n_super].log_softmax(-1).cpu().numpy()
    return log_x, log_y


def _evaluate(model, rows: list[dict], psi: np.ndarray, device: torch.device, *, legacy: bool = False) -> dict[str, float]:
    model.eval()
    scored = []
    for row in rows:
        log_x, log_y = _forward_super(model, row, device, legacy=legacy)
        n = row["n_super"]
        gt = row["state"][:n]
        vx, vy, xy, _dt = _decode(log_x, log_y, row["times"][:n], float(gt[0, 0]), float(gt[0, 2]), psi)
        scored.append(_score(vx, vy, xy, gt, psi))
    return _mean(scored)


def _quantized(state: np.ndarray, times: np.ndarray):
    vx = CENTERS_MPS[velocity_bin_index(state[:, 1])]
    vy = CENTERS_MPS[velocity_bin_index(state[:, 3])]
    xy = integrate_trapezoid(vx, vy, np.diff(times), x0=float(state[0, 0]), y0=float(state[0, 2]))
    return vx, vy, xy


def _figure(clip_id: str, vehicle: str, state: np.ndarray, times: np.ndarray, pred, path: Path) -> None:
    t = times - times[0]
    qv = _quantized(state, times)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharex=True)
    panels = (
        (axes[0, 0], state[:, 1], pred[0], qv[0], r"$v_x$ (m/s)"),
        (axes[0, 1], state[:, 3], pred[1], qv[1], r"$v_y$ (m/s)"),
        (axes[1, 0], state[:, 0], pred[2][:, 0], qv[2][:, 0], r"$x$ (m)"),
        (axes[1, 1], state[:, 2], pred[2][:, 1], qv[2][:, 1], r"$y$ (m)"),
    )
    for ax, truth, guess, bins, label in panels:
        ax.plot(t, truth, color="black", lw=1.6, label="ground truth")
        ax.plot(t, guess, color="#1f77b4", lw=1.4, label="decoded")
        ax.plot(t, bins, color="0.45", lw=1.0, ls=":", label="2 m/s bins")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("t (s)")
    axes[1, 1].set_xlabel("t (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(f"{clip_id}  {vehicle}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        number = value.item()
        return None if isinstance(number, float) and not np.isfinite(number) else number
    return value


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = {str(path): _sha256(path) for path in PROTECTED}
    device = _device()
    clips = discover_clips(resolve_audio_clips(args.data))
    splits = split_by_vehicle(clips, seed=SEED)
    print(f"device {device}  loading windows", flush=True)
    rows = {}
    for name, indices in splits.items():
        loaded = []
        for count, index in enumerate(indices, start=1):
            loaded.append(load_motion_window(clips[index]))
            if count % 200 == 0 or count == len(indices):
                print(f"  {name} {count}/{len(indices)}", flush=True)
        rows[name] = loaded
    for row in rows["train"] + rows["val"] + rows["test"]:
        velocity = np.concatenate([row["state"][: row["n_super"], 1], row["state"][: row["n_super"], 3]])
        if float(np.min(velocity)) < -30.0 - 1e-3 or float(np.max(velocity)) > 30.0 + 1e-3:
            raise SystemExit(f"{row['clip_id']} velocity leaves the alphabet")
        if np.any(np.diff(row["times"][: row["n_super"]]) <= 0):
            raise SystemExit(f"{row['clip_id']} timestamps are not increasing")
    cover = _coverage(rows["train"] + rows["val"] + rows["test"])
    print(
        f"tail {cover['mean_tail_frames']:.1f} frames  "
        f"hot covered {cover['hot_covered']}/{cover['hot_frames']}  past audio {cover['hot_past_audio']}",
        flush=True,
    )
    if cover["hot_frames"] < 1 or cover["hot_covered"] != cover["hot_frames"] or cover["mean_tail_frames"] < 1.0:
        raise SystemExit("trailing audio does not cover the delayed lateral motion; training was not started")

    psi = transition_matrix(DT, a_max_for_jump(JUMP_MPS, DT), lam=LAMBDA)
    implied = JUMP_MPS / DT
    print(
        f"decoder jump {JUMP_MPS:.0f} m/s per {DT * 1000:.2f} ms frame "
        f"({implied:.0f} m/s²). This is not a vehicle acceleration limit.",
        flush=True,
    )
    print("scoring epoch 38 on the supervised frames", flush=True)
    legacy = Epoch38Net().to(device)
    saved = torch.load(PARENT, map_location=device, weights_only=False)
    legacy.load_state_dict(saved["model"], strict=True)
    legacy.eval()
    baseline = _evaluate(legacy, rows["val"], psi, device, legacy=True)
    print(
        f"  epoch 38 val turn vy {baseline['vy_turn_mae']:.3f}  path {baseline['trajectory_rmse_m']:.3f}  "
        f"vx {baseline['vx_mae']:.3f}",
        flush=True,
    )
    del legacy
    scales = _scales(rows["train"])
    mag = _stats(rows["train"], slice(0, N_MELS))
    phase = _stats(rows["train"], slice(N_MELS, None))
    for split_rows in rows.values():
        _apply_norm(split_rows, mag, phase)
    print(
        f"scales vx {scales['s_vx']:.2f} vy {scales['s_vy']:.2f} "
        f"x {scales['s_x']:.2f} y {scales['s_y']:.2f} "
        f"ax {scales['s_ax']:.2f} ay {scales['s_ay']:.2f} origin {scales['origin_scale']:.2f}",
        flush=True,
    )
    torch.manual_seed(SEED)
    model = FullMotionNet().to(device)
    init_report = load_epoch38(model, PARENT, scales["origin_scale"])
    print("initialized", json.dumps(init_report), flush=True)
    objective = FullObjective(WEIGHTS, jump_mps=JUMP_MPS, scales=scales).to(device)
    pretrained, fresh = [], []
    for name, param in model.named_parameters():
        if name.startswith(("cnn.", "proj.", "origin.", "pos.", "acc.")):
            fresh.append(param)
        else:
            pretrained.append(param)
    opt = torch.optim.AdamW(
        [{"params": pretrained, "lr": LR_PRETRAINED}, {"params": fresh, "lr": LR_NEW}],
        weight_decay=1e-4,
    )
    rng = np.random.default_rng(SEED)
    steps_per_epoch = int(np.ceil(len(rows["train"]) / BATCH))
    total_steps = EPOCHS * steps_per_epoch
    warm_steps = max(int(round(0.1 * total_steps)), 1)
    history = []
    best_path = None
    best_turn = None
    step = 0
    grad_report = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(rows["train"]))
        totals: dict[str, list[float]] = {}
        for start in range(0, len(order), BATCH):
            step += 1
            batch = [rows["train"][int(i)] for i in order[start : start + BATCH]]
            feat, state, times, audio, supervised, origin = _collate(batch)
            feat = feat.to(device)
            state = state.to(device)
            times = times.to(device)
            audio = audio.to(device)
            supervised = supervised.to(device)
            origin = origin.to(device)
            aux_scale = min(1.0, step / warm_steps)
            opt.zero_grad(set_to_none=True)
            pred = model(feat, audio, origin)
            loss, parts = objective(pred, state, times, supervised, aux_scale)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {parts}")
            loss.backward()
            if grad_report is None:
                grad_report = {}
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    group = name.split(".", 1)[0]
                    grad_report[group] = grad_report.get(group, 0.0) + float(param.grad.detach().norm())
                if grad_report.get("cnn", 0.0) <= 0.0 or grad_report.get("rnn", 0.0) <= 0.0:
                    raise RuntimeError(f"encoder did not receive a gradient: {grad_report}")
                print("gradients " + " ".join(f"{key} {value:.3e}" for key, value in sorted(grad_report.items())), flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            for key, value in parts.items():
                totals.setdefault(key, []).append(value)
        model.eval()
        val = _evaluate(model, rows["val"], psi, device)
        record = {"epoch": epoch, "step": step, "train": {key: float(np.mean(values)) for key, values in totals.items()}, "val": val}
        history.append(record)
        improved_path = best_path is None or val["trajectory_rmse_m"] < best_path["val"]["trajectory_rmse_m"]
        if improved_path:
            best_path = record
            torch.save(
                {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epoch": epoch, "val": val},
                OUT / "best_path.pt",
            )
        turn_better = val["vy_turn_mae"] < baseline["vy_turn_mae"]
        improved_turn = turn_better and (best_turn is None or val["vy_turn_mae"] < best_turn["val"]["vy_turn_mae"])
        if improved_turn:
            best_turn = record
            torch.save(
                {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epoch": epoch, "val": val},
                OUT / "best_turn.pt",
            )
        print(
            f"epoch {epoch:02d}  aux {record['train']['aux_scale']:.2f}  "
            f"vx {record['train']['vx']:.3f}  vy {record['train']['vy']:.3f}  "
            f"ctc {record['train']['ctc']:.3f}  int {record['train']['integrated']:.3f}  "
            f"dir {record['train']['direct']:.3f}  acc {record['train']['acceleration']:.3f}  "
            f"pv {record['train']['pv']:.3f}  va {record['train']['va']:.3f}  "
            f"val path {val['trajectory_rmse_m']:.3f}  turn vy {val['vy_turn_mae']:.3f}  "
            f"vx {val['vx_mae']:.3f}  false {val['zero_false_motion']:.3f}",
            flush=True,
        )

    chosen = torch.load(OUT / "best_path.pt", map_location=device, weights_only=False)
    model.load_state_dict(chosen["model"])
    train_metrics = _evaluate(model, rows["train"], psi, device)
    val_metrics = _evaluate(model, rows["val"], psi, device)
    by_id = {row["clip_id"]: row for split_rows in rows.values() for row in split_rows}
    val_ids = []
    for vehicle in sorted({row["vehicle"] for row in rows["val"]}):
        ids = sorted(row["clip_id"] for row in rows["val"] if row["vehicle"] == vehicle)
        for frac in (0.0, 0.25, 0.5, 0.75):
            val_ids.append(ids[int(round(frac * (len(ids) - 1)))])
    plot_ids = list(dict.fromkeys(val_ids + list(PLOT_HELD_OUT)))
    model.eval()
    for clip_id in plot_ids:
        row = by_id[clip_id]
        log_x, log_y = _forward_super(model, row, device, legacy=False)
        n = row["n_super"]
        gt = row["state"][:n]
        pred = _decode(log_x, log_y, row["times"][:n], float(gt[0, 0]), float(gt[0, 2]), psi)
        _figure(clip_id, row["vehicle"], gt, row["times"][:n], pred, OUT / f"{clip_id}.png")
    after = {str(path): _sha256(path) for path in PROTECTED}
    if before != after:
        raise RuntimeError("an existing checkpoint hash changed")
    turn_improved = best_turn is not None
    config = {
        "parent": str(PARENT),
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "seed": SEED,
        "lr_pretrained": LR_PRETRAINED,
        "lr_new": LR_NEW,
        "pretrained_parameters": "rnn, vel_x, vel_y, ctc_x, ctc_y",
        "new_parameters": "cnn, proj, origin, pos, acc",
        "weights": WEIGHTS.__dict__,
        "scales": scales,
        "mag_mean_std": mag,
        "phase_mean_std": phase,
        "aux_ramp_steps": warm_steps,
        "total_steps": total_steps,
        "decoder": {
            "jump_mps_per_frame": JUMP_MPS,
            "frame_dt_s": DT,
            "implied_acceleration_mps2": implied,
            "transition_lambda": LAMBDA,
            "note": "6 m/s per frame is the audited graded Viterbi limit, not a realistic acceleration bound",
        },
        "init": init_report,
        "gradients": grad_report,
        "coverage": cover,
        "plot_clips": plot_ids,
    }
    summary = {
        "config": config,
        "epoch38_val": baseline,
        "best_path": best_path,
        "best_turn": best_turn,
        "turn_improved_over_epoch38": turn_improved,
        "selected_is_turn_recovery": bool(
            turn_improved and best_turn is not None and best_path is not None and best_turn["epoch"] == best_path["epoch"]
        ),
        "best_path_train": train_metrics,
        "best_path_val": val_metrics,
        "history": history,
        "protected_unchanged": True,
    }
    (OUT / "full.json").write_text(json.dumps(_jsonable(summary), indent=2))
    keys = (
        ("vx_mae", r"$v_x$ MAE"),
        ("vy_mae", r"$v_y$ MAE"),
        ("vy_turn_mae", r"Turning $v_y$ MAE"),
        ("zero_false_motion", "Zero-bin false motion"),
        ("trajectory_rmse_m", "Trajectory RMSE (m)"),
        ("max_lateral_m", "Max lateral error (m)"),
        ("anchor_error_m", "Anchor error (m)"),
        ("forbidden_steps", "Forbidden steps"),
    )
    lines = ["| Metric | Epoch 38 val | Best path, train | Best path, val |", "| --- | ---: | ---: | ---: |"]
    for key, label in keys:
        lines.append(
            f"| {label} | {_fmt(baseline[key])} | {_fmt(train_metrics[key])} | {_fmt(val_metrics[key])} |"
        )
    turn_line = (
        f"Epoch {best_turn['epoch']} reached validation turning $v_y$ MAE "
        f"{best_turn['val']['vy_turn_mae']:.3f}, below epoch 38's {baseline['vy_turn_mae']:.3f}."
        if turn_improved
        else f"No epoch beat epoch 38's validation turning $v_y$ MAE of {baseline['vy_turn_mae']:.3f}."
    )
    text = f"""# Full motion model

Selected checkpoint: `best_path.pt`, epoch {best_path['epoch']}, saved for the lowest validation trajectory RMSE ({best_path['val']['trajectory_rmse_m']:.3f} m). This is the integrated Viterbi trajectory. It is not, by itself, evidence that lateral turns were recovered.

{turn_line}

## Data

Observer-time labels are unchanged. The leading stationary pad is still removed. The trailing audio now extends through the propagation delay of the supervised motion (`range / 343`), so the bidirectional LSTM can hear the sound of the last lateral frames. Those extra frames are not supervised and are not included in the trajectory metrics.

Mean tail: {cover['mean_tail_frames']:.1f} frames. Delayed lateral frames covered: {cover['hot_covered']} / {cover['hot_frames']}.

Input mean and standard deviation, and the continuous loss scales, were computed on the training split only.

## Decoder

Dense $P_x$ and $P_y$ are decoded by independent Viterbi. CTC emissions are training targets only. The graded cost uses a {JUMP_MPS:.0f} m/s jump per {DT * 1000:.2f} ms frame, which is {implied:.0f} m/s². That cutoff is the audited decoder setting. It is not a claim about how fast the vehicle can accelerate. Bin centers run from −30 to 30 m/s in steps of 2. The integral uses each clip's frame timestamps and starts at the labeled $(x_0, y_0)$.

## Initialization

Epoch 38 supplied the LSTM recurrent weights, the velocity and CTC heads, and the position and acceleration matrices. Its convolution was not loaded: that network pooled time and mixed the mel bins, and this one downsamples frequency only. The new CNN, feature projection, and initial-position MLP were initialized explicitly. The extra LSTM input columns that carry $p_0$ were drawn from $N(0, 0.02)$.

Pretrained LSTM and velocity/CTC parameters use learning rate {LR_PRETRAINED:.1e}. The new CNN, projection, position MLP, and auxiliary position/acceleration heads use {LR_NEW:.1e}. Auxiliary loss weights ramp from 0 over the first {warm_steps} of {total_steps} updates. The first backward pass reached the CNN and the LSTM.

## Validation

{chr(10).join(lines)}

Forbidden steps are the total count on the split. The other rows are means over clips. Maximum lateral error is the mean, over clips, of each clip's largest $|\\hat y - y|$.

Train trajectory RMSE at this checkpoint is {train_metrics['trajectory_rmse_m']:.3f} m and validation is {val_metrics['trajectory_rmse_m']:.3f} m. Train turning $v_y$ MAE is {train_metrics['vy_turn_mae']:.3f}; validation is {val_metrics['vy_turn_mae']:.3f}.
"""
    (OUT / "FULL.md").write_text(text)
    print(text, flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
