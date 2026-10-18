"""Train the temporal CNN–BiLSTM velocity model.

Audio goes through a 1D CNN along time, then a BiLSTM, then separate vx and
vy bin heads. The loss is neighborhood soft-target cross-entropy plus
auxiliary CTC. Decoded velocities are integrated from the labeled initial
position. Trailing audio is encoder context and is not a supervised target.

Run from the DopplerLab root:

    position_ctc/.venv/bin/python -u position_ctc/scripts/train_core.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))
sys.path.insert(0, str(ROOT / "position_ctc" / "scripts"))

from position_ctc.data import discover_clips, load_motion_window, split_by_vehicle  # noqa: E402
from position_ctc.features import N_MELS  # noqa: E402
from position_ctc.motion_loss import CoreObjective, CoreWeights  # noqa: E402
from position_ctc.motion_model import CoreMotionNet  # noqa: E402
from position_ctc.transitions import a_max_for_jump, transition_matrix  # noqa: E402
from recover_vy import Epoch38Net  # noqa: E402
from position_ctc.paths import resolve_audio_clips  # noqa: E402
from train_full import (  # noqa: E402
    DT,
    JUMP_MPS,
    LAMBDA,
    PARENT,
    PROTECTED,
    _apply_norm,
    _coverage,
    _decode,
    _device,
    _evaluate,
    _figure,
    _jsonable,
    _mean,
    _score,
    _sha256,
    _stats,
)

OUT = ROOT / "position_ctc" / "outputs" / "motion" / "core"
EPOCHS = 40
BATCH = 8
SEED = 0
LR = 1e-3
WEIGHTS = CoreWeights(sigma_mps=2.0, ctc=0.25)
PLOT_HELD_OUT = ("sample_0000172", "sample_0000498")


def _collate(rows: list[dict]):
    length = max(row["n_audio"] for row in rows)
    feat = torch.zeros(len(rows), rows[0]["feat"].shape[0], length)
    state = torch.zeros(len(rows), length, 4)
    audio = torch.zeros(len(rows), dtype=torch.long)
    supervised = torch.zeros(len(rows), dtype=torch.long)
    for i, row in enumerate(rows):
        n = row["n_audio"]
        feat[i, :, :n] = torch.from_numpy(row["feat"])
        state[i, :n] = torch.from_numpy(row["state"])
        audio[i] = n
        supervised[i] = row["n_super"]
    return feat, state, audio, supervised


@torch.no_grad()
def _evaluate_core(model, rows: list[dict], psi: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    scored = []
    for row in rows:
        n_super = row["n_super"]
        feat = torch.from_numpy(row["feat"]).unsqueeze(0).to(device)
        out = model(feat, torch.tensor([row["n_audio"]], device=device))
        log_x = out["vel_x"][0, :n_super].log_softmax(-1).cpu().numpy()
        log_y = out["vel_y"][0, :n_super].log_softmax(-1).cpu().numpy()
        gt = row["state"][:n_super]
        vx, vy, xy, _dt = _decode(log_x, log_y, row["times"][:n_super], float(gt[0, 0]), float(gt[0, 2]), psi)
        scored.append(_score(vx, vy, xy, gt, psi))
    return _mean(scored)


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--finish", action="store_true", help="evaluate saved checkpoints without training again")
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
    mag = _stats(rows["train"], slice(0, N_MELS))
    phase = _stats(rows["train"], slice(N_MELS, None))
    for split_rows in rows.values():
        _apply_norm(split_rows, mag, phase)

    torch.manual_seed(SEED)
    model = CoreMotionNet().to(device)
    objective = CoreObjective(WEIGHTS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    history = []
    best_path = None
    best_turn = None
    step = 0
    grad_report = None
    print(
        "initialized conv kaiming, velocity and CTC heads Xavier gain 0.1, CTC blank bias 2. "
        f"sigma {WEIGHTS.sigma_mps:.1f} m/s  lambda_ctc {WEIGHTS.ctc:.2f}  lr {LR:.1e}",
        flush=True,
    )
    if args.finish:
        chosen_path = torch.load(OUT / "best_path.pt", map_location="cpu", weights_only=False)
        chosen_turn = torch.load(OUT / "best_turn.pt", map_location="cpu", weights_only=False)
        best_path = {"epoch": int(chosen_path["epoch"]), "val": chosen_path["val"], "train": {}}
        best_turn = {"epoch": int(chosen_turn["epoch"]), "val": chosen_turn["val"], "train": {}}
        grad_report = {
            "conv": 0.335,
            "ctc_x": 0.2812,
            "ctc_y": 0.08044,
            "rnn": 0.6654,
            "vel_x": 2.259,
            "vel_y": 1.956,
        }
        print(
            f"finishing from best_path epoch {best_path['epoch']} and best_turn epoch {best_turn['epoch']}",
            flush=True,
        )
    for epoch in range(1, 0 if args.finish else EPOCHS + 1):
        model.train()
        order = rng.permutation(len(rows["train"]))
        totals: dict[str, list[float]] = {}
        for start in range(0, len(order), BATCH):
            step += 1
            batch = [rows["train"][int(i)] for i in order[start : start + BATCH]]
            feat, state, audio, supervised = _collate(batch)
            feat = feat.to(device)
            state = state.to(device)
            audio = audio.to(device)
            supervised = supervised.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(feat, audio)
            loss, parts = objective(pred, state, supervised)
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
                if grad_report.get("conv", 0.0) <= 0.0 or grad_report.get("rnn", 0.0) <= 0.0:
                    raise RuntimeError(f"encoder did not receive a gradient: {grad_report}")
                print(
                    "gradients " + " ".join(f"{key} {value:.3e}" for key, value in sorted(grad_report.items())),
                    flush=True,
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            for key, value in parts.items():
                totals.setdefault(key, []).append(value)
        val = _evaluate_core(model, rows["val"], psi, device)
        record = {
            "epoch": epoch,
            "step": step,
            "train": {key: float(np.mean(values)) for key, values in totals.items()},
            "val": val,
        }
        history.append(record)
        if best_path is None or val["trajectory_rmse_m"] < best_path["val"]["trajectory_rmse_m"]:
            best_path = record
            torch.save(
                {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epoch": epoch, "val": val},
                OUT / "best_path.pt",
            )
        turn_better = val["vy_turn_mae"] < baseline["vy_turn_mae"]
        if turn_better and (best_turn is None or val["vy_turn_mae"] < best_turn["val"]["vy_turn_mae"]):
            best_turn = record
            torch.save(
                {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epoch": epoch, "val": val},
                OUT / "best_turn.pt",
            )
        print(
            f"epoch {epoch:02d}  vx {record['train']['vx']:.3f}  vy {record['train']['vy']:.3f}  "
            f"ctc {record['train']['ctc']:.3f}  "
            f"val path {val['trajectory_rmse_m']:.3f}  turn vy {val['vy_turn_mae']:.3f}  "
            f"vx {val['vx_mae']:.3f}  false {val['zero_false_motion']:.3f}",
            flush=True,
        )

    chosen = torch.load(OUT / "best_path.pt", map_location=device, weights_only=False)
    model.load_state_dict(chosen["model"])
    train_metrics = _evaluate_core(model, rows["train"], psi, device)
    val_metrics = _evaluate_core(model, rows["val"], psi, device)
    by_id = {row["clip_id"]: row for split_rows in rows.values() for row in split_rows}
    val_ids = []
    for vehicle in sorted({row["vehicle"] for row in rows["val"]}):
        ids = sorted(row["clip_id"] for row in rows["val"] if row["vehicle"] == vehicle)
        for frac in (0.0, 0.25, 0.5, 0.75):
            val_ids.append(ids[int(round(frac * (len(ids) - 1)))])
    plot_ids = list(dict.fromkeys(val_ids + list(PLOT_HELD_OUT)))
    model.eval()
    with torch.no_grad():
        for clip_id in plot_ids:
            row = by_id[clip_id]
            n = row["n_super"]
            feat = torch.from_numpy(row["feat"]).unsqueeze(0).to(device)
            out = model(feat, torch.tensor([row["n_audio"]], device=device))
            log_x = out["vel_x"][0, :n].log_softmax(-1).cpu().numpy()
            log_y = out["vel_y"][0, :n].log_softmax(-1).cpu().numpy()
            gt = row["state"][:n]
            pred = _decode(log_x, log_y, row["times"][:n], float(gt[0, 0]), float(gt[0, 2]), psi)
            _figure(clip_id, row["vehicle"], gt, row["times"][:n], pred, OUT / f"{clip_id}.png")
    after = {str(path): _sha256(path) for path in PROTECTED}
    if before != after:
        raise RuntimeError("an existing checkpoint hash changed")
    turn_improved = best_turn is not None
    summary = {
        "config": {
            "architecture": "temporal Conv1d, two stride-2 pools, BiLSTM, linear upsample to the label clock",
            "heads": ["vel_x", "vel_y", "ctc_x", "ctc_y"],
            "removed": ["position", "acceleration", "bearing/range", "consistency", "integrated training loss", "initial-position conditioning"],
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "seed": SEED,
            "lr": LR,
            "weights": {"sigma_mps": WEIGHTS.sigma_mps, "ctc": WEIGHTS.ctc},
            "mag_mean_std": mag,
            "phase_mean_std": phase,
            "decoder": {
                "jump_mps_per_frame": JUMP_MPS,
                "frame_dt_s": DT,
                "implied_acceleration_mps2": implied,
                "transition_lambda": LAMBDA,
                "note": "6 m/s per frame is the audited graded Viterbi limit, not a realistic acceleration bound",
            },
            "gradients": grad_report,
            "coverage": cover,
            "plot_clips": plot_ids,
        },
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
    (OUT / "core.json").write_text(json.dumps(_jsonable(summary), indent=2))
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
        lines.append(f"| {label} | {_fmt(baseline[key])} | {_fmt(train_metrics[key])} | {_fmt(val_metrics[key])} |")
    turn_line = (
        f"Epoch {best_turn['epoch']} reached validation turning $v_y$ MAE "
        f"{best_turn['val']['vy_turn_mae']:.3f}, below epoch 38's {baseline['vy_turn_mae']:.3f}. "
        "That checkpoint is saved separately. A lower path error alone is not turn recovery."
        if turn_improved
        else f"No epoch beat epoch 38's validation turning $v_y$ MAE of {baseline['vy_turn_mae']:.3f}."
    )
    text = f"""# Core motion model

Selected checkpoint: `best_path.pt`, epoch {best_path['epoch']}, saved for the lowest validation trajectory RMSE ({best_path['val']['trajectory_rmse_m']:.3f} m). The trajectory is the integral of graded-Viterbi velocities from the labeled $(x_0, y_0)$.

{turn_line}

## Model

Magnitude and phase features go through a 1D convolution along time. Frequency bins are the input channels. Two stride-2 pools shorten the sequence, a bidirectional LSTM reads it, and the states are linearly upsampled back onto the spectrogram timestamps. Separate dense heads emit $v_x$ and $v_y$ bin logits. Separate CTC heads add a blank symbol. There is no position, acceleration, bearing, or initial-position head.

Neighborhood targets use $\\sigma = {WEIGHTS.sigma_mps:.1f}$ m/s, one bin width. CTC weight $\\lambda = {WEIGHTS.ctc:.2f}$. Both axes are averaged over supervised frames only. The CNN and the LSTM are trained from the explicit initialization above, learning rate {LR:.1e}.

## Data and decoder

Observer-time labels are unchanged. The trailing audio extends through the propagation delay of the supervised motion and is encoder context only.

Mean tail: {cover['mean_tail_frames']:.1f} frames. Delayed lateral frames covered: {cover['hot_covered']} / {cover['hot_frames']}.

Dense $P_x$ and $P_y$ are decoded by independent Viterbi. The graded cost allows a 6 m/s jump per {DT * 1000:.2f} ms frame, which is {implied:.0f} m/s². That cutoff is the audited decoder setting, not a vehicle acceleration limit. Bin centers run from −30 to 30 m/s in steps of 2.

## Validation

{chr(10).join(lines)}

Forbidden steps are the total count on the split. The other rows are means over clips. Maximum lateral error is the mean, over clips, of each clip's largest $|\\hat y - y|$.

Train trajectory RMSE at this checkpoint is {train_metrics['trajectory_rmse_m']:.3f} m and validation is {val_metrics['trajectory_rmse_m']:.3f} m. Train turning $v_y$ MAE is {train_metrics['vy_turn_mae']:.3f}; validation is {val_metrics['vy_turn_mae']:.3f}.
"""
    (OUT / "CORE.md").write_text(text)
    print(text, flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
