#!/usr/bin/env python3
"""Unconstrained-path tracking: speed bins + continuous xy and velocity.

  .venv/bin/python ctc_exp/scripts/run_track.py --epochs 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ctc_exp" / "src"))

from ctc_exp.alphabet import N_CLASS, speed_bin_index, speed_token  # noqa: E402
from ctc_exp.audio import N_MELS, load_log_mel  # noqa: E402
from ctc_exp.catalog import Clip, load_straight_audio_only, load_unconstrained  # noqa: E402
from ctc_exp.loss import (  # noqa: E402
    bin_conditioned_speed_residual,
    centered_traj_huber_m,
    class_weights_from_counts,
    consistency_huber,
    focal_balanced_ce,
    prior_collapse_penalty,
    train_speed_counts,
)
from ctc_exp.model import CtcKinematics  # noqa: E402
from ctc_exp.paths import load_dataset_paths  # noqa: E402
from ctc_exp.track_geom import downsample_series, orbit_rms, straight_passby_xy  # noqa: E402


def split_clips(clips: list[Clip], seed: int = 0) -> dict[str, list[int]]:
    rng = random.Random(seed)
    by_veh: dict[str, list[int]] = {}
    for i, c in enumerate(clips):
        by_veh.setdefault(c.vehicle, []).append(i)
    train, val, test = [], [], []
    for idxs in by_veh.values():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(0.15 * n)))
        n_val = max(1, int(round(0.15 * n)))
        if n_test + n_val >= n:
            n_test = max(1, n // 5)
            n_val = max(1, n // 5)
        test.extend(idxs[:n_test])
        val.extend(idxs[n_test : n_test + n_val])
        train.extend(idxs[n_test + n_val :])
    splits = {"train": train, "val": val, "test": test}
    assert_disjoint_splits(clips, splits)
    return splits


def assert_disjoint_splits(clips: list[Clip], splits: dict[str, list[int]]) -> None:
    sets = {k: {clips[i].clip_id for i in idxs} for k, idxs in splits.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise RuntimeError("clip_id leakage across train/val/test")
    n = sum(len(v) for v in sets.values())
    if n != len({c for s in sets.values() for c in s}):
        raise RuntimeError("duplicate clip_id inside a split")


class TrackDataset(torch.utils.data.Dataset):
    def __init__(self, clips: list[Clip], indices: list[int], *, derive_straight: bool):
        self.clips = clips
        self.indices = indices
        self.derive_straight = derive_straight
        self.cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        clip = self.clips[idx]
        if idx not in self.cache:
            mel, times = load_log_mel(clip.wav_path, clip.log_mel_path)
            t = mel.shape[1]
            if self.derive_straight:
                if clip.cpa_m is None:
                    raise ValueError(f"{clip.clip_id} missing CPA distance")
                xy, vel = straight_passby_xy(
                    times,
                    speed_mps=clip.speed_mps,
                    cpa_s=clip.cpa_s,
                    cpa_m=clip.cpa_m,
                )
            else:
                xy = downsample_series(clip.xy_m, t)
                vel = downsample_series(clip.vel_mps, t)
            self.cache[idx] = {
                "mel": mel,
                "xy": xy.astype(np.float32),
                "vel": vel.astype(np.float32),
                "dt": float((times[-1] - times[0]) / max(t - 1, 1)),
            }
        item = self.cache[idx]
        return {
            "mel": torch.from_numpy(item["mel"]),
            "xy": torch.from_numpy(item["xy"]),
            "vel": torch.from_numpy(item["vel"]),
            "dt": item["dt"],
            "speed_idx": speed_bin_index(clip.speed_mps),
            "idx": idx,
        }


def collate(batch: list[dict]) -> dict:
    mels = [b["mel"] for b in batch]
    t_lens = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    max_t = int(t_lens.max())
    n_mels = mels[0].shape[0]
    padded = torch.full((len(batch), n_mels, max_t), -80.0)
    xy = torch.zeros(len(batch), max_t, 2)
    vel = torch.zeros(len(batch), max_t, 2)
    for i, b in enumerate(batch):
        t = b["mel"].shape[1]
        padded[i, :, :t] = b["mel"]
        xy[i, :t] = b["xy"]
        vel[i, :t] = b["vel"]
    return {
        "mel": padded,
        "in_len": t_lens,
        "xy": xy,
        "vel": vel,
        "dt": torch.tensor([b["dt"] for b in batch], dtype=torch.float32),
        "speed_idx": torch.tensor([b["speed_idx"] for b in batch], dtype=torch.long),
        "idx": [b["idx"] for b in batch],
    }


def _resample_bt(x: torch.Tensor, n_out: int) -> torch.Tensor:
    y = F.interpolate(x.transpose(1, 2), size=n_out, mode="linear", align_corners=True)
    return y.transpose(1, 2)


def run_epoch(model, loader, opt, device, train, class_w, train_prior, *, use_xy: bool) -> dict:
    model.train(train)
    tot = {k: 0.0 for k in ("loss", "speed", "xy", "vel_mag", "vel_dir", "cons")}
    n = 0
    for batch in loader:
        mel = batch["mel"].to(device)
        raw_len = batch["in_len"].to(device)
        speed_idx = batch["speed_idx"].to(device)
        # Train: condition velocity on GT bins. Eval: on predicted bins.
        cond = speed_idx if train else None
        _ctc, speed_logits, pred_xy, pred_v = model(mel, raw_len, speed_idx=cond)
        enc_len = CtcKinematics.encoded_lengths(raw_len)
        t_enc = pred_v.size(1)
        vel_gt = _resample_bt(batch["vel"].to(device), t_enc)
        xy_gt = _resample_bt(batch["xy"].to(device), t_enc)
        ar = torch.arange(t_enc, device=device).unsqueeze(0)
        mask = (ar < enc_len.unsqueeze(1)).float()
        speed_loss = focal_balanced_ce(speed_logits, speed_idx, class_w)
        collapse = prior_collapse_penalty(speed_logits, train_prior)
        mag_loss, dir_loss = bin_conditioned_speed_residual(
            pred_v, vel_gt, speed_idx, model.bin_centers, mask
        )
        xy_loss = centered_traj_huber_m(pred_xy, xy_gt, mask)
        cons = consistency_huber(pred_xy, pred_v, batch["dt"].to(device), mask)
        loss = 2.0 * speed_loss + 0.75 * collapse + 4.0 * mag_loss + 2.0 * dir_loss
        if use_xy:
            loss = loss + 1.0 * xy_loss + 0.25 * cons
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        bs = len(batch["idx"])
        tot["loss"] += float(loss.item()) * bs
        tot["speed"] += float(speed_loss.item()) * bs
        tot["vel_mag"] += float(mag_loss.item()) * bs
        tot["xy"] += float(xy_loss.item()) * bs
        tot["vel_dir"] += float(dir_loss.item()) * bs
        tot["cons"] += float(cons.item()) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in tot.items()} | {"n": n}


@torch.no_grad()
def eval_test(model, clips, indices, device, *, derive_straight: bool) -> dict:
    model.eval()
    ds = TrackDataset(clips, indices, derive_straight=derive_straight)
    rows = []
    for i in range(len(ds)):
        item = ds[i]
        clip = clips[item["idx"]]
        mel = item["mel"].unsqueeze(0).to(device)
        _ctc, speed_logits, pred_xy, pred_v = model(
            mel, torch.tensor([mel.shape[-1]], device=device), speed_idx=None
        )
        t_enc = pred_v.size(1)
        pred_xy = pred_xy[0].cpu().numpy()
        pred_v = pred_v[0].cpu().numpy()
        gt_xy = downsample_series(item["xy"].numpy(), t_enc)
        gt_v = downsample_series(item["vel"].numpy(), t_enc)
        from ctc_exp.alphabet import SPEED_TOKENS

        v_hyp = SPEED_TOKENS[int(speed_logits[0].argmax())]
        v_ref = speed_token(clip.speed_mps)
        n = min(len(pred_xy), len(gt_xy))
        pxy, gxy = pred_xy[:n], gt_xy[:n]
        px = pxy[:, 0] - pxy[:, 0].mean()
        gx = gxy[:, 0] - gxy[:, 0].mean()
        rows.append(
            {
                "clip_id": clip.clip_id,
                "v_ref": v_ref,
                "v_hyp": v_hyp,
                "v_ok": int(v_hyp == v_ref),
                "orbit_rms_m": orbit_rms(pxy, gxy),
                "along_x_mae_m": float(np.mean(np.abs(px - gx))),
                "cpa_y_mae_m": float(np.mean(np.abs(pxy[:, 1] - gxy[:, 1]))),
                "vel_mae": float(np.mean(np.linalg.norm(pred_v[:n] - gt_v[:n], axis=1))),
                "speed_mag_mae": float(
                    np.mean(
                        np.abs(
                            np.linalg.norm(pred_v[:n], axis=1) - np.linalg.norm(gt_v[:n], axis=1)
                        )
                    )
                ),
            }
        )
    v_acc = float(np.mean([r["v_ok"] for r in rows]))
    maj = Counter(speed_token(clips[j].speed_mps) for j in indices)
    # majority from train computed outside
    return {
        "test_n": len(rows),
        "speed_bin_acc": v_acc,
        "orbit_rms_m": float(np.mean([r["orbit_rms_m"] for r in rows])),
        "along_x_mae_m": float(np.mean([r["along_x_mae_m"] for r in rows])),
        "cpa_y_mae_m": float(np.mean([r["cpa_y_mae_m"] for r in rows])),
        "vel_mae_mps": float(np.mean([r["vel_mae"] for r in rows])),
        "speed_mag_mae_mps": float(np.mean([r["speed_mag_mae"] for r in rows])),
        "pred_speed_hist": dict(Counter(r["v_hyp"] for r in rows)),
        "true_speed_hist": dict(Counter(r["v_ref"] for r in rows)),
        "examples": rows[:8],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--source",
        choices=("unconstrained", "straight"),
        default="unconstrained",
    )
    args = p.parse_args()
    paths = load_dataset_paths()
    derive_straight = args.source == "straight"
    use_xy = derive_straight
    if derive_straight:
        clips = load_straight_audio_only(paths["dopplersim_test_dataset"])
        tag = "straight"
        out_dir = REPO / "ctc_exp" / "outputs" / "straight_pos"
    else:
        clips = load_unconstrained(paths["dopplersim_unconstrained"])
        tag = "unconstrained"
        out_dir = REPO / "ctc_exp" / "outputs" / "unconstrained_velcond"
    splits = split_clips(clips, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_ids = {k: [clips[i].clip_id for i in idxs] for k, idxs in splits.items()}
    (out_dir / "splits.json").write_text(json.dumps(split_ids, indent=2), encoding="utf-8")
    print(
        f"[{tag}] n={len(clips)} train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])} "
        f"derive_straight={derive_straight} use_xy={use_xy} patience={args.patience}",
        flush=True,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    counts = train_speed_counts(clips, splits["train"], device)
    class_w = class_weights_from_counts(counts)
    prior = counts / counts.sum().clamp(min=1)
    model = CtcKinematics(n_mels=N_MELS, n_class=N_CLASS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = torch.utils.data.DataLoader(
        TrackDataset(clips, splits["train"], derive_straight=derive_straight),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        TrackDataset(clips, splits["val"], derive_straight=derive_straight),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    best = float("inf")
    ckpt = out_dir / "best.pt"
    history = []
    stale = 0
    stopped_at = args.epochs
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, opt, device, True, class_w, prior, use_xy=use_xy)
        va = run_epoch(model, val_loader, opt, device, False, class_w, prior, use_xy=use_xy)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"[{tag}] epoch {epoch}/{args.epochs} "
            f"loss={tr['loss']:.4f}/{va['loss']:.4f} "
            f"speed={tr['speed']:.3f}/{va['speed']:.3f} "
            f"xy={tr['xy']:.3f}/{va['xy']:.3f} "
            f"vmag={tr['vel_mag']:.3f}/{va['vel_mag']:.3f} "
            f"vdir={tr['vel_dir']:.3f}/{va['vel_dir']:.3f}",
            flush=True,
        )
        if va["loss"] + 1e-4 < best:
            best = va["loss"]
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        else:
            stale += 1
            if stale >= args.patience:
                stopped_at = epoch
                print(f"[{tag}] early stop at epoch {epoch} (patience={args.patience})", flush=True)
                break
    payload = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    maj = Counter(speed_token(clips[i].speed_mps) for i in splits["train"]).most_common(1)[0][0]
    test = eval_test(model, clips, splits["test"], device, derive_straight=derive_straight)
    test["majority_speed_bin_acc"] = float(
        np.mean([int(speed_token(clips[i].speed_mps) == maj) for i in splits["test"]])
    )
    test["majority_token"] = maj
    test["best_val_loss"] = best
    test["epochs"] = args.epochs
    test["stopped_at"] = stopped_at
    test["best_epoch"] = int(payload["epoch"])
    test["label_source"] = (
        "pythagoras_from_speed_cpa_wav_times"
        if derive_straight
        else "state_frames"
    )
    test["audio_only"] = True
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in test.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
