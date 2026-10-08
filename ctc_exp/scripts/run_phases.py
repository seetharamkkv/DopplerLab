#!/usr/bin/env python3
"""Run CTC phases 0–2 on VS13 or the DopplerSim test_dataset.

  .venv/bin/python ctc_exp/scripts/run_phases.py --dataset sim
  .venv/bin/python ctc_exp/scripts/run_phases.py --dataset vs13
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
from torch.nn.utils.rnn import pad_sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ctc_exp" / "src"))

from ctc_exp.alphabet import (  # noqa: E402
    N_CLASS,
    ids_to_tokens,
    speed_bin_index,
    speed_token,
    tokens_to_ids,
)
from ctc_exp.audio import N_MELS, load_log_mel  # noqa: E402
from ctc_exp.catalog import Clip, load_sim, load_vs13  # noqa: E402
from ctc_exp.loss import (  # noqa: E402
    class_weights_from_counts,
    focal_balanced_ce,
    prior_collapse_penalty,
    splice_speed_token,
    train_speed_counts,
)
from ctc_exp.model import CtcKinematics, greedy_decode  # noqa: E402
from ctc_exp.paths import load_dataset_paths  # noqa: E402
from ctc_exp.retrieve import random_recall_at_k, rank_by_token_distance  # noqa: E402
from ctc_exp.tokens import (  # noqa: E402
    has_apr_cpa_rec,
    parse_min_range_token,
    parse_speed_token,
    token_error_rate,
    token_string,
)


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
            if n_test + n_val >= n:
                n_test, n_val = 1, 1
                if n < 3:
                    n_val = 0
                    n_test = max(1, n // 3)
        test.extend(idxs[:n_test])
        val.extend(idxs[n_test : n_test + n_val])
        train.extend(idxs[n_test + n_val :])
    if not train:
        train = list(range(len(clips)))
        val, test = train[:1], train[-1:]
    return {"train": train, "val": val, "test": test}


def phase0_report(clips: list[Clip]) -> dict:
    lengths = [len(c.tokens) for c in clips]
    apr = sum(1 for c in clips if has_apr_cpa_rec(c.tokens))
    vcounts = Counter(parse_speed_token(c.tokens) or "?" for c in clips)
    rcounts = Counter(parse_min_range_token(c.tokens) or "none" for c in clips)
    examples = [
        {"clip_id": c.clip_id, "vehicle": c.vehicle, "tokens": c.tokens[:24]}
        for c in clips[:8]
    ]
    return {
        "n_clips": len(clips),
        "token_len_min": int(min(lengths)),
        "token_len_max": int(max(lengths)),
        "token_len_mean": float(np.mean(lengths)),
        "frac_apr_cpa_rec": float(apr) / len(clips),
        "speed_token_hist": dict(vcounts),
        "range_token_hist": dict(rcounts),
        "examples": examples,
    }


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, clips: list[Clip], indices: list[int]):
        self.clips = clips
        self.indices = indices
        self.cache: dict[int, tuple[np.ndarray, list[int]]] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        clip = self.clips[idx]
        if idx not in self.cache:
            crop = clip.dataset == "vs13"
            mel, times = load_log_mel(
                clip.wav_path,
                clip.log_mel_path,
                cpa_s=clip.cpa_s,
                crop_around_cpa=crop,
            )
            range_m = clip.range_m
            if range_m is not None and clip.times_s is not None and len(range_m) != len(times):
                range_m = np.interp(times, clip.times_s, range_m)
            elif range_m is not None and len(range_m) != len(times):
                range_m = range_m[: len(times)]
            toks = token_string(
                speed_mps=clip.speed_mps,
                cpa_s=clip.cpa_s,
                times_s=times,
                range_m=range_m,
                include_range=clip.dataset == "sim",
            )
            self.cache[idx] = (mel, tokens_to_ids(toks))
        mel, ids = self.cache[idx]
        return {
            "mel": torch.from_numpy(mel),
            "targets": torch.tensor(ids, dtype=torch.long),
            "speed_idx": speed_bin_index(clip.speed_mps),
            "idx": idx,
        }


def collate(batch: list[dict]) -> dict:
    mels = [b["mel"] for b in batch]
    t_lens = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    max_t = int(t_lens.max())
    n_mels = mels[0].shape[0]
    padded = torch.full((len(batch), n_mels, max_t), -80.0)
    for i, m in enumerate(mels):
        padded[i, :, : m.shape[1]] = m
    targets = [b["targets"] for b in batch]
    tlen = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    packed = pad_sequence(targets, batch_first=True, padding_value=0)
    return {
        "mel": padded,
        "in_len": t_lens,
        "targets": packed,
        "tgt_len": tlen,
        "speed_idx": torch.tensor([b["speed_idx"] for b in batch], dtype=torch.long),
        "idx": [b["idx"] for b in batch],
    }


def run_epoch(
    model,
    loader,
    opt,
    device,
    train: bool,
    *,
    class_w: torch.Tensor,
    train_prior: torch.Tensor,
    lambda_speed: float = 2.0,
    lambda_collapse: float = 0.75,
) -> tuple[float, float, float, int, int]:
    model.train(train)
    ctc = nn.CTCLoss(blank=0, zero_infinity=False)
    total, n, skipped = 0.0, 0, 0
    speed_tot, col_tot = 0.0, 0.0
    for batch in loader:
        mel = batch["mel"].to(device)
        raw_len = batch["in_len"].to(device)
        speed_idx = batch["speed_idx"].to(device)
        ctc_logits, speed_logits, _xy, _vel = model(mel, raw_len)
        logp = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
        in_len = CtcKinematics.encoded_lengths(raw_len)
        max_t = logp.shape[0]
        in_len = torch.clamp(in_len, max=max_t)
        tgt = batch["targets"].to(device)
        tgt_len = batch["tgt_len"].to(device)
        valid = tgt_len <= in_len
        if not bool(valid.any()):
            skipped += len(batch["idx"])
            continue
        if not bool(valid.all()):
            skipped += int((~valid).sum().item())
            keep = valid.nonzero(as_tuple=False).view(-1)
            logp = logp[:, keep]
            in_len = in_len[keep]
            tgt = tgt[keep]
            tgt_len = tgt_len[keep]
            speed_logits = speed_logits[keep]
            speed_idx = speed_idx[keep]
        ctc_loss = ctc(logp, tgt, in_len, tgt_len)
        if not torch.isfinite(ctc_loss):
            skipped += int(tgt.size(0))
            continue
        speed_loss = focal_balanced_ce(speed_logits, speed_idx, class_w)
        collapse = prior_collapse_penalty(speed_logits, train_prior)
        loss = ctc_loss + lambda_speed * speed_loss + lambda_collapse * collapse
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        bs = int(tgt.size(0))
        total += float(loss.item()) * bs
        speed_tot += float(speed_loss.item()) * bs
        col_tot += float(collapse.item()) * bs
        n += bs
    return total / max(n, 1), speed_tot / max(n, 1), col_tot / max(n, 1), n, skipped


@torch.no_grad()
def decode_split(model, clips, indices, device) -> list[list[str]]:
    model.eval()
    out = []
    ds = ClipDataset(clips, indices)
    for i in range(len(ds)):
        item = ds[i]
        mel = item["mel"].unsqueeze(0).to(device)
        ctc_logits, speed_logits, _xy, _vel = model(mel, torch.tensor([mel.shape[-1]], device=device))
        ids = greedy_decode(ctc_logits[0])
        toks = ids_to_tokens(ids)
        v_idx = int(speed_logits[0].argmax().item())
        out.append(splice_speed_token(toks, v_idx))
    return out


def kinematics_from_decode(hyp: list[str], clip: Clip) -> dict:
    v_hyp = parse_speed_token(hyp)
    v_ref = speed_token(clip.speed_mps)
    r_hyp = parse_min_range_token(hyp)
    r_ref = parse_min_range_token(clip.tokens)
    v_ok = int(v_hyp == v_ref)
    r_ok = int(r_hyp == r_ref) if r_ref is not None else None
    v_true = clip.speed_mps
    v_pred = float(int((v_hyp or v_ref)[1:])) if (v_hyp or v_ref) else v_true
    return {
        "v_ref": v_ref,
        "v_hyp": v_hyp,
        "v_ok": v_ok,
        "r_ref": r_ref,
        "r_hyp": r_hyp,
        "r_ok": r_ok,
        "speed_mae_mps": abs(v_pred - v_true),
        "ter": token_error_rate(clip.tokens, hyp),
    }


def majority_speed(clips: list[Clip], train_idx: list[int]) -> str:
    vs = [speed_token(clips[i].speed_mps) for i in train_idx]
    return Counter(vs).most_common(1)[0][0]


def retrieval_metrics(clips, test_idx, hyps, k: int = 10) -> dict:
    gallery_tok = hyps
    n = len(test_idx)
    if n < 2:
        return {"n": n}
    rec_speed = rec_range = rec_joint = 0
    for q in range(n):
        order = rank_by_token_distance(gallery_tok[q], gallery_tok)
        order = [j for j in order if j != q]
        top = order[:k]
        q_clip = clips[test_idx[q]]
        qv = speed_token(q_clip.speed_mps)
        qr = parse_min_range_token(q_clip.tokens)
        def match_v(j):
            return speed_token(clips[test_idx[j]].speed_mps) == qv

        def match_r(j):
            return parse_min_range_token(clips[test_idx[j]].tokens) == qr

        if any(match_v(j) for j in top):
            rec_speed += 1
        if qr is not None and any(match_r(j) for j in top):
            rec_range += 1
        if qr is not None and any(match_v(j) and match_r(j) for j in top):
            rec_joint += 1
    return {
        "k": k,
        "n_queries": n,
        "recall_speed": rec_speed / n,
        "recall_range": rec_range / n if any(parse_min_range_token(clips[i].tokens) for i in test_idx) else None,
        "recall_joint": rec_joint / n if any(parse_min_range_token(clips[i].tokens) for i in test_idx) else None,
        "random_recall": random_recall_at_k(n - 1, k),
    }


def run_dataset(name: str, clips: list[Clip], out_dir: Path, epochs: int, batch_size: int, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    p0 = phase0_report(clips)
    (out_dir / "phase0.json").write_text(json.dumps(p0, indent=2), encoding="utf-8")
    splits = split_clips(clips, seed=seed)
    (out_dir / "splits.json").write_text(
        json.dumps({k: [clips[i].clip_id for i in v] for k, v in splits.items()}, indent=2),
        encoding="utf-8",
    )

    print(
        f"[{name}] phase0 n={p0['n_clips']} token_len={p0['token_len_min']}-{p0['token_len_max']} "
        f"mean={p0['token_len_mean']:.1f} apr_cpa_rec={p0['frac_apr_cpa_rec']:.2f}",
        flush=True,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    counts = train_speed_counts(clips, splits["train"], device)
    class_w = class_weights_from_counts(counts)
    train_prior = counts / counts.sum().clamp(min=1)
    model = CtcKinematics(n_mels=N_MELS, n_class=N_CLASS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_ds = ClipDataset(clips, splits["train"])
    val_ds = ClipDataset(clips, splits["val"])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )

    history = []
    best = float("inf")
    ckpt = out_dir / "best.pt"
    last_state = None
    for epoch in range(1, epochs + 1):
        tr, tr_sp, tr_col, n_tr, skip_tr = run_epoch(
            model, train_loader, opt, device, True, class_w=class_w, train_prior=train_prior
        )
        va, va_sp, va_col, n_va, skip_va = run_epoch(
            model, val_loader, opt, device, False, class_w=class_w, train_prior=train_prior
        )
        last_state = model.state_dict()
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr,
                "val_loss": va,
                "train_speed_ce": tr_sp,
                "val_speed_ce": va_sp,
                "train_collapse": tr_col,
                "val_collapse": va_col,
                "train_n": n_tr,
                "val_n": n_va,
                "train_skipped": skip_tr,
                "val_skipped": skip_va,
            }
        )
        print(
            f"[{name}] epoch {epoch}/{epochs} loss={tr:.4f}/{va:.4f} "
            f"speed={tr_sp:.3f}/{va_sp:.3f} collapse={tr_col:.3f}/{va_col:.3f} "
            f"n={n_tr}/{n_va}",
            flush=True,
        )
        if n_va > 0 and va < best:
            best = va
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
    if not ckpt.is_file() and last_state is not None:
        torch.save({"model": last_state, "epoch": epochs}, ckpt)

    payload = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    test_idx = splits["test"]
    hyps = decode_split(model, clips, test_idx, device)
    rows = []
    for i, hyp in zip(test_idx, hyps):
        rec = kinematics_from_decode(hyp, clips[i])
        rec["clip_id"] = clips[i].clip_id
        rec["hyp"] = hyp
        rec["ref"] = clips[i].tokens
        rows.append(rec)
    maj = majority_speed(clips, splits["train"])
    v_acc = float(np.mean([r["v_ok"] for r in rows]))
    maj_acc = float(np.mean([int(speed_token(clips[i].speed_mps) == maj) for i in test_idx]))
    ter = float(np.mean([r["ter"] for r in rows]))
    mae = float(np.mean([r["speed_mae_mps"] for r in rows]))
    r_ok_vals = [r["r_ok"] for r in rows if r["r_ok"] is not None]
    pred_hist = Counter(r["v_hyp"] or "?" for r in rows)
    true_hist = Counter(r["v_ref"] for r in rows)
    p1 = {
        "device": str(device),
        "epochs": epochs,
        "best_val_loss": best,
        "history": history,
        "test_n": len(test_idx),
        "token_error_rate": ter,
        "speed_bin_acc": v_acc,
        "speed_mae_mps": mae,
        "majority_speed_bin_acc": maj_acc,
        "majority_token": maj,
        "pred_speed_hist": dict(pred_hist),
        "true_speed_hist": dict(true_hist),
        "range_bin_acc": float(np.mean(r_ok_vals)) if r_ok_vals else None,
        "examples": rows[:12],
    }
    (out_dir / "phase1.json").write_text(json.dumps(p1, indent=2), encoding="utf-8")

    p2 = retrieval_metrics(clips, test_idx, hyps, k=10)
    gt_tok = [clips[i].tokens for i in test_idx]
    p2["oracle_gt_strings"] = retrieval_metrics(clips, test_idx, gt_tok, k=10)
    (out_dir / "phase2.json").write_text(json.dumps(p2, indent=2), encoding="utf-8")

    summary = {"dataset": name, "phase0": p0, "phase1": {k: v for k, v in p1.items() if k != "examples" and k != "history"}, "phase2": p2}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("vs13", "sim", "both"), default="both")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    paths = load_dataset_paths()
    out_root = REPO / "ctc_exp" / "outputs"
    summaries = {}
    if args.dataset in ("vs13", "both"):
        clips = load_vs13(paths["vs13"])
        summaries["vs13"] = run_dataset("vs13", clips, out_root / "vs13", args.epochs, args.batch_size, args.seed)
    if args.dataset in ("sim", "both"):
        clips = load_sim(paths["dopplersim_test_dataset"])
        summaries["sim"] = run_dataset("sim", clips, out_root / "sim", args.epochs, args.batch_size, args.seed)
    print(json.dumps({k: v for k, v in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
