"""Smoke training for the motion CTC on a small DopplerSim batch.

The batch is a few hundred to about a thousand free paths. Two compensations
keep the CTC on the motion instead of the vehicle timbre:

- Validation and test vehicles are held out entirely.
- Each epoch draws several random time crops per clip and retokenizes that
  crop, so the target is the velocity change inside the window.

Run from the DopplerLab root:

    position_ctc/.venv/bin/python position_ctc/scripts/train.py \\
        --data /path/to/audio_clips
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "position_ctc" / "src"))

from position_ctc.alphabet import TOKEN_TO_ID  # noqa: E402
from position_ctc.clock import DT  # noqa: E402
from position_ctc.paths import resolve_audio_clips  # noqa: E402
from position_ctc.data import (  # noqa: E402
    CROP_FRAMES,
    augment_features,
    crop_pair,
    discover_clips,
    load_arrays,
    split_by_vehicle,
    standardize_mag,
)
from position_ctc.decode import greedy_decode, initial_pair, token_error_rate  # noqa: E402
from position_ctc.features import DATA_HOP, DATA_SR  # noqa: E402
from position_ctc.model import MotionCTC  # noqa: E402
from position_ctc.tokens import motion_tokens  # noqa: E402

FRAME_DT = DATA_HOP / DATA_SR


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ctc_ok(device: torch.device) -> bool:
    try:
        loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        logits = torch.zeros(8, 1, 4, device=device)
        logits[0, 0, 1] = 5.0
        targets = torch.tensor([1], device=device)
        in_len = torch.tensor([8], device=device)
        tgt_len = torch.tensor([1], device=device)
        value = loss_fn(logits.log_softmax(0), targets, in_len, tgt_len)
        return bool(torch.isfinite(value).item())
    except Exception:
        return False


def _prepare(clips, indices: list[int]) -> list[dict]:
    rows = []
    for i in indices:
        clip = clips[i]
        feat, state = load_arrays(clip)
        rows.append({"clip_id": clip.clip_id, "vehicle": clip.vehicle, "feat": feat, "state": state})
        if len(rows) % 100 == 0:
            print(f"  features {len(rows)}/{len(indices)}", flush=True)
    return rows


def _limit_tokens(tokens: list[str], max_tokens: int) -> list[str]:
    """Keep the initial velocity pair and the earliest acceleration events.

    Full free-path strings run past 30 tokens. On a few hundred clips the CTC
    then emits only the majority speed and ignores the turn. A short prefix is
    the motion the small run can actually fit.
    """
    if max_tokens < 2:
        raise ValueError("max_tokens")
    return tokens[:max_tokens]


def _example(
    row: dict,
    rng: np.random.Generator,
    *,
    train: bool,
    crop: int,
    max_tokens: int,
) -> dict | None:
    feat, state = row["feat"], row["state"]
    if train:
        feat = augment_features(feat, rng)
        feat, state = crop_pair(feat, state, rng, crop)
    feat = standardize_mag(feat)
    if state.shape[0] < 8:
        return None
    tokens = _limit_tokens(motion_tokens(state, FRAME_DT).tokens, max_tokens)
    ids = [TOKEN_TO_ID[tok] for tok in tokens if tok in TOKEN_TO_ID]
    enc_len = max(1, feat.shape[1] // 4)
    if len(ids) < 2 or len(ids) > enc_len:
        return None
    return {
        "feat": torch.from_numpy(np.ascontiguousarray(feat)),
        "ids": torch.tensor(ids, dtype=torch.long),
        "tokens": tokens,
        "clip_id": row["clip_id"],
    }


def _collate(items: list[dict]) -> dict:
    feats = [item["feat"] for item in items]
    lengths = torch.tensor([feat.shape[1] for feat in feats], dtype=torch.long)
    max_t = int(lengths.max())
    n_ch = feats[0].shape[0]
    padded = torch.zeros(len(items), n_ch, max_t)
    for i, feat in enumerate(feats):
        padded[i, :, : feat.shape[1]] = feat
    ids = [item["ids"] for item in items]
    return {
        "feat": padded,
        "in_len": lengths,
        "targets": torch.cat(ids),
        "tgt_len": torch.tensor([len(item) for item in ids], dtype=torch.long),
        "tokens": [item["tokens"] for item in items],
        "clip_id": [item["clip_id"] for item in items],
    }


def _batch_loss(model, batch, device, ctc_loss) -> torch.Tensor | None:
    feat = batch["feat"].to(device)
    logits = model(feat)
    in_len = MotionCTC.encoded_lengths(batch["in_len"].to(device))
    in_len = torch.clamp(in_len, max=logits.size(1))
    tgt_len = batch["tgt_len"].to(device)
    keep = tgt_len <= in_len
    if not bool(keep.any()):
        return None
    if not bool(keep.all()):
        idx = keep.nonzero(as_tuple=False).view(-1)
        logits = logits[idx]
        in_len = in_len[idx]
        tgt_len = tgt_len[idx]
        pieces = []
        offset = 0
        full_len = batch["tgt_len"].tolist()
        targets = batch["targets"]
        for i, length in enumerate(full_len):
            if bool(keep[i]):
                pieces.append(targets[offset : offset + length])
            offset += length
        targets = torch.cat(pieces)
    else:
        targets = batch["targets"]
    logp = logits.log_softmax(-1).transpose(0, 1)
    return ctc_loss(logp, targets.to(device), in_len, tgt_len)


def _balanced_order(rows: list[dict], rng: random.Random) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        pair = initial_pair(motion_tokens(row["state"], FRAME_DT).tokens)
        if pair[0] is None or pair[1] is None:
            continue
        groups.setdefault(pair, []).append(row)
    if not groups:
        return list(rows)
    keys = list(groups)
    order: list[dict] = []
    while len(order) < len(rows):
        rng.shuffle(keys)
        for key in keys:
            order.append(rng.choice(groups[key]))
            if len(order) >= len(rows):
                break
    return order


def _evaluate(model, rows, device, ctc_loss, train_pairs: Counter, max_tokens: int) -> dict:
    model.eval()
    majority = train_pairs.most_common(1)[0][0]
    ters, pair_ok, maj_ok, vx_ok, vy_ok = [], [], [], [], []
    examples = []
    total_loss, n_loss = 0.0, 0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for row in rows:
            item = _example(row, rng, train=False, crop=CROP_FRAMES, max_tokens=max_tokens)
            if item is None:
                continue
            batch = _collate([item])
            loss = _batch_loss(model, batch, device, ctc_loss)
            if loss is not None and torch.isfinite(loss):
                total_loss += float(loss.item())
                n_loss += 1
            logits = model(batch["feat"].to(device))[0]
            hyp = greedy_decode(logits)
            ref = item["tokens"]
            ters.append(token_error_rate(ref, hyp))
            ref_pair = initial_pair(ref)
            hyp_pair = initial_pair(hyp)
            pair_ok.append(int(hyp_pair == ref_pair and ref_pair[0] is not None))
            maj_ok.append(int(majority == ref_pair))
            vx_ok.append(int(hyp_pair[0] is not None and hyp_pair[0] == ref_pair[0]))
            vy_ok.append(int(hyp_pair[1] is not None and hyp_pair[1] == ref_pair[1]))
            if len(examples) < 6:
                examples.append({"clip_id": item["clip_id"], "ref": ref[:8], "hyp": hyp[:8]})
    return {
        "loss": total_loss / max(n_loss, 1),
        "ter": float(np.mean(ters)) if ters else None,
        "pair_acc": float(np.mean(pair_ok)) if pair_ok else None,
        "vx_acc": float(np.mean(vx_ok)) if vx_ok else None,
        "vy_acc": float(np.mean(vy_ok)) if vy_ok else None,
        "majority_pair_acc": float(np.mean(maj_ok)) if maj_ok else None,
        "majority_pair": list(majority),
        "n": len(ters),
        "examples": examples,
    }


def _train_pairs(rows, crop: int) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        tokens = motion_tokens(row["state"], FRAME_DT).tokens
        pair = initial_pair(tokens)
        if pair[0] is not None:
            counts[pair] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="audio_clips directory. Default: env DOPPLER_AUDIO_CLIPS.",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--pair-epochs", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-frames", type=int, default=CROP_FRAMES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = ROOT / "position_ctc" / "outputs" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = discover_clips(resolve_audio_clips(args.data))
    splits = split_by_vehicle(clips, seed=args.seed)
    vehicles = {
        name: sorted({clips[i].vehicle for i in idxs}) for name, idxs in splits.items()
    }
    print(
        f"clips={len(clips)} train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])}",
        flush=True,
    )
    print(f"vehicles {json.dumps(vehicles)}", flush=True)
    (out_dir / "splits.json").write_text(
        json.dumps(
            {
                "vehicles": vehicles,
                "ids": {name: [clips[i].clip_id for i in idxs] for name, idxs in splits.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("extracting features", flush=True)
    rows = {name: _prepare(clips, idxs) for name, idxs in splits.items()}
    train_pairs = _train_pairs(rows["train"], args.crop_frames)
    print(f"train initial pairs {train_pairs.most_common(6)}", flush=True)

    device = _device()
    if device.type == "mps" and not _ctc_ok(device):
        print("MPS CTC is unavailable, using CPU", flush=True)
        device = torch.device("cpu")
    elif device.type == "cuda" and not _ctc_ok(device):
        device = torch.device("cpu")
    print(f"device {device} frame_dt {FRAME_DT:.6f} label_clock {DT:.6f}", flush=True)

    model = MotionCTC().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    best = float("inf")
    stale = 0
    history = []
    ckpt = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        max_tokens = 2 if epoch <= args.pair_epochs else args.max_tokens
        model.train()
        rng_py = random.Random(args.seed + epoch)
        np_rng = np.random.default_rng(args.seed + epoch)
        order: list[dict] = []
        for _ in range(args.repeats):
            order.extend(_balanced_order(rows["train"], rng_py))
        rng_py.shuffle(order)
        total, seen, skipped = 0.0, 0, 0
        for start in range(0, len(order), args.batch_size):
            items = []
            for row in order[start : start + args.batch_size]:
                item = _example(
                    row, np_rng, train=True, crop=args.crop_frames, max_tokens=max_tokens
                )
                if item is None:
                    skipped += 1
                else:
                    items.append(item)
            if len(items) < 2:
                continue
            batch = _collate(items)
            opt.zero_grad(set_to_none=True)
            loss = _batch_loss(model, batch, device, ctc_loss)
            if loss is None or not torch.isfinite(loss):
                skipped += len(items)
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(items)
            seen += len(items)
        train_loss = total / max(seen, 1)
        val = _evaluate(model, rows["val"], device, ctc_loss, train_pairs, max_tokens)
        history.append(
            {
                "epoch": epoch,
                "max_tokens": max_tokens,
                "train_loss": train_loss,
                "val": {k: v for k, v in val.items() if k != "examples"},
            }
        )
        print(
            f"epoch {epoch}/{args.epochs} tokens={max_tokens} train={train_loss:.3f} "
            f"val={val['loss']:.3f} ter={val['ter']:.3f} pair={val['pair_acc']:.3f} "
            f"vx={val['vx_acc']:.3f} vy={val['vy_acc']:.3f} "
            f"majority={val['majority_pair_acc']:.3f} skipped={skipped}",
            flush=True,
        )
        for ex in val["examples"][:2]:
            print(f"  {ex['clip_id']} ref={ex['ref']} hyp={ex['hyp']}", flush=True)
        score = val["ter"] if val["ter"] is not None else 1.0
        # Once acceleration tokens are in the target, do not keep the pair-only checkpoint.
        if epoch == args.pair_epochs + 1:
            best = score
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_ter": best}, ckpt)
        elif score + 1e-4 < best:
            best = score
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_ter": best}, ckpt)
        else:
            stale += 1
            if epoch > args.pair_epochs and stale >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    if ckpt.is_file():
        payload = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
    test = _evaluate(model, rows["test"], device, ctc_loss, train_pairs, args.max_tokens)
    summary = {
        "best_val_ter": best,
        "frame_dt_s": FRAME_DT,
        "crop_frames": args.crop_frames,
        "repeats": args.repeats,
        "test": {k: v for k, v in test.items()},
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in test.items() if k != "examples"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
