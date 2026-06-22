"""Training loop and checkpoint I/O for direction CNN."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from idmt_experiments.config import (
    DEFAULT_CHECKPOINT_DIR,
    DirectionConfig,
    NormStats,
    checkpoint_subdir,
    resolve_class_labels,
)
from idmt_experiments.src.direction.dataset import collate_batch, make_dataset, precompute_batch
from idmt_experiments.src.direction.metrics import classification_metrics
from idmt_experiments.src.direction.model import DirectionCNN
from idmt_experiments.src.features import fit_norm_stats
from idmt_experiments.src.preprocess import ClipRecord, filter_records
from idmt_experiments.src.splits import (
    _sanitize_location,
    build_location_loo_splits,
    build_split,
    default_split_meta_path,
    persist_split_meta,
    verify_no_event_leakage,
)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Direction training requires PyTorch. Install: pip install -r IDMT_experiments/requirements.txt"
        ) from exc
    return torch, nn, DataLoader


def resolve_device(device: str = "auto") -> str:
    torch, _, _ = _require_torch()
    cuda_ok = torch.cuda.is_available()
    if device == "auto":
        return "cuda" if cuda_ok else "cpu"
    if device.startswith("cuda") and not cuda_ok:
        print("WARNING: CUDA unavailable — falling back to CPU.")
        return "cpu"
    return device


def save_checkpoint(
    path: Path,
    model,
    cfg: DirectionConfig,
    *,
    norm_stats: NormStats | None = None,
    extra: dict | None = None,
) -> None:
    torch, _, _ = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "task": cfg.task,
        "norm_stats": norm_stats.to_dict() if norm_stats else None,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu"):
    torch, _, _ = _require_torch()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = DirectionConfig.from_dict(ckpt.get("config", ckpt.get("cfg", {})))
    model = DirectionCNN.build(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    norm_stats = NormStats.from_dict(ckpt.get("norm_stats"))
    return model, cfg, norm_stats, ckpt


def _run_epoch(model, loader, optim, loss_fn, device, train: bool, *, cfg: DirectionConfig):
    torch, _, _ = _require_torch()
    model.train() if train else model.eval()

    losses: list[float] = []
    preds: list[int] = []
    truths: list[int] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _meta in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()

            losses.append(float(loss.item()))
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            truths.extend(y.cpu().numpy().tolist())

    labels = resolve_class_labels(cfg)
    metrics = classification_metrics(np.array(truths), np.array(preds), labels=labels)
    return float(np.mean(losses)), metrics, truths, preds


def train_on_records(
    train_records: list[ClipRecord],
    val_records: list[ClipRecord],
    cfg: DirectionConfig,
    device: str,
    checkpoint_path: Path,
    *,
    split_meta: dict | None = None,
) -> dict:
    torch, nn, DataLoader = _require_torch()
    device = resolve_device(device)

    train_records = filter_records(train_records, cfg)
    val_records = filter_records(val_records, cfg)

    audit = verify_no_event_leakage(train_records, val_records, [])
    if not audit["ok"]:
        raise RuntimeError(f"Train/valid event leakage detected: {audit}")

    feat_tag = cfg.feature_type.upper()
    print("  Fitting normalization stats on train clips only...", flush=True)
    n_norm = cfg.norm_fit_max_samples or len(train_records)
    print(f"    ({n_norm} clips — CC is slower than mel, ~1–3 min)", flush=True)
    norm_stats = fit_norm_stats(
        train_records, cfg, max_samples=cfg.norm_fit_max_samples, show_progress=True
    )

    print("  Precomputing train features...", flush=True)
    cc_note = "expect ~15–30 min on CPU for CC" if cfg.feature_type == "cc" else "mel is fast (~1 min)"
    print(f"    ({len(train_records)} clips — {cc_note})", flush=True)
    train_items = precompute_batch(
        train_records, cfg, norm_stats, show_progress=True, desc=f"train {feat_tag}"
    )
    print("  Precomputing val features...", flush=True)
    print(f"    ({len(val_records)} clips)", flush=True)
    val_items = precompute_batch(val_records, cfg, norm_stats, show_progress=True, desc=f"val {feat_tag}")

    train_ds = make_dataset(train_records, cfg, norm_stats, precomputed=train_items)
    val_ds = make_dataset(val_records, cfg, norm_stats, precomputed=val_items)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0
    )

    model = DirectionCNN.build(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state: dict | None = None
    patience_left = cfg.patience
    history: list[dict] = []

    print(f"  preempt={cfg.preempt}  task={cfg.task}  feature={cfg.feature_type}  n_classes={cfg.n_classes}")
    print(f"  train={len(train_records)}  val={len(val_records)}  device={device}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_m, _, _ = _run_epoch(
            model, train_loader, optim, loss_fn, device, train=True, cfg=cfg
        )
        val_loss, val_m, _, _ = _run_epoch(
            model, val_loader, optim, loss_fn, device, train=False, cfg=cfg
        )
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_m["accuracy"],
                "val_loss": val_loss,
                "val_acc": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
            }
        )
        print(
            f"  epoch {epoch:3d}  train_acc={train_m['accuracy']:.4f}  "
            f"val_acc={val_m['accuracy']:.4f}  val_f1={val_m['macro_f1']:.4f}"
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_m["accuracy"]
            best_epoch = epoch
            patience_left = cfg.patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(
                checkpoint_path,
                model,
                cfg,
                norm_stats=norm_stats,
                extra={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_m["accuracy"],
                    "val_metrics": val_m,
                    "n_train": len(train_records),
                    "n_val": len(val_records),
                    "split_meta": split_meta,
                    "fold_complete": False,
                },
            )
        elif cfg.preempt:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop @ epoch {epoch} (best val_acc={best_val_acc:.4f} @ {best_epoch})")
                break

    final_epoch = history[-1]["epoch"] if history else 0
    (checkpoint_path.parent / f"{checkpoint_path.stem}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(
            checkpoint_path,
            model,
            cfg,
            norm_stats=norm_stats,
            extra={
                "epoch": best_epoch,
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "final_epoch": final_epoch,
                "epochs_configured": cfg.epochs,
                "preempt": cfg.preempt,
                "fold_complete": True,
                "n_train": len(train_records),
                "n_val": len(val_records),
                "split_meta": split_meta,
            },
        )

    summary_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.summary.json")
    summary = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_epoch": final_epoch,
        "preempt": cfg.preempt,
        "epochs_configured": cfg.epochs,
        "fold_complete": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
    }


def train_split(
    data_dir=None,
    checkpoint_dir: Path | None = None,
    run_name: str | None = None,
    cfg: DirectionConfig | None = None,
    device: str = "auto",
    *,
    split_name: str | None = None,
) -> Path:
    cfg = cfg or DirectionConfig()
    split_name = split_name or cfg.split_name
    train_records, val_records, test_records, meta = build_split(
        split_name,
        data_dir,
        mic_filter=cfg.mic_filter,
        channel_filter=cfg.channel_filter,
        val_fraction=cfg.val_fraction,
        seed=cfg.split_seed,
    )
    meta["n_test_clips"] = len(test_records)
    persist_split_meta(meta, default_split_meta_path(split_name))

    prefix = cfg.task
    run_name = run_name or f"{prefix}_{cfg.feature_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR) / checkpoint_subdir(cfg) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    (out_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"{cfg.task} train ({split_name}) — {len(train_records)} train / {len(val_records)} valid")
    print(f"  held-out test clips: {len(test_records)}")
    print(f"  leakage audit ok: {meta['audit']['ok']}")
    print(f"  checkpoint -> {ckpt_path}")

    summary = train_on_records(train_records, val_records, cfg, device, ckpt_path, split_meta=meta)
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ckpt_path


def _fold_summary_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_name(f"{ckpt_path.stem}.summary.json")


def _fold_is_complete(ckpt_path: Path) -> bool:
    summary_path = _fold_summary_path(ckpt_path)
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return bool(data.get("fold_complete", False))
    if not ckpt_path.exists():
        return False
    try:
        _, _, _, ckpt = load_checkpoint(ckpt_path, "cpu")
        return bool(ckpt.get("fold_complete", False))
    except Exception:
        return False


def _load_skipped_fold_summary(ckpt_path: Path, fold_name: str) -> dict:
    summary_path = _fold_summary_path(ckpt_path)
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["fold_name"] = fold_name
        data["skipped_resume"] = True
        return data
    _, _, _, ckpt = load_checkpoint(ckpt_path, "cpu")
    return {
        "fold_name": fold_name,
        "checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("best_epoch", ckpt.get("epoch")),
        "best_val_acc": ckpt.get("best_val_acc", ckpt.get("val_acc")),
        "fold_complete": ckpt.get("fold_complete", False),
        "skipped_resume": True,
    }


def _should_train_fold(
    fold_name: str,
    ckpt_path: Path,
    *,
    force_retrain: bool,
    retrain_folds: list[str] | None,
    resume: bool,
) -> bool:
    if force_retrain:
        return True
    if retrain_folds and fold_name in retrain_folds:
        return True
    if not resume:
        return True
    if _fold_is_complete(ckpt_path):
        return False
    if ckpt_path.exists():
        return False
    return True


def train_location_loo(
    data_dir=None,
    checkpoint_dir: Path | None = None,
    run_name: str | None = None,
    cfg: DirectionConfig | None = None,
    device: str = "auto",
    *,
    resume: bool = True,
    force_retrain: bool = False,
    retrain_folds: list[str] | None = None,
) -> Path:
    cfg = cfg or DirectionConfig()
    folds = build_location_loo_splits(
        data_dir,
        mic_filter=cfg.mic_filter,
        channel_filter=cfg.channel_filter,
        val_fraction=cfg.val_fraction,
        seed=cfg.split_seed,
    )

    run_name = run_name or f"loc_loo_{cfg.feature_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR) / checkpoint_subdir(cfg) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    fold_summaries: list[dict] = []
    n_skipped = 0
    print(f"{cfg.task} train (location LOO) — {len(folds)} folds, device={resolve_device(device)}")
    if resume and not force_retrain:
        print("  resume=on — skipping folds with completed checkpoints")

    for location, train_records, val_records, test_records, meta in folds:
        fold_key = _sanitize_location(location)
        ckpt_path = out_dir / f"fold_{fold_key}.pt"

        if not _should_train_fold(
            fold_key, ckpt_path, force_retrain=force_retrain, retrain_folds=retrain_folds, resume=resume
        ):
            n_skipped += 1
            info = _load_skipped_fold_summary(ckpt_path, fold_key)
            print(f"\n  fold held-out: {location} — SKIPPED (best val_acc={info.get('best_val_acc', '?')})")
            fold_summaries.append(info)
            continue

        print(f"\n  fold held-out: {location}  train={len(train_records)} val={len(val_records)} test={len(test_records)}")
        meta["n_test_clips"] = len(test_records)
        summary = train_on_records(train_records, val_records, cfg, device, ckpt_path, split_meta=meta)
        summary["fold_name"] = fold_key
        summary["held_out_location"] = location
        fold_summaries.append(summary)

    (out_dir / "loo_train_summary.json").write_text(json.dumps(fold_summaries, indent=2), encoding="utf-8")
    n_done = len(list(out_dir.glob("fold_*.pt")))
    print(f"\nLocation LOO progress: {n_done}/{len(folds)} fold checkpoints ({n_skipped} skipped)")
    return out_dir
