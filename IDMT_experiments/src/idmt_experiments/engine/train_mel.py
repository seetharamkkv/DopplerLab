"""Shared mel training engine for experimental models (transfer / fusion / FiLM)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from idmt_experiments.config import DEFAULT_CHECKPOINT_DIR, DirectionConfig, NormStats, resolve_class_labels
from idmt_experiments.cnn.dataset import collate_batch, make_dataset, precompute_batch
from idmt_experiments.cnn.metrics import classification_metrics
from idmt_experiments.cnn.train import resolve_device, save_checkpoint
from idmt_experiments.cnn.train_recipe import (
    TrainRecipe,
    build_loss_fn,
    build_train_loader,
    phase_a_recipe,
    set_epoch_lr,
)
from idmt_experiments.src.features import fit_norm_stats
from idmt_experiments.src.preprocess import ClipRecord, filter_records
from idmt_experiments.src.splits import build_split, default_split_meta_path, persist_split_meta, verify_no_event_leakage
from idmt_experiments.training_resume import load_training_state, save_training_state


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Requires PyTorch.") from exc
    return torch, nn, DataLoader


def _run_epoch_mel(
    model,
    loader,
    optim,
    loss_fn,
    device,
    train: bool,
    *,
    cfg: DirectionConfig,
    grad_clip_norm: float | None = None,
    forward_fn=None,
    start_batch: int = 0,
    on_batch_end=None,
    progress_desc: str | None = None,
):
    torch, _, _ = _require_torch()
    from tqdm import tqdm

    model.train() if train else model.eval()
    forward_fn = forward_fn or (lambda m, batch: m(batch[0]))

    losses: list[float] = []
    preds: list[int] = []
    truths: list[int] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    batch_iter = tqdm(loader, desc=progress_desc or "batches", leave=True, total=len(loader)) if progress_desc else loader
    with ctx:
        for batch_i, batch in enumerate(batch_iter):
            if train and batch_i < start_batch:
                continue
            x, y = batch[0].to(device), batch[1].to(device)
            # forward_fn expects a batch tuple; keep any trailing meta on CPU
            moved = (x, y, *batch[2:]) if len(batch) > 2 else (x, y)
            logits = forward_fn(model, moved)
            loss = loss_fn(logits, y)
            if train:
                optim.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optim.step()
                if on_batch_end is not None:
                    on_batch_end(batch_i + 1, float(loss.item()))
            losses.append(float(loss.item()))
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            truths.extend(y.cpu().numpy().tolist())
            if progress_desc and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{losses[-1]:.4f}", refresh=False)

    labels = resolve_class_labels(cfg)
    metrics = classification_metrics(np.array(truths), np.array(preds), labels=labels)
    return float(np.mean(losses)), metrics, truths, preds


def train_mel_experiment(
    train_records: list[ClipRecord],
    val_records: list[ClipRecord],
    cfg: DirectionConfig,
    device: str,
    checkpoint_path: Path,
    model_factory: Callable,
    *,
    split_meta: dict | None = None,
    resume: bool = False,
    train_recipe: TrainRecipe | None = None,
    forward_fn=None,
    extra_ckpt: dict | None = None,
    checkpoint_every_batches: int = 25,
) -> dict:
    torch, nn, DataLoader = _require_torch()
    device = resolve_device(device)
    recipe = train_recipe or phase_a_recipe()

    train_records = filter_records(train_records, cfg)
    val_records = filter_records(val_records, cfg)
    audit = verify_no_event_leakage(train_records, val_records, [])
    if not audit["ok"]:
        raise RuntimeError(f"Train/valid event leakage detected: {audit}")

    resume_state = load_training_state(checkpoint_path, device) if resume else None
    if resume_state is not None and resume_state.norm_stats is not None:
        norm_stats = NormStats.from_dict(resume_state.norm_stats)
    else:
        norm_stats = fit_norm_stats(
            train_records, cfg, max_samples=cfg.norm_fit_max_samples, show_progress=True
        )

    train_items = precompute_batch(train_records, cfg, norm_stats, show_progress=True, desc=f"train {cfg.feature_type}")
    val_items = precompute_batch(val_records, cfg, norm_stats, show_progress=True, desc=f"val {cfg.feature_type}")

    train_ds = make_dataset(
        train_records,
        cfg,
        norm_stats,
        precomputed=train_items,
        augment=recipe.spec_augment,
        time_mask_param=recipe.time_mask_param,
        freq_mask_param=recipe.freq_mask_param,
        num_time_masks=recipe.num_time_masks,
        num_freq_masks=recipe.num_freq_masks,
    )
    val_ds = make_dataset(val_records, cfg, norm_stats, precomputed=val_items)
    train_labels = [item.y for item in train_items]
    train_loader = build_train_loader(DataLoader, train_ds, cfg, train_labels, recipe, collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model = model_factory(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None if recipe.lr_cosine else torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)
    loss_fn = build_loss_fn(nn, train_labels, recipe, device)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_left = cfg.patience
    history: list[dict] = []
    start_epoch = 1
    start_batch = 0

    if resume_state is not None:
        model.load_state_dict(resume_state.model_state)
        optim.load_state_dict(resume_state.optim_state)
        if scheduler is not None and resume_state.scheduler_state is not None:
            scheduler.load_state_dict(resume_state.scheduler_state)
        best_val_loss = resume_state.best_val_loss
        best_val_acc = resume_state.best_val_acc
        best_epoch = resume_state.best_epoch
        patience_left = resume_state.patience_left
        history = resume_state.history
        best_state = resume_state.best_state
        if resume_state.mid_epoch:
            start_epoch = resume_state.epoch
            start_batch = resume_state.batch_idx
            print(
                f"  RESUMED mid-epoch: epoch {start_epoch} batch {start_batch} "
                f"(best val_bal @ epoch {best_epoch})",
                flush=True,
            )
        else:
            start_epoch = resume_state.epoch + 1
            print(
                f"  RESUMED after epoch {resume_state.epoch} "
                f"(best val_acc={best_val_acc:.4f} @ {best_epoch}) — continuing to {cfg.epochs}",
                flush=True,
            )

    print(
        f"  recipe: augment={recipe.spec_augment} balanced={recipe.balanced_sampler} "
        f"focal={recipe.focal_loss} grad_clip={recipe.grad_clip_norm}",
        flush=True,
    )
    print(
        f"  preempt={cfg.preempt}  min_epochs={cfg.min_epochs}  patience={cfg.patience}  "
        f"mono={cfg.mono_source}  n_classes={cfg.n_classes}  device={device}",
        flush=True,
    )

    ckpt_extra_base = dict(extra_ckpt or {})

    def _persist(batch_idx: int, epoch_num: int) -> None:
        save_training_state(
            checkpoint_path, model, optim, scheduler, epoch=epoch_num,
            best_val_loss=best_val_loss, best_val_acc=best_val_acc, best_epoch=best_epoch,
            patience_left=patience_left, history=history, best_state=best_state,
            cfg_dict=cfg.to_dict(), norm_stats=norm_stats.to_dict() if norm_stats else None,
            physics_scaler=None, epochs_configured=cfg.epochs, batch_idx=batch_idx,
        )

    for epoch in range(start_epoch, cfg.epochs + 1):
        set_epoch_lr(optim, cfg, recipe, epoch)
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        start_batch = 0

        def on_batch_end(done: int, _loss: float) -> None:
            if checkpoint_every_batches > 0 and done % checkpoint_every_batches == 0:
                _persist(done, epoch)

        print(f"\n=== Epoch {epoch}/{cfg.epochs} ===", flush=True)
        train_loss, train_m, _, _ = _run_epoch_mel(
            model, train_loader, optim, loss_fn, device, True, cfg=cfg,
            grad_clip_norm=recipe.grad_clip_norm, forward_fn=forward_fn,
            start_batch=epoch_start_batch, on_batch_end=on_batch_end,
            progress_desc=f"epoch {epoch}/{cfg.epochs} train",
        )
        val_loss, val_m, _, _ = _run_epoch_mel(
            model, val_loader, optim, loss_fn, device, False, cfg=cfg, forward_fn=forward_fn,
            progress_desc=f"epoch {epoch}/{cfg.epochs} val",
        )
        if scheduler is not None:
            scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_m["accuracy"],
                "val_loss": val_loss,
                "val_acc": val_m["accuracy"],
                "val_bal_acc": val_m["balanced_accuracy"],
                "val_macro_f1": val_m["macro_f1"],
            }
        )
        print(
            f"  epoch {epoch:3d}  train_acc={train_m['accuracy']:.4f}  "
            f"val_acc={val_m['accuracy']:.4f}  val_bal={val_m['balanced_accuracy']:.4f}  "
            f"val_f1={val_m['macro_f1']:.4f}"
        )

        if val_loss < best_val_loss:
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
                    **ckpt_extra_base,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_m["accuracy"],
                    "val_bal_acc": val_m["balanced_accuracy"],
                    "val_metrics": val_m,
                    "n_train": len(train_records),
                    "n_val": len(val_records),
                    "split_meta": split_meta,
                    "train_recipe": recipe.to_dict(),
                    "fold_complete": False,
                },
            )
        elif cfg.preempt and epoch >= cfg.min_epochs:
            patience_left -= 1

        save_training_state(
            checkpoint_path, model, optim, scheduler, epoch=epoch,
            best_val_loss=best_val_loss, best_val_acc=best_val_acc, best_epoch=best_epoch,
            patience_left=patience_left, history=history, best_state=best_state,
            cfg_dict=cfg.to_dict(), norm_stats=norm_stats.to_dict() if norm_stats else None,
            physics_scaler=None, epochs_configured=cfg.epochs, batch_idx=0,
        )
        if cfg.preempt and patience_left <= 0 and epoch >= cfg.min_epochs:
            print(
                f"  early stop @ epoch {epoch} "
                f"(best val_bal={history[best_epoch-1]['val_bal_acc']:.4f} @ {best_epoch}; "
                f"min_epochs={cfg.min_epochs})",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(
            checkpoint_path, model, cfg, norm_stats=norm_stats,
            extra={
                **ckpt_extra_base,
                "epoch": best_epoch,
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "final_epoch": history[-1]["epoch"] if history else 0,
                "epochs_configured": cfg.epochs,
                "preempt": cfg.preempt,
                "fold_complete": True,
                "train_recipe": recipe.to_dict(),
            },
        )

    (checkpoint_path.parent / f"{checkpoint_path.stem}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
    }
    (checkpoint_path.parent / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_mel_split(
    *,
    run_name: str,
    cfg: DirectionConfig,
    model_factory: Callable,
    checkpoint_subdir: str,
    device: str = "auto",
    data_dir=None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    train_recipe: TrainRecipe | None = None,
    forward_fn=None,
    extra_ckpt: dict | None = None,
    checkpoint_every_batches: int = 25,
) -> Path:
    train_records, val_records, test_records, meta = build_split(
        cfg.split_name,
        data_dir,
        mic_filter=cfg.mic_filter,
        channel_filter=cfg.channel_filter,
        val_fraction=cfg.val_fraction,
        seed=cfg.split_seed,
    )
    meta["n_test_clips"] = len(test_records)
    persist_split_meta(meta, default_split_meta_path(cfg.split_name))

    out_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR) / checkpoint_subdir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    (out_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    if train_recipe:
        (out_dir / "train_recipe.json").write_text(json.dumps(train_recipe.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"train ({cfg.split_name}) — {len(train_records)} train / {len(val_records)} valid / {len(test_records)} test")
    print(f"  checkpoint -> {ckpt_path}")

    train_mel_experiment(
        train_records, val_records, cfg, device, ckpt_path, model_factory,
        split_meta=meta, resume=resume, train_recipe=train_recipe,
        forward_fn=forward_fn, extra_ckpt=extra_ckpt,
        checkpoint_every_batches=checkpoint_every_batches,
    )
    return ckpt_path
