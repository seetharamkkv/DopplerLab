#!/usr/bin/env python3
"""Phase D: FiLM hybrid + flip-consistency loss (2-class L2R/R2L)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from idmt_experiments.config import (
    DEFAULT_CHECKPOINT_DIR,
    HybridConfig,
    NormStats,
    PhysicsScaler,
    hybrid_checkpoint_subdir,
    resolve_class_labels,
)
from idmt_experiments.cnn.metrics import classification_metrics
from idmt_experiments.cnn.train import resolve_device
from idmt_experiments.hybrid.train import save_checkpoint
from idmt_experiments.cnn.train_recipe import phase_a_recipe, set_epoch_lr
from idmt_experiments.hybrid.dataset import collate_batch, fit_physics_scaler, make_dataset, precompute_batch
from idmt_experiments.hybrid.film_model import FiLMHybridDirectionCNN
from idmt_experiments.src.features import fit_norm_stats
from idmt_experiments.src.preprocess import filter_records
from idmt_experiments.src.splits import build_split, default_split_meta_path, persist_split_meta, verify_no_event_leakage
from idmt_experiments.training_resume import load_training_state, save_training_state


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Requires PyTorch.") from exc
    return torch, nn, F, DataLoader


def _run_epoch_film(
    model, loader, optim, device, train: bool, *, cfg: HybridConfig, flip_lambda: float, grad_clip: float | None
):
    torch, _, F, _ = _require_torch()
    model.train() if train else model.eval()
    direction_cfg = cfg.to_direction_config()
    losses: list[float] = []
    preds: list[int] = []
    truths: list[int] = []
    flip_ok: list[float] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_mel, x_phys, y, _meta in loader:
            x_mel, x_phys, y = x_mel.to(device), x_phys.to(device), y.to(device)
            logits = model(x_mel, x_phys)
            x_mel_rev = torch.flip(x_mel, dims=[-1])
            x_phys_rev = -x_phys
            logits_rev = model(x_mel_rev, x_phys_rev)

            ce = F.cross_entropy(logits, y)
            y_flip = 1 - y
            flip_ce = F.cross_entropy(logits_rev, y_flip)
            loss = ce + flip_lambda * flip_ce

            if train:
                optim.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()

            losses.append(float(loss.item()))
            pred = logits.argmax(dim=1)
            preds.extend(pred.detach().cpu().numpy().tolist())
            truths.extend(y.cpu().numpy().tolist())
            flip_ok.extend((logits_rev.argmax(dim=1) == y_flip).float().cpu().numpy().tolist())

    labels = resolve_class_labels(direction_cfg)
    metrics = classification_metrics(np.array(truths), np.array(preds), labels=labels)
    metrics["flip_batch_consistency"] = float(np.mean(flip_ok)) if flip_ok else None
    return float(np.mean(losses)), metrics, truths, preds


def train_film_split(
    *,
    run_name: str,
    cfg: HybridConfig,
    device: str = "auto",
    flip_lambda: float = 0.5,
    resume: bool = False,
) -> Path:
    torch, _, _, DataLoader = _require_torch()
    device = resolve_device(device)
    recipe = phase_a_recipe()

    train_records, val_records, test_records, meta = build_split(
        cfg.split_name, None,
        mic_filter=cfg.mic_filter, channel_filter=cfg.channel_filter,
        val_fraction=cfg.val_fraction, seed=cfg.split_seed,
    )
    meta["n_test_clips"] = len(test_records)
    persist_split_meta(meta, default_split_meta_path(cfg.split_name))

    train_records = filter_records(train_records, cfg.to_direction_config())
    val_records = filter_records(val_records, cfg.to_direction_config())

    audit = verify_no_event_leakage(train_records, val_records, [])
    if not audit["ok"]:
        raise RuntimeError(f"Leakage: {audit}")

    out_dir = Path(DEFAULT_CHECKPOINT_DIR) / hybrid_checkpoint_subdir(cfg) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    (out_dir / "run_config.json").write_text(json.dumps({**cfg.to_dict(), "flip_lambda": flip_lambda, "model": "film", "phase": "D"}, indent=2), encoding="utf-8")

    resume_state = load_training_state(ckpt_path, device) if resume else None
    norm_stats = NormStats.from_dict(resume_state.norm_stats) if resume_state and resume_state.norm_stats else fit_norm_stats(
        train_records, cfg.to_direction_config(), max_samples=cfg.norm_fit_max_samples, show_progress=True
    )
    physics_scaler = PhysicsScaler.from_dict(resume_state.physics_scaler) if resume_state and resume_state.physics_scaler else fit_physics_scaler(
        train_records, cfg, show_progress=True
    )

    train_items = precompute_batch(train_records, cfg, norm_stats, physics_scaler, show_progress=True, desc="train film")
    val_items = precompute_batch(val_records, cfg, norm_stats, physics_scaler, show_progress=True, desc="val film")
    train_ds = make_dataset(train_records, cfg, norm_stats, physics_scaler, precomputed=train_items)
    val_ds = make_dataset(val_records, cfg, norm_stats, physics_scaler, precomputed=val_items)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model = FiLMHybridDirectionCNN.build(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None if recipe.lr_cosine else torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_bal = 0.0
    best_epoch = 0
    best_state = None
    patience_left = cfg.patience
    history: list[dict] = []
    start_epoch = 1

    if resume_state is not None:
        model.load_state_dict(resume_state.model_state)
        optim.load_state_dict(resume_state.optim_state)
        best_val_loss = resume_state.best_val_loss
        best_bal = resume_state.best_val_acc
        best_epoch = resume_state.best_epoch
        patience_left = resume_state.patience_left
        history = resume_state.history
        best_state = resume_state.best_state
        start_epoch = resume_state.epoch + 1

    print(f"  Phase D FiLM  flip_lambda={flip_lambda}  n_classes={cfg.n_classes}  epochs={cfg.epochs}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        set_epoch_lr(optim, cfg.to_direction_config(), recipe, epoch)
        tr_loss, tr_m, _, _ = _run_epoch_film(
            model, train_loader, optim, device, True, cfg=cfg, flip_lambda=flip_lambda, grad_clip=recipe.grad_clip_norm
        )
        va_loss, va_m, _, _ = _run_epoch_film(
            model, val_loader, optim, device, False, cfg=cfg, flip_lambda=flip_lambda, grad_clip=None
        )
        if scheduler is not None:
            scheduler.step(va_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_bal_acc": tr_m["balanced_accuracy"],
                "val_loss": va_loss,
                "val_bal_acc": va_m["balanced_accuracy"],
                "val_flip_consistency": va_m.get("flip_batch_consistency"),
            }
        )
        print(
            f"  epoch {epoch:3d}  train_bal={tr_m['balanced_accuracy']:.4f}  "
            f"val_bal={va_m['balanced_accuracy']:.4f}  flip={va_m.get('flip_batch_consistency') or 0:.4f}"
        )
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_bal = va_m["balanced_accuracy"]
            best_epoch = epoch
            patience_left = cfg.patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(
                ckpt_path, model, cfg, norm_stats=norm_stats, physics_scaler=physics_scaler,
                extra={"epoch": epoch, "val_bal_acc": best_bal, "model": "film", "phase": "D", "flip_lambda": flip_lambda},
            )
        elif cfg.preempt:
            patience_left -= 1

        save_training_state(
            ckpt_path, model, optim, scheduler, epoch=epoch,
            best_val_loss=best_val_loss, best_val_acc=best_bal, best_epoch=best_epoch,
            patience_left=patience_left, history=history, best_state=best_state,
            cfg_dict=cfg.to_dict(), norm_stats=norm_stats.to_dict() if norm_stats else None,
            physics_scaler=physics_scaler.to_dict() if physics_scaler else None,
            epochs_configured=cfg.epochs,
        )
        if cfg.preempt and patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(
            ckpt_path, model, cfg, norm_stats=norm_stats, physics_scaler=physics_scaler,
            extra={"best_epoch": best_epoch, "best_val_bal_acc": best_bal, "model": "film", "phase": "D"},
        )
    (out_dir / "train_summary.json").write_text(json.dumps({"best_epoch": best_epoch, "best_val_bal_acc": best_bal}, indent=2), encoding="utf-8")
    return ckpt_path


def main() -> None:
    p = argparse.ArgumentParser(description="Phase D FiLM + flip loss")
    p.add_argument("--run-name", default="film_2class_left_100ep")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--flip-lambda", type=float, default=0.5)
    p.add_argument("--preempt", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    cfg = HybridConfig(
        mono_source="left",
        n_classes=2,
        epochs=args.epochs,
        lr=1e-4,
        weight_decay=1e-3,
        preempt=args.preempt,
        patience=15,
        include_no_vehicle=False,
    )

    print("=" * 72)
    print("PHASE D - FiLM + flip-consistency loss (2-class)")
    print("=" * 72)
    ckpt = train_film_split(run_name=args.run_name, cfg=cfg, device=args.device, flip_lambda=args.flip_lambda)

    if not args.skip_eval:
        from idmt_experiments.hybrid.eval import run_eval

        print("\nAUTO EVAL (test)")
        run_eval(checkpoint=ckpt, device=args.device)
    print("\nDone.")


if __name__ == "__main__":
    main()
