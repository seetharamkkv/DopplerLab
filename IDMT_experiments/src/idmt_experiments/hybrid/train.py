"""Training loop and checkpoint I/O for hybrid direction models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
from idmt_experiments.hybrid.dataset import (
    collate_batch,
    fit_physics_scaler,
    make_dataset,
    precompute_batch,
)
from idmt_experiments.hybrid.model import HybridDirectionCNN
from idmt_experiments.src.features import fit_norm_stats
from idmt_experiments.src.preprocess import ClipRecord, filter_records
from idmt_experiments.src.splits import (
    build_split,
    default_split_meta_path,
    persist_split_meta,
    verify_no_event_leakage,
)
from idmt_experiments.training_resume import (
    load_training_state,
    save_training_state,
)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Hybrid training requires PyTorch. Install: pip install -r IDMT_experiments/requirements.txt"
        ) from exc
    return torch, nn, DataLoader


def save_checkpoint(
    path: Path,
    model,
    cfg: HybridConfig,
    *,
    norm_stats: NormStats | None = None,
    physics_scaler: PhysicsScaler | None = None,
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
        "physics_scaler": physics_scaler.to_dict() if physics_scaler else None,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu"):
    torch, _, _ = _require_torch()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = HybridConfig.from_dict(ckpt.get("config", ckpt.get("cfg", {})))
    model_type = ckpt.get("model", "hybrid")
    if model_type == "film":
        from idmt_experiments.hybrid.film_model import FiLMHybridDirectionCNN

        model = FiLMHybridDirectionCNN.build(cfg)
    else:
        model = HybridDirectionCNN.build(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    norm_stats = NormStats.from_dict(ckpt.get("norm_stats"))
    physics_scaler = PhysicsScaler.from_dict(ckpt.get("physics_scaler"))
    return model, cfg, norm_stats, physics_scaler, ckpt


def _run_epoch(model, loader, optim, loss_fn, device, train: bool, *, cfg: HybridConfig):
    torch, _, _ = _require_torch()
    model.train() if train else model.eval()

    losses: list[float] = []
    preds: list[int] = []
    truths: list[int] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    direction_cfg = cfg.to_direction_config()
    with ctx:
        for x_mel, x_phys, y, _meta in loader:
            x_mel, x_phys, y = x_mel.to(device), x_phys.to(device), y.to(device)
            logits = model(x_mel, x_phys)
            loss = loss_fn(logits, y)

            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()

            losses.append(float(loss.item()))
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            truths.extend(y.cpu().numpy().tolist())

    labels = resolve_class_labels(direction_cfg)
    metrics = classification_metrics(np.array(truths), np.array(preds), labels=labels)
    return float(np.mean(losses)), metrics, truths, preds


def train_on_records(
    train_records: list[ClipRecord],
    val_records: list[ClipRecord],
    cfg: HybridConfig,
    device: str,
    checkpoint_path: Path,
    *,
    split_meta: dict | None = None,
    resume: bool = False,
) -> dict:
    torch, nn, DataLoader = _require_torch()
    device = resolve_device(device)
    direction_cfg = cfg.to_direction_config()

    train_records = filter_records(train_records, direction_cfg)
    val_records = filter_records(val_records, direction_cfg)

    audit = verify_no_event_leakage(train_records, val_records, [])
    if not audit["ok"]:
        raise RuntimeError(f"Train/valid event leakage detected: {audit}")

    resume_state = load_training_state(checkpoint_path, device) if resume else None
    if resume and resume_state is None:
        print("  --resume-training set but no last.pt found — training from scratch.", flush=True)

    if resume_state is not None and resume_state.norm_stats is not None:
        print("  Reusing mel normalization stats from last.pt (resume)...", flush=True)
        norm_stats = NormStats.from_dict(resume_state.norm_stats)
    else:
        print("  Fitting mel normalization stats on train clips only...", flush=True)
        norm_stats = fit_norm_stats(
            train_records, direction_cfg, max_samples=cfg.norm_fit_max_samples, show_progress=True
        )

    if resume_state is not None and resume_state.physics_scaler is not None:
        print("  Reusing physics scaler from last.pt (resume)...", flush=True)
        physics_scaler = PhysicsScaler.from_dict(resume_state.physics_scaler)
    else:
        print("  Fitting physics scaler on train clips only...", flush=True)
        physics_scaler = fit_physics_scaler(train_records, cfg, show_progress=True)

    print("  Precomputing train features (mel + physics)...", flush=True)
    train_items = precompute_batch(
        train_records,
        cfg,
        norm_stats,
        physics_scaler,
        show_progress=True,
        desc="train hybrid",
    )
    print("  Precomputing val features (mel + physics)...", flush=True)
    val_items = precompute_batch(
        val_records, cfg, norm_stats, physics_scaler, show_progress=True, desc="val hybrid"
    )

    train_ds = make_dataset(
        train_records, cfg, norm_stats, physics_scaler, precomputed=train_items
    )
    val_ds = make_dataset(val_records, cfg, norm_stats, physics_scaler, precomputed=val_items)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0
    )

    model = HybridDirectionCNN.build(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state: dict | None = None
    patience_left = cfg.patience
    history: list[dict] = []
    start_epoch = 1

    if resume_state is not None:
        model.load_state_dict(resume_state.model_state)
        optim.load_state_dict(resume_state.optim_state)
        scheduler.load_state_dict(resume_state.scheduler_state)
        best_val_loss = resume_state.best_val_loss
        best_val_acc = resume_state.best_val_acc
        best_epoch = resume_state.best_epoch
        patience_left = resume_state.patience_left
        history = resume_state.history
        best_state = resume_state.best_state
        start_epoch = resume_state.epoch + 1
        print(
            f"  RESUMED from epoch {resume_state.epoch} (best val_acc={best_val_acc:.4f} "
            f"@ {best_epoch}) — continuing to epoch {cfg.epochs}",
            flush=True,
        )

    print(
        f"  preempt={cfg.preempt}  task={cfg.task}  feature={cfg.feature_type}+{cfg.feature_set}  "
        f"mono={cfg.mono_source}  n_classes={cfg.n_classes}"
    )
    print(f"  train={len(train_records)}  val={len(val_records)}  device={device}")

    if start_epoch > cfg.epochs:
        print(
            f"  Nothing to train: last completed epoch {start_epoch - 1} >= requested {cfg.epochs}.",
            flush=True,
        )
        return {
            "checkpoint": str(checkpoint_path),
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "history": history,
        }

    for epoch in range(start_epoch, cfg.epochs + 1):
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
                physics_scaler=physics_scaler,
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

        save_training_state(
            checkpoint_path,
            model,
            optim,
            scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            patience_left=patience_left,
            history=history,
            best_state=best_state,
            cfg_dict=cfg.to_dict(),
            norm_stats=norm_stats.to_dict() if norm_stats else None,
            physics_scaler=physics_scaler.to_dict() if physics_scaler else None,
            epochs_configured=cfg.epochs,
        )

        if cfg.preempt and patience_left <= 0:
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
            physics_scaler=physics_scaler,
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
    cfg: HybridConfig | None = None,
    device: str = "auto",
    *,
    split_name: str | None = None,
    resume: bool = False,
) -> Path:
    cfg = cfg or HybridConfig()
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

    run_name = run_name or (
        f"hybrid_{cfg.feature_type}_{cfg.mono_source}_{cfg.feature_set}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR) / hybrid_checkpoint_subdir(cfg) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    (out_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"hybrid train ({split_name}) — {len(train_records)} train / {len(val_records)} valid")
    print(f"  held-out test clips: {len(test_records)}")
    print(f"  leakage audit ok: {meta['audit']['ok']}")
    print(f"  checkpoint -> {ckpt_path}")

    summary = train_on_records(
        train_records, val_records, cfg, device, ckpt_path, split_meta=meta, resume=resume
    )
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir
