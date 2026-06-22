"""Training loop and checkpoint I/O for length CNN."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from length_estimation.config import DEFAULT_CHECKPOINT_DIR, PhaseBConfig
from length_estimation.src.evaluate import lovo_splits, regression_metrics
from length_estimation.src.phase_b.dataset import collate_batch, make_dataset
from length_estimation.src.phase_b.model import PassByLengthCNN
from length_estimation.src.preprocess import ClipRecord, load_clips, resolve_data_dir


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Phase B requires PyTorch. Install: pip install -r length_estimation/requirements.txt"
        ) from exc
    return torch, nn, DataLoader


def resolve_device(device: str = "auto") -> str:
    torch, _, _ = _require_torch()
    cuda_ok = torch.cuda.is_available()
    if device == "auto":
        return "cuda" if cuda_ok else "cpu"
    if device.startswith("cuda") and not cuda_ok:
        print(
            "WARNING: CUDA requested but this PyTorch build has no GPU support "
            "(CPU-only install). Falling back to CPU.\n"
            "  To use GPU: pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  Or run now with: --device cpu"
        )
        return "cpu"
    return device


def save_checkpoint(
    path: Path,
    model,
    cfg: PhaseBConfig,
    *,
    extra: dict | None = None,
) -> None:
    torch, _, _ = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "target": cfg.target,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu"):
    torch, _, _ = _require_torch()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = PhaseBConfig.from_dict(ckpt.get("config", ckpt.get("cfg", {})))
    model = PassByLengthCNN.build(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, ckpt


def _run_epoch(model, loader, optim, loss_fn, device, train: bool):
    torch, _, _ = _require_torch()
    model.train() if train else model.eval()

    losses: list[float] = []
    preds: list[float] = []
    truths: list[float] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, speed, _meta in loader:
            x, y = x.to(device), y.to(device)
            speed_t = speed.to(device) if speed is not None else None
            pred = model(x, speed_t)
            loss = loss_fn(pred, y)

            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()

            losses.append(float(loss.item()))
            preds.extend(pred.detach().cpu().numpy().tolist())
            truths.extend(y.cpu().numpy().tolist())

    mae = float(np.mean(np.abs(np.array(preds) - np.array(truths))))
    return float(np.mean(losses)), mae, truths, preds


def train_on_records(
    train_records: list[ClipRecord],
    val_records: list[ClipRecord],
    cfg: PhaseBConfig,
    device: str,
    checkpoint_path: Path,
) -> dict:
    """
    Train on train_records, validate on val_records.

    Always tracks best validation epoch and saves weights to checkpoint_path (best.pt).
    - preempt=False (default): run all epochs, no early stopping
    - preempt=True: stop when val MAE fails to improve for `patience` epochs
    """
    torch, nn, DataLoader = _require_torch()
    device = resolve_device(device)

    train_ds = make_dataset(train_records, cfg)
    val_ds = make_dataset(val_records, cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0
    )

    model = PassByLengthCNN.build(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=4)
    loss_fn = nn.HuberLoss(delta=cfg.huber_delta)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    patience_left = cfg.patience
    history: list[dict] = []

    print(f"  preempt={cfg.preempt}  (early stop {'on' if cfg.preempt else 'off — full epoch schedule'})")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_mae, _, _ = _run_epoch(model, train_loader, optim, loss_fn, device, train=True)
        val_loss, val_mae, val_true, val_pred = _run_epoch(
            model, val_loader, optim, loss_fn, device, train=False
        )
        scheduler.step(val_mae)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        print(f"  epoch {epoch:3d}  train_mae={train_mae:.4f}m  val_mae={val_mae:.4f}m")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_left = cfg.patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            target_range = max(val_true) - min(val_true) if val_true else 1.0
            metrics = regression_metrics(np.array(val_true), np.array(val_pred), float(target_range))
            save_checkpoint(
                checkpoint_path,
                model,
                cfg,
                extra={
                    "epoch": epoch,
                    "val_mae": val_mae,
                    "val_metrics": metrics,
                    "n_train": len(train_records),
                    "n_val": len(val_records),
                    "fold_complete": False,
                },
            )
        else:
            if cfg.preempt:
                patience_left -= 1
                if patience_left <= 0:
                    print(
                        f"  early stop at epoch {epoch} "
                        f"(best val_mae={best_val_mae:.4f}m @ epoch {best_epoch})"
                    )
                    break

    final_epoch = history[-1]["epoch"] if history else 0
    if cfg.preempt and final_epoch < cfg.epochs and patience_left <= 0:
        pass  # message already printed
    else:
        print(
            f"  finished epoch {final_epoch}/{cfg.epochs}  "
            f"best val_mae={best_val_mae:.4f}m @ epoch {best_epoch}"
        )

    (checkpoint_path.parent / f"{checkpoint_path.stem}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(
            checkpoint_path,
            model,
            cfg,
            extra={
                "epoch": best_epoch,
                "val_mae": best_val_mae,
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "final_epoch": final_epoch,
                "epochs_configured": cfg.epochs,
                "preempt": cfg.preempt,
                "fold_complete": True,
                "n_train": len(train_records),
                "n_val": len(val_records),
            },
        )

    summary_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.summary.json")
    summary = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "final_epoch": final_epoch,
        "preempt": cfg.preempt,
        "epochs_configured": cfg.epochs,
        "fold_complete": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "final_epoch": final_epoch,
        "preempt": cfg.preempt,
        "epochs_configured": cfg.epochs,
        "history": history,
    }


def train_split(
    data_dir=None,
    checkpoint_dir: Path | None = None,
    run_name: str | None = None,
    cfg: PhaseBConfig | None = None,
    device: str = "auto",
) -> Path:
    """
    VS13 official train/valid split (Train_valid_split.txt per vehicle).
    Saves best checkpoint to checkpoints/length_cnn/{run_name}/best.pt
    """
    cfg = cfg or PhaseBConfig()
    data_dir = resolve_data_dir(data_dir)
    records = load_clips(data_dir)
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "valid"]

    if not train_records or not val_records:
        raise RuntimeError(
            "Train/valid split not found. Ensure Train_valid_split.txt exists in each vehicle folder."
        )

    run_name = run_name or f"{cfg.spec_type}_length_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR) / "length_cnn" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    (out_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    print(f"Phase B train (split) — {len(train_records)} train / {len(val_records)} valid clips")
    print(f"  spec={cfg.spec_type}  speed_aux={cfg.include_speed}  device={resolve_device(device)}")
    print(f"  checkpoint -> {ckpt_path}")

    summary = train_on_records(train_records, val_records, cfg, device, ckpt_path)
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ckpt_path


def _fold_summary_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_name(f"{ckpt_path.stem}.summary.json")


def _fold_is_complete(ckpt_path: Path) -> bool:
    """True when this LOVO fold finished normally (safe to skip on resume)."""
    summary_path = _fold_summary_path(ckpt_path)
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return bool(data.get("fold_complete", False))
    if not ckpt_path.exists():
        return False
    try:
        _, _, ckpt = load_checkpoint(ckpt_path, "cpu")
        return bool(ckpt.get("fold_complete", False))
    except Exception:
        return False


def _load_skipped_fold_summary(ckpt_path: Path, vehicle: str) -> dict:
    summary_path = _fold_summary_path(ckpt_path)
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["vehicle"] = vehicle
        data["skipped_resume"] = True
        return data
    _, _, ckpt = load_checkpoint(ckpt_path, "cpu")
    return {
        "vehicle": vehicle,
        "checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("best_epoch", ckpt.get("epoch")),
        "best_val_mae": ckpt.get("best_val_mae", ckpt.get("val_mae")),
        "fold_complete": ckpt.get("fold_complete", False),
        "skipped_resume": True,
        "note": "legacy checkpoint (no summary.json); treated as complete because file exists",
    }


def _should_train_fold(
    vehicle: str,
    ckpt_path: Path,
    *,
    force_retrain: bool,
    retrain_folds: list[str] | None,
    resume: bool,
) -> bool:
    if force_retrain:
        return True
    if retrain_folds and vehicle in retrain_folds:
        return True
    if not resume:
        return True
    if _fold_is_complete(ckpt_path):
        return False
    # Legacy checkpoint (finished before fold_complete existed): keep unless --retrain-folds
    if ckpt_path.exists():
        return False
    return True


def train_lovo(
    data_dir=None,
    checkpoint_dir: Path | None = None,
    run_name: str | None = None,
    cfg: PhaseBConfig | None = None,
    device: str = "auto",
    *,
    resume: bool = True,
    force_retrain: bool = False,
    retrain_folds: list[str] | None = None,
) -> Path:
    """
    13-fold leave-one-vehicle-out training.
    Saves per-fold checkpoints + combined manifest; returns run directory.

    Resume (default): skip folds that already have a completed checkpoint.
    - ``--force-retrain``: retrain all folds (overwrite).
    - ``--retrain-folds KiaSportage``: retrain only named folds (e.g. interrupted).
    - ``--no-resume``: always train every fold (same as fresh run without deleting files).
    """
    import pandas as pd

    cfg = cfg or PhaseBConfig()
    data_dir = resolve_data_dir(data_dir)
    records = load_clips(data_dir)
    by_id = {r.clip_id: r for r in records}
    df = pd.DataFrame([{"clip_id": r.clip_id, "vehicle": r.vehicle, "length_m": r.length_m} for r in records])

    run_name = run_name or f"lovo_{cfg.spec_type}_length_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR) / "length_cnn" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    fold_summaries: list[dict] = []
    n_skipped = 0
    print(f"Phase B train (LOVO) — 13 folds, device={resolve_device(device)}")
    if resume and not force_retrain:
        print("  resume=on — skipping folds with existing completed checkpoints")
    if retrain_folds:
        print(f"  retrain-folds: {', '.join(retrain_folds)}")

    for vehicle, train_idx, test_idx in lovo_splits(df):
        train_records = [by_id[cid] for cid in df.loc[train_idx, "clip_id"]]
        test_records = [by_id[cid] for cid in df.loc[test_idx, "clip_id"]]
        ckpt_path = out_dir / f"fold_{vehicle}.pt"

        if not _should_train_fold(
            vehicle, ckpt_path, force_retrain=force_retrain, retrain_folds=retrain_folds, resume=resume
        ):
            n_skipped += 1
            info = _load_skipped_fold_summary(ckpt_path, vehicle)
            best_mae = info.get("best_val_mae", "?")
            best_ep = info.get("best_epoch", "?")
            print(f"\n  fold held-out: {vehicle}  — SKIPPED (checkpoint exists, best val_mae={best_mae} @ epoch {best_ep})")
            fold_summaries.append(info)
            continue

        print(f"\n  fold held-out: {vehicle}  (train={len(train_records)}, test={len(test_records)})")
        summary = train_on_records(train_records, test_records, cfg, device, ckpt_path)
        summary["vehicle"] = vehicle
        fold_summaries.append(summary)

    (out_dir / "lovo_train_summary.json").write_text(json.dumps(fold_summaries, indent=2), encoding="utf-8")
    n_done = len([p for p in out_dir.glob("fold_*.pt")])
    print(f"\nLOVO progress: {n_done}/13 fold checkpoints  ({n_skipped} skipped this run)")
    return out_dir
