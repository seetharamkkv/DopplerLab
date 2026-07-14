"""Shared epoch-level resume support for PyTorch trainers (CNN + hybrid).

Enables ``--resume-training``: a run can be extended by increasing ``--epochs`` and
continuing from the last completed epoch instead of retraining from scratch. Full training
state (model + optimizer + scheduler + bookkeeping) is snapshotted to ``last.pt`` after
every epoch; the best-by-val-loss weights stay in ``best.pt`` exactly as before.

This is additive and opt-in: when resume is not requested the training numerics are
unchanged (an extra ``last.pt`` artifact is written but no results depend on it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Requires PyTorch.") from exc
    return torch


def last_state_path(checkpoint_path: Path) -> Path:
    """Sibling ``last.pt`` next to the run's ``best.pt``."""
    return Path(checkpoint_path).parent / "last.pt"


@dataclass
class ResumeState:
    epoch: int  # last fully completed epoch, or in-progress epoch if batch_idx > 0
    batch_idx: int  # training batches completed in current in-progress epoch
    best_val_loss: float
    best_val_acc: float
    best_epoch: int
    patience_left: int
    history: list[dict]
    best_state: dict | None
    model_state: dict
    optim_state: dict
    scheduler_state: dict | None
    norm_stats: dict | None
    physics_scaler: dict | None
    epochs_configured: int

    @property
    def mid_epoch(self) -> bool:
        return self.batch_idx > 0


def save_training_state(
    checkpoint_path: Path,
    model,
    optim,
    scheduler,
    *,
    epoch: int,
    best_val_loss: float,
    best_val_acc: float,
    best_epoch: int,
    patience_left: int,
    history: list[dict],
    best_state: dict | None,
    cfg_dict: dict,
    norm_stats: dict | None = None,
    physics_scaler: dict | None = None,
    epochs_configured: int,
    batch_idx: int = 0,
) -> None:
    torch = _require_torch()
    path = last_state_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optim_state": optim.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "patience_left": patience_left,
        "history": history,
        "best_state": best_state,
        "config": cfg_dict,
        "norm_stats": norm_stats,
        "physics_scaler": physics_scaler,
        "epochs_configured": epochs_configured,
    }
    torch.save(payload, path)


def load_training_state(checkpoint_path: Path, device: str = "cpu") -> ResumeState | None:
    torch = _require_torch()
    path = last_state_path(checkpoint_path)
    if not path.exists():
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    return ResumeState(
        epoch=int(ck["epoch"]),
        batch_idx=int(ck.get("batch_idx", 0)),
        best_val_loss=float(ck["best_val_loss"]),
        best_val_acc=float(ck["best_val_acc"]),
        best_epoch=int(ck["best_epoch"]),
        patience_left=int(ck["patience_left"]),
        history=list(ck.get("history", [])),
        best_state=ck.get("best_state"),
        model_state=ck["state_dict"],
        optim_state=ck["optim_state"],
        scheduler_state=ck.get("scheduler_state"),
        norm_stats=ck.get("norm_stats"),
        physics_scaler=ck.get("physics_scaler"),
        epochs_configured=int(ck.get("epochs_configured", 0)),
    )
