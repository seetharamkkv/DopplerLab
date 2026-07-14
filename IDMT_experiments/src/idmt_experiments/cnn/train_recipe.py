"""Opt-in training recipe for Phase A+ improvements (baseline defaults unchanged)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class TrainRecipe:
    spec_augment: bool = False
    time_mask_param: int = 20
    freq_mask_param: int = 8
    num_time_masks: int = 2
    num_freq_masks: int = 2
    balanced_sampler: bool = False
    focal_loss: bool = False
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    grad_clip_norm: float | None = None
    lr_warmup_epochs: int = 0
    lr_cosine: bool = False

    def is_active(self) -> bool:
        return (
            self.spec_augment
            or self.balanced_sampler
            or self.focal_loss
            or self.label_smoothing > 0
            or self.grad_clip_norm is not None
            or self.lr_warmup_epochs > 0
            or self.lr_cosine
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> TrainRecipe:
        if not d:
            return cls()
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def phase_a_recipe() -> TrainRecipe:
    """Full Phase A recipe tuned for mono-left direction (max generalization)."""
    return TrainRecipe(
        spec_augment=True,
        balanced_sampler=True,
        focal_loss=True,
        focal_gamma=2.0,
        label_smoothing=0.1,
        grad_clip_norm=1.0,
        lr_warmup_epochs=3,
        lr_cosine=True,
    )


def build_loss_fn(nn, labels: list[int], recipe: TrainRecipe, device: str):
    """Cross-entropy or focal loss with optional label smoothing."""
    import torch
    import torch.nn.functional as F

    n_classes = max(int(max(labels)) + 1 if labels else 1, 1)
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (n_classes * counts.astype(np.float64))
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)

    if recipe.focal_loss:

        class FocalLoss(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gamma = recipe.focal_gamma
                self.register_buffer("weight", weight_t)
                self.label_smoothing = recipe.label_smoothing

            def forward(self, logits, targets):
                ce = F.cross_entropy(
                    logits,
                    targets,
                    weight=self.weight,
                    reduction="none",
                    label_smoothing=self.label_smoothing,
                )
                pt = torch.exp(-ce)
                return (((1.0 - pt) ** self.gamma) * ce).mean()

        return FocalLoss()

    return nn.CrossEntropyLoss(weight=weight_t, label_smoothing=recipe.label_smoothing)


def build_train_loader(DataLoader, dataset, cfg, labels: list[int], recipe: TrainRecipe, collate_fn):
    if recipe.balanced_sampler:
        import torch
        from torch.utils.data import WeightedRandomSampler

        n_classes = cfg.n_classes
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes)
        counts = np.maximum(counts, 1)
        sample_weights = [1.0 / counts[y] for y in labels]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )


def set_epoch_lr(optim, cfg, recipe: TrainRecipe, epoch: int) -> None:
    """Warmup + cosine schedule (Phase A); no-op when recipe.lr_cosine is False."""
    if not recipe.lr_cosine:
        return
    base = cfg.lr
    if recipe.lr_warmup_epochs > 0 and epoch <= recipe.lr_warmup_epochs:
        scale = epoch / recipe.lr_warmup_epochs
    else:
        import math

        denom = max(1, cfg.epochs - recipe.lr_warmup_epochs)
        t = (epoch - recipe.lr_warmup_epochs) / denom
        t = min(max(t, 0.0), 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * t))
    for group in optim.param_groups:
        group["lr"] = base * scale
