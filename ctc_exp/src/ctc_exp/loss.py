"""Speed-bin losses that do not reward predicting the mean / majority class."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ctc_exp.alphabet import N_SPEED, SPEED_TOKENS, speed_bin_index


def class_weights_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights. Unseen bins get weight 0 (not forced)."""
    counts = counts.float()
    present = counts > 0
    n_present = int(present.sum().item())
    w = torch.zeros_like(counts)
    if n_present == 0:
        return w
    n = float(counts.sum().item())
    w[present] = n / (n_present * counts[present])
    return w


def train_speed_counts(clips, train_idx: list[int], device: torch.device) -> torch.Tensor:
    counts = torch.zeros(N_SPEED, device=device)
    for i in train_idx:
        counts[speed_bin_index(clips[i].speed_mps)] += 1
    return counts


def focal_balanced_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_w: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Per-example focal CE with class weights. Not MSE — no pull to the mean speed."""
    logp = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(logp, targets, reduction="none")
    pt = torch.exp(-nll).clamp(min=1e-6)
    w = class_w[targets]
    return (w * (1.0 - pt) ** gamma * nll).mean()


def prior_collapse_penalty(logits: torch.Tensor, train_prior: torch.Tensor) -> torch.Tensor:
    """Inner product of batch-mean softmax with the train class prior.

    High when every clip is predicted as the majority / mean bin.
    """
    p = torch.softmax(logits, dim=-1).mean(dim=0)
    prior = train_prior / train_prior.sum().clamp(min=1e-6)
    return torch.dot(p, prior)


def splice_speed_token(tokens: list[str], speed_idx: int) -> list[str]:
    vtok = SPEED_TOKENS[int(speed_idx)]
    rest = [t for t in tokens if not (t.startswith("V") and t[1:].isdigit())]
    return [vtok, *rest]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(-1)
    return (x * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp(min=1.0)


def centered_traj_huber(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber on time-centered paths, scaled by GT shape RMS.

    Subtracting the per-clip centroid means a constant 'average position'
    does not reduce the loss. Scaling by GT RMS blocks shrinking to zero.
    """
    pc = (pred_xy - _masked_mean(pred_xy, mask)) * mask.unsqueeze(-1)
    gc = (gt_xy - _masked_mean(gt_xy, mask)) * mask.unsqueeze(-1)
    rms = torch.sqrt((gc.pow(2).sum(dim=-1) * mask).sum(dim=1).clamp(min=1e-6))
    scale = rms.view(-1, 1, 1).clamp(min=1.0)
    return F.huber_loss(pc / scale, gc / scale, reduction="mean", delta=delta)


def centered_traj_huber_m(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 2.0,
) -> torch.Tensor:
    """Huber on time-centered paths in metres (no RMS normalization)."""
    pc = (pred_xy - _masked_mean(pred_xy, mask)) * mask.unsqueeze(-1)
    gc = (gt_xy - _masked_mean(gt_xy, mask)) * mask.unsqueeze(-1)
    return F.huber_loss(pc, gc, reduction="mean", delta=delta)


def velocity_huber(
    pred_v: torch.Tensor,
    gt_v: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    scale = 20.0
    m = mask.unsqueeze(-1)
    return F.huber_loss((pred_v / scale) * m, (gt_v / scale) * m, reduction="mean", delta=delta)


def consistency_huber(
    pred_xy: torch.Tensor,
    pred_v: torch.Tensor,
    dt_s: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Finite-difference of xy should match predicted velocity (anti-static paths)."""
    dxy = pred_xy[:, 1:] - pred_xy[:, :-1]
    dt = dt_s.clamp(min=1e-3).view(-1, 1, 1)
    v_fd = dxy / dt
    m = mask[:, 1:]
    return velocity_huber(pred_v[:, 1:], v_fd, m, delta=delta)


def bin_conditioned_speed_residual(
    pred_v: torch.Tensor,
    gt_v: torch.Tensor,
    speed_idx: torch.Tensor,
    bin_centers: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Huber on |v| residual vs bin center; cosine on direction.

    Magnitude target is (||v|| - center), so predicting the global mean
    velocity vector cannot hide inside the bin offset.
    """
    gt_mag = gt_v.norm(dim=-1)
    pred_mag = pred_v.norm(dim=-1)
    center = bin_centers[speed_idx].unsqueeze(1)
    gt_res = gt_mag - center
    pred_res = pred_mag - center
    mag_loss = F.huber_loss(pred_res * mask, gt_res * mask, reduction="mean", delta=1.0)
    gt_dir = gt_v / gt_mag.unsqueeze(-1).clamp(min=1e-3)
    pred_dir = pred_v / pred_mag.unsqueeze(-1).clamp(min=1e-3)
    cosine = (pred_dir * gt_dir).sum(dim=-1)
    dir_loss = ((1.0 - cosine) * mask).sum() / mask.sum().clamp(min=1.0)
    return mag_loss, dir_loss
