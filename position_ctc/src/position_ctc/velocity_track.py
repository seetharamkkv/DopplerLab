"""Continuous per-frame velocity, then trapezoid integration from a known p0.

No CTC and no velocity bins. Position is the integral of the predicted
velocity. The first sample is the labeled initial position.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from position_ctc.features import N_MELS


def step_times(times: torch.Tensor) -> torch.Tensor:
    """Positive Δt per step, shape ``(B, T-1)``."""
    dt = times[:, 1:] - times[:, :-1]
    return dt.clamp(min=1e-6)


def integrate_trapezoid(vel: torch.Tensor, times: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    """``p_k = p0 + sum (v_{i-1}+v_i)/2 Δt_i``. ``vel`` is ``(B, T, 2)``."""
    mid = 0.5 * (vel[:, :-1, :] + vel[:, 1:, :])
    steps = mid * step_times(times)[..., None]
    xy = vel.new_zeros(vel.shape)
    xy[:, 0, :] = origin
    xy[:, 1:, :] = origin[:, None, :] + torch.cumsum(steps, dim=1)
    return xy


def displacement(vel: torch.Tensor, times: torch.Tensor, start: torch.Tensor, stop: torch.Tensor) -> torch.Tensor:
    """Trapezoid displacement from frame ``start`` to ``stop``, exclusive of start."""
    batch, _, _ = vel.shape
    out = vel.new_zeros(batch, 2)
    for i in range(batch):
        a = int(start[i])
        b = int(stop[i])
        if b <= a:
            continue
        mid = 0.5 * (vel[i, a:b, :] + vel[i, a + 1 : b + 1, :])
        dt = (times[i, a + 1 : b + 1] - times[i, a:b]).clamp(min=1e-6)
        out[i] = (mid * dt[:, None]).sum(dim=0)
    return out


class VelocityNet(nn.Module):
    """Small CNN–GRU. Each audio frame gets a continuous ``(vx, vy)``."""

    def __init__(self, n_mels: int = N_MELS, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3 * n_mels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(64, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, feat: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(feat).transpose(1, 2)
        hidden, _ = self.rnn(hidden)
        return self.head(self.drop(hidden))


def corner_mask(vel: torch.Tensor, min_deg: float = 12.0) -> torch.Tensor:
    """True on frames where heading jumps by at least ``min_deg``."""
    speed = vel.norm(dim=-1)
    moving = speed > 0.5
    head = torch.atan2(vel[..., 1], vel[..., 0])
    dhead = torch.diff(head, dim=1)
    dhead = (dhead + math.pi) % (2 * math.pi) - math.pi
    jump = dhead.abs() > math.radians(min_deg)
    jump = jump & moving[:, 1:] & moving[:, :-1]
    mask = torch.zeros(vel.shape[:2], dtype=torch.bool, device=vel.device)
    mask[:, 1:] = jump
    mask[:, :-1] = mask[:, :-1] | jump
    return mask


def huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    err = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(dim=-1)
    weight = mask.to(pred.dtype)
    return (err * weight).sum() / weight.sum().clamp(min=1.0)


def velocity_objective(
    pred_v: torch.Tensor,
    times: torch.Tensor,
    state: torch.Tensor,
    lengths: torch.Tensor,
    *,
    spans: tuple[int, ...] = (8, 32, 86, 258),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Velocity Huber + integrated path Huber + multi-duration displacement."""
    valid = torch.arange(pred_v.shape[1], device=pred_v.device)[None, :] < lengths[:, None]
    gt_v = torch.stack((state[..., 1], state[..., 3]), dim=-1)
    gt_xy = torch.stack((state[..., 0], state[..., 2]), dim=-1)
    origin = gt_xy[:, 0, :]
    pred_xy = integrate_trapezoid(pred_v, times, origin)
    corners = corner_mask(gt_v) & valid
    straight = valid & ~corners
    vel_x = huber(pred_v[..., :1], gt_v[..., :1], valid)
    vel_y = huber(pred_v[..., 1:], gt_v[..., 1:], valid)
    vel_all = vel_x + vel_y
    vel_corner = huber(pred_v, gt_v, corners) if bool(corners.any()) else pred_v.new_zeros(())
    vel_straight = huber(pred_v, gt_v, straight)
    path_x = huber(pred_xy[..., :1], gt_xy[..., :1], valid, beta=2.0)
    path_y = huber(pred_xy[..., 1:], gt_xy[..., 1:], valid, beta=2.0)
    path = path_x + path_y
    disp_terms = []
    for span in spans:
        starts = []
        stops = []
        for i, n in enumerate(lengths.tolist()):
            n = int(n)
            if n <= span:
                continue
            a = 0 if n - span <= 1 else int(torch.randint(0, n - span, (1,), device=pred_v.device))
            starts.append((i, a, a + span))
        if not starts:
            continue
        idx = torch.tensor([s[0] for s in starts], device=pred_v.device)
        a = torch.tensor([s[1] for s in starts], device=pred_v.device)
        b = torch.tensor([s[2] for s in starts], device=pred_v.device)
        pred_d = displacement(pred_v[idx], times[idx], a, b)
        true_d = gt_xy[idx, b, :] - gt_xy[idx, a, :]
        disp_terms.append(F.smooth_l1_loss(pred_d, true_d, beta=1.0))
    disp = torch.stack(disp_terms).mean() if disp_terms else pred_v.new_zeros(())
    total = vel_all + path + 0.5 * disp + 0.5 * vel_corner
    parts = {
        "vel": float(vel_all.detach()),
        "vel_x": float(vel_x.detach()),
        "vel_y": float(vel_y.detach()),
        "vel_corner": float(vel_corner.detach()),
        "vel_straight": float(vel_straight.detach()),
        "path": float(path.detach()),
        "path_x": float(path_x.detach()),
        "path_y": float(path_y.detach()),
        "disp": float(disp.detach()),
        "total": float(total.detach()),
    }
    return total, parts
