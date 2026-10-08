"""Framewise velocity loss, auxiliary CTC, and an anchored trajectory loss.

Velocity classification is cross-entropy plus the expected squared distance
from the continuous label. Frames that have left the clip's opening velocity
are upweighted, because a uniform mean is minimized by the straight-path bin.
CTC is a separate head on the collapsed bin string. Its blank class is 0;
physical bin ``k`` is class ``k + 1``.

The trajectory loss integrates the expected bin velocity with the trapezoid
rule, starting at the labeled first position. That is the same anchor
inference uses. It is differentiable, so a flat velocity cannot hide behind
a decoder. Corner spikes that jump past the transition mask are left out of
the classifier and CTC targets. The path loss still follows the position
label, which does not require the network to emit an illegal bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from position_ctc.alphabet import VEL_MAX_MPS, VEL_MIN_MPS, VEL_STEP_MPS
from position_ctc.bins import N_VEL, collapse_runs
from position_ctc.transitions import DEFAULT_JUMP_MPS

VEL_SCALE_MPS = 10.0
PATH_SCALE_M = 2.0


@dataclass(frozen=True)
class LossWeights:
    dist: float = 1.0
    ctc: float = 0.25
    path: float = 1.0


def expected_distance(
    probs: torch.Tensor,
    v_star: torch.Tensor,
    centers: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """``sum_k P(k) ((c_k - v*) / scale)²``, shape matching ``v_star``."""
    delta = (centers - v_star.unsqueeze(-1)) / float(scale)
    return (probs * delta.square()).sum(dim=-1)


def zero_bin_balanced_nll(
    nll: torch.Tensor,
    bins: torch.Tensor,
    valid: torch.Tensor,
    zero_index: int,
) -> torch.Tensor:
    """Half the loss from the zero-velocity bin, half from every other bin.

    ``nll`` and ``bins`` are ``(B, T)``. ``valid`` drops padding. Within each
    clip, frames whose label is ``zero_index`` and frames whose label is any
    other bin contribute equally. If one of those groups is missing, the mean
    of the remaining frames is used. This is ordinary cross-entropy with a
    different frame weight, not a second task.
    """
    pieces: list[torch.Tensor] = []
    zero_index = int(zero_index)
    for i in range(nll.shape[0]):
        mask = valid[i]
        if int(mask.sum()) < 1:
            continue
        values = nll[i, mask]
        labels = bins[i, mask]
        on_zero = labels == zero_index
        if bool(on_zero.any()) and bool((~on_zero).any()):
            pieces.append(0.5 * values[on_zero].mean() + 0.5 * values[~on_zero].mean())
        else:
            pieces.append(values.mean())
    if not pieces:
        return nll.sum() * 0.0
    return torch.stack(pieces).mean()


def dominant_balanced_nll(nll: torch.Tensor, bins: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Half the loss from the clip's dominant bin, half from every other bin.

    ``nll`` and ``bins`` are ``(B, T)``. ``valid`` selects real frames. The
    dominant bin is the mode of that clip's own labels, computed per axis by
    the caller. An empty group is dropped and the other group is used alone.
    This is a reweighting of framewise cross-entropy, not a different task:
    a predictor that is right on both groups still has the lower loss.
    """
    pieces: list[torch.Tensor] = []
    for i in range(nll.shape[0]):
        mask = valid[i]
        if int(mask.sum()) < 1:
            continue
        values = nll[i, mask]
        labels = bins[i, mask]
        labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
        counts = torch.bincount(labels_cpu, minlength=int(labels_cpu.max()) + 1)
        dominant = int(torch.argmax(counts))
        on_dominant = labels == dominant
        if bool(on_dominant.any()) and bool((~on_dominant).any()):
            pieces.append(0.5 * values[on_dominant].mean() + 0.5 * values[~on_dominant].mean())
        else:
            pieces.append(values.mean())
    if not pieces:
        return nll.sum() * 0.0
    return torch.stack(pieces).mean()


def _bin_index(velocity: torch.Tensor) -> torch.Tensor:
    lo = VEL_MIN_MPS / VEL_STEP_MPS
    hi = VEL_MAX_MPS / VEL_STEP_MPS
    k = torch.round(velocity / VEL_STEP_MPS).clamp(lo, hi)
    return (k - lo).to(torch.long)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def _huber_axes(residual: torch.Tensor) -> torch.Tensor:
    """Sum over axes so an error in ``y`` is not diluted by a correct ``x``."""
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none", beta=1.0).sum(dim=-1)


def turn_frame_weight(gt_v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Upweight frames that have left the clip's opening velocity.

    Most frames of a pass-by sit on one ``(vx, vy)`` bin. A uniform mean lets
    that majority bin own the gradient, so the turn is never fit. Weight is 1
    on the opening fifth and rises to 9 where speed has moved by 4 m/s.
    """
    time = gt_v.shape[1]
    n0 = max(time // 5, 1)
    head = valid[:, :n0].to(dtype=gt_v.dtype)
    denom = head.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    approach = (gt_v[:, :n0] * head.unsqueeze(-1)).sum(dim=1) / denom
    deviation = (gt_v - approach[:, None, :]).norm(dim=-1)
    weight = 1.0 + 4.0 * (deviation / 2.0).clamp(max=2.0)
    return weight * valid.to(dtype=gt_v.dtype)


def reliable_velocity_mask(
    state: torch.Tensor,
    valid: torch.Tensor,
    dt: float,
    jump_mps: float,
    gap_mps: float = 1.0,
) -> torch.Tensor:
    """Frames whose labeled step stays inside the jump limit and matches ``Δp/Δt``.

    A simulator corner that jumps farther than the decoder allows is not a
    classifier target. Both endpoints of a bad step are marked unreliable.
    The position path is still supervised by the trajectory loss.
    """
    vel = torch.stack((state[..., 1], state[..., 3]), dim=-1)
    xy = torch.stack((state[..., 0], state[..., 2]), dim=-1)
    bad = torch.zeros_like(valid)
    if state.shape[1] >= 2:
        label_step = (vel[:, 1:] - vel[:, :-1]).abs().amax(dim=-1)
        implied = (xy[:, 1:] - xy[:, :-1]) / float(dt)
        trap = 0.5 * (vel[:, :-1] + vel[:, 1:])
        gap = (implied - trap).abs().amax(dim=-1)
        over = (label_step > float(jump_mps)) | (gap > float(gap_mps))
        bad[:, 1:] |= over
        bad[:, :-1] |= over
    return valid & ~bad


def integrate_expected(
    vel: torch.Tensor,
    anchor: torch.Tensor,
    valid: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Trapezoid integral of ``vel`` ``(B, T, 2)`` from ``anchor`` ``(B, 1, 2)``.

    The first sample is exactly ``anchor``. Padded steps contribute nothing.
    """
    step = 0.5 * (vel[:, :-1] + vel[:, 1:]) * float(dt)
    ok = (valid[:, :-1] & valid[:, 1:]).unsqueeze(-1).to(dtype=vel.dtype)
    disp = torch.cumsum(step * ok, dim=1)
    return torch.cat((anchor, anchor + disp), dim=1)


class MotionObjective(nn.Module):
    def __init__(
        self,
        weights: LossWeights | None = None,
        jump_mps: float = DEFAULT_JUMP_MPS,
        balance: str = "mean",
        class_weight_x: torch.Tensor | None = None,
        class_weight_y: torch.Tensor | None = None,
    ):
        super().__init__()
        if balance not in ("mean", "dominant", "frequency"):
            raise ValueError(balance)
        self.weights = weights or LossWeights()
        self.jump_mps = float(jump_mps)
        self.balance = balance
        centers = torch.arange(N_VEL, dtype=torch.float32) * VEL_STEP_MPS + VEL_MIN_MPS
        self.register_buffer("centers", centers)
        ones = torch.ones(N_VEL, dtype=torch.float32)
        self.register_buffer("class_weight_x", ones if class_weight_x is None else class_weight_x.float())
        self.register_buffer("class_weight_y", ones.clone() if class_weight_y is None else class_weight_y.float())
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        state: torch.Tensor,
        lengths: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``state`` is ``(B, T, 4)`` with columns ``x, vx, y, vy`` in meters and m/s.

        The integration anchor is ``state[:, 0, (x, y)]``, the same initial
        position passed into the network.
        """
        valid = torch.arange(state.shape[1], device=state.device)[None, :] < lengths[:, None]
        vx = state[..., 1]
        vy = state[..., 3]
        gt_xy = torch.stack((state[..., 0], state[..., 2]), dim=-1)
        gt_v = torch.stack((vx, vy), dim=-1)
        reliable = reliable_velocity_mask(state, valid, dt, self.jump_mps)
        emphasis = turn_frame_weight(gt_v, reliable)
        if self.balance == "dominant":
            ce = 0.5 * (
                dominant_balanced_nll(self._ce(pred["vel_x"], vx), _bin_index(vx), valid)
                + dominant_balanced_nll(self._ce(pred["vel_y"], vy), _bin_index(vy), valid)
            )
            dist = 0.5 * (
                _masked_mean(self._dist(pred["vel_x"], vx), valid)
                + _masked_mean(self._dist(pred["vel_y"], vy), valid)
            )
        elif self.balance == "frequency":
            ce = 0.5 * (
                _masked_mean(self._ce(pred["vel_x"], vx, self.class_weight_x), valid)
                + _masked_mean(self._ce(pred["vel_y"], vy, self.class_weight_y), valid)
            )
            dist = 0.5 * (
                _masked_mean(self._dist(pred["vel_x"], vx), valid)
                + _masked_mean(self._dist(pred["vel_y"], vy), valid)
            )
        else:
            ce, dist = self._velocity(pred["vel_x"], pred["vel_y"], vx, vy, emphasis)
        w = self.weights
        if w.ctc != 0.0:
            ctc = self._ctc(pred["ctc_x"], vx, lengths, reliable) + self._ctc(pred["ctc_y"], vy, lengths, reliable)
        else:
            ctc = ce.new_zeros(())
        expected = self._expected_velocity(pred["vel_x"], pred["vel_y"])
        path = self._integrated_path(expected, gt_xy, valid, dt) if w.path != 0.0 else ce.new_zeros(())
        total = ce + w.dist * dist + w.ctc * ctc + w.path * path
        parts = {
            "ce": ce.detach().item(),
            "dist": dist.detach().item(),
            "ctc": ctc.detach().item(),
            "path": path.detach().item(),
            "total": total.detach().item(),
        }
        return total, parts

    def _velocity(
        self,
        logit_x: torch.Tensor,
        logit_y: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ce = _masked_mean(self._ce(logit_x, vx), valid) + _masked_mean(self._ce(logit_y, vy), valid)
        dist = _masked_mean(self._dist(logit_x, vx), valid) + _masked_mean(self._dist(logit_y, vy), valid)
        return 0.5 * ce, 0.5 * dist

    def _ce(self, logits: torch.Tensor, velocity: torch.Tensor, class_weight: torch.Tensor | None = None) -> torch.Tensor:
        target = _bin_index(velocity)
        flat = F.cross_entropy(
            logits.reshape(-1, self.n_vel),
            target.reshape(-1),
            weight=class_weight,
            reduction="none",
        )
        return flat.view(velocity.shape)

    def _dist(self, logits: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        return expected_distance(probs, velocity, self.centers, VEL_SCALE_MPS)

    @property
    def n_vel(self) -> int:
        return int(self.centers.shape[0])

    def _expected_velocity(self, logit_x: torch.Tensor, logit_y: torch.Tensor) -> torch.Tensor:
        vx = logit_x.softmax(dim=-1) @ self.centers
        vy = logit_y.softmax(dim=-1) @ self.centers
        return torch.stack((vx, vy), dim=-1)

    def _ctc(
        self,
        logits: torch.Tensor,
        velocity: torch.Tensor,
        lengths: torch.Tensor,
        reliable: torch.Tensor,
    ) -> torch.Tensor:
        targets: list[torch.Tensor] = []
        tgt_len: list[int] = []
        bins = _bin_index(velocity)
        for i in range(velocity.shape[0]):
            n = int(lengths[i])
            ids = bins[i, :n].detach().cpu().numpy().copy()
            keep = reliable[i, :n].detach().cpu().numpy()
            if n > 0:
                last = int(ids[0])
                for t in range(n):
                    if keep[t]:
                        last = int(ids[t])
                    else:
                        # A corner past the mask is not a CTC symbol. Hold the
                        # last legal bin so the target string stays physical.
                        ids[t] = last
            collapsed = collapse_runs(ids)
            # Blank is class 0. Bin k, including the -30 m/s bin, is class k + 1.
            targets.append(torch.as_tensor(collapsed + 1, dtype=torch.long))
            tgt_len.append(int(collapsed.size))
        target = torch.cat(targets).to(logits.device)
        if int(target.min()) < 1:
            raise RuntimeError("CTC target used the blank class as a velocity")
        in_len = lengths.to(dtype=torch.long, device=logits.device)
        tgt = torch.tensor(tgt_len, dtype=torch.long, device=logits.device)
        logp = logits.log_softmax(dim=-1).transpose(0, 1)
        # CTCLoss sums over time. Divide by the frame count so this matches
        # the per-frame classifier losses.
        nll = self.ctc(logp, target, in_len, tgt)
        return nll / in_len.to(nll.dtype).mean().clamp(min=1.0)

    def _integrated_path(
        self,
        vel: torch.Tensor,
        gt_xy: torch.Tensor,
        valid: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Trapezoid integral against the labeled path.

        The anchor is the true first position, matching inference and the
        position given to the encoder. A late turn still moves the end of
        the path, so a flat ``vy`` cannot hide there.
        """
        pred = integrate_expected(vel, gt_xy[:, :1], valid, dt)
        return _masked_mean(_huber_axes((pred - gt_xy) / PATH_SCALE_M), valid)
