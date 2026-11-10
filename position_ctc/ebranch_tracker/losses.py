import torch
from torch.nn import functional as F


def objective(logits, batch, bins, corner_weight=3., ordinal_weight=0.2,
              position_weight=0.):
    velocity = batch['velocity']
    active = batch['valid'] & batch['moving']
    width = bins[1] - bins[0]
    # Soft adjacent-bin targets avoid rounding all near-bin velocities to integers.
    fractional = ((velocity - bins[0]) / width).clamp(0, len(bins) - 1)
    low = fractional.floor().long()
    high = (low + 1).clamp_max(len(bins) - 1)
    blend = fractional - low
    target = torch.zeros_like(logits)
    target.scatter_add_(-1, low.unsqueeze(-1), (1 - blend).unsqueeze(-1))
    target.scatter_add_(-1, high.unsqueeze(-1), blend.unsqueeze(-1))
    logp = logits.log_softmax(-1)
    ce = -(target * logp).sum(-1).mean(-1)
    ordinal = (logp.exp().cumsum(-1)[..., :-1] - target.cumsum(-1)[..., :-1]).square().mean((-1, -2))
    angle = torch.atan2(velocity[..., 1], velocity[..., 0])
    change = torch.atan2(torch.sin(angle[:, 1:] - angle[:, :-1]),
                         torch.cos(angle[:, 1:] - angle[:, :-1])).abs()
    corners = (change >= 12 * torch.pi / 180) & active[:, 1:] & active[:, :-1]
    corner_frames = torch.zeros_like(active)
    corner_frames[:, 1:] |= corners
    corner_frames[:, :-1] |= corners
    # A 5-frame neighborhood around each sharp heading change.
    corner_frames = F.max_pool1d(corner_frames.float().unsqueeze(1), 5,
                                 stride=1, padding=2).squeeze(1) > 0
    weights = active.float() * torch.where(corner_frames, corner_weight, 1.)
    classification = ((ce + ordinal_weight * ordinal) * weights).sum() / weights.sum().clamp_min(1)
    expected = (logp.exp() * bins).sum(-1)
    position_loss = logits.new_zeros(())
    if position_weight:
        dt = (batch['times'][:, 1:] - batch['times'][:, :-1]).to(logits.dtype)
        dt = torch.where(batch['valid'][:, 1:], dt, 0)
        increments = (expected[:, 1:] + expected[:, :-1]) * 0.5 * dt[..., None]
        displacement = torch.cat((torch.zeros_like(expected[:, :1]), increments.cumsum(1)), dim=1)
        pred = batch['position'][:, :1] + displacement
        err = F.smooth_l1_loss(pred, batch['position'], reduction='none').mean(-1)
        position_loss = (err * weights).sum() / weights.sum().clamp_min(1)
    loss = classification + position_weight * position_loss
    return loss, {'classification': float(classification.detach()),
                  'position': float(position_loss.detach())}
