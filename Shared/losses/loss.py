import torch
import torch.nn as nn
from Shared.features.extraction import MAX_SPEED_MPS

_ce    = nn.CrossEntropyLoss(label_smoothing=0.05)
_huber = nn.SmoothL1Loss(beta=0.5)

def compute_loss(p_hat, s_hat, d_hat, p_gt, s_gt, d_gt, max_dist):
    loss_path  = _ce(p_hat, p_gt)
    loss_speed = _huber(s_hat, s_gt / MAX_SPEED_MPS)
    loss_dist  = _huber(d_hat, torch.log1p(d_gt) / np.log1p(max_dist))
    return loss_path + 2.5 * loss_speed + 1.5 * loss_dist
