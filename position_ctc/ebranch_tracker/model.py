"""Self-contained E-Branchformer-style encoder with continuous sinusoidal positions.

Implements macaron FFNs, parallel attention/cgMLP branches, depthwise enhanced
merging and final normalization. Uses absolute sinusoidal positions rather than
ESPnet relative attention. No ESPnet/NeMo checkpoint compatibility is claimed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def clear_padding(x, valid):
    return x.masked_fill(~valid.unsqueeze(-1), 0)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim),
                                 nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class CGMLP(nn.Module):
    def __init__(self, dim, kernel, dropout):
        super().__init__()
        inner = 2 * dim
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 2 * inner)
        self.gate_norm = nn.LayerNorm(inner)
        self.depthwise = nn.Conv1d(inner, inner, kernel, padding=kernel // 2, groups=inner)
        self.project = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        x = F.gelu(self.expand(self.norm(x)))
        a, b = x.chunk(2, dim=-1)
        b = clear_padding(self.gate_norm(b), valid)
        b = self.depthwise(b.transpose(1, 2)).transpose(1, 2)
        return clear_padding(self.dropout(self.project(a * b)), valid)


class EBranchformerBlock(nn.Module):
    def __init__(self, dim, heads, kernel, dropout):
        super().__init__()
        self.ff1, self.ff2 = FeedForward(dim, dropout), FeedForward(dim, dropout)
        self.att_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.local = CGMLP(dim, kernel, dropout)
        self.merge_conv = nn.Conv1d(2 * dim, 2 * dim, 3, padding=1, groups=2 * dim)
        self.merge = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, valid):
        x = clear_padding(x + 0.5 * self.ff1(x), valid)
        a = self.att_norm(x)
        a, _ = self.attention(a, a, a, key_padding_mask=~valid, need_weights=False)
        branches = clear_padding(torch.cat((a, self.local(x, valid)), dim=-1), valid)
        merged = branches + self.merge_conv(branches.transpose(1, 2)).transpose(1, 2)
        x = clear_padding(x + self.dropout(self.merge(merged)), valid)
        return clear_padding(self.final_norm(x + 0.5 * self.ff2(x)), valid)


class Tracker(nn.Module):
    def __init__(self, n_bins, hidden=128, layers=6, heads=4, kernel=15,
                 dropout=0.1, encoder='ebranchformer', **unused):
        super().__init__()
        if hidden % heads or hidden % 2 or kernel % 2 != 1 or layers < 1:
            raise ValueError('hidden must be even/divisible by heads, kernel odd, layers >= 1')
        self.encoder_type = encoder
        # LayerNorm is per time-frequency point: padding never changes valid-frame statistics.
        self.conv1 = nn.Conv2d(1, 32, 5, stride=(2, 2), padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1)
        self.norm1, self.norm2 = nn.LayerNorm(32), nn.LayerNorm(64)
        self.project = nn.Linear(64 * 8, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        if encoder == 'ebranchformer':
            self.blocks = nn.ModuleList(EBranchformerBlock(hidden, heads, kernel, dropout)
                                        for _ in range(layers))
        elif encoder == 'gru':
            self.gru = nn.GRU(hidden, hidden // 2, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=dropout)
        else:
            raise ValueError(encoder)
        self.head = nn.Linear(hidden, 2 * n_bins)
        self.n_bins = n_bins

    def forward(self, features, lengths):
        # features [B, T, F], label-aligned before entering the model.
        # 1025 bins become 257 after the two stride-2 convolutions, and MPS
        # adaptive pooling requires that axis to divide the output size.
        # Dropping Nyquist yields 1024 -> 512 -> 256, which divides by 8.
        if features.shape[-1] > 1024:
            features = features[..., :1024]
        batch, total, _ = features.shape
        valid = torch.arange(total, device=features.device)[None, :] < lengths[:, None]
        x = features.transpose(1, 2).unsqueeze(1)
        x = self.conv1(x).permute(0, 2, 3, 1)
        x = F.silu(self.norm1(x)).permute(0, 3, 1, 2)
        low_lengths = (lengths + 1) // 2
        low_valid = torch.arange(x.shape[-1], device=x.device)[None, :] < low_lengths[:, None]
        x = x.masked_fill(~low_valid[:, None, None, :], 0)
        x = self.conv2(x).permute(0, 2, 3, 1)
        x = F.silu(self.norm2(x)).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (8, x.shape[-1]))
        x = x.permute(0, 3, 1, 2).flatten(2)
        x = clear_padding(self.input_norm(self.project(x)), low_valid)
        if self.encoder_type == 'ebranchformer':
            pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
            rate = torch.exp(torch.arange(0, x.shape[-1], 2, device=x.device,
                                          dtype=x.dtype) * (-math.log(10000) / x.shape[-1]))
            pe = torch.stack((torch.sin(pos * rate), torch.cos(pos * rate)), dim=-1).flatten(1)
            x = clear_padding(x + pe[None], low_valid)
            for block in self.blocks:
                x = block(x, low_valid)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(x, low_lengths.cpu(),
                                                       batch_first=True, enforce_sorted=False)
            packed, _ = self.gru(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True,
                                                   total_length=x.shape[1])
        # Stride-2 centers correspond to input indices 0,2,4,...; don't use
        # generic interpolate endpoints (which shifts odd/even-length sequences).
        index = torch.arange(total, device=x.device)
        left = (index // 2)[None].expand(batch, -1)
        right = torch.minimum(left + 1, low_lengths[:, None] - 1)
        left = torch.minimum(left, low_lengths[:, None] - 1)
        left_x = torch.gather(x, 1, left[..., None].expand(-1, -1, x.shape[-1]))
        right_x = torch.gather(x, 1, right[..., None].expand(-1, -1, x.shape[-1]))
        alpha = (index % 2).to(x.dtype)[None, :, None] * 0.5
        x = clear_padding(left_x * (1 - alpha) + right_x * alpha, valid)
        return self.head(x).reshape(batch, total, 2, self.n_bins)
