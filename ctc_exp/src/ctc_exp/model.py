"""CNN-GRU CTC encoder plus a class-balanced speed head."""

from __future__ import annotations

import torch
from torch import nn

from ctc_exp.alphabet import BLANK_ID, N_CLASS, N_SPEED, SPEED_BINS_MPS


class CtcKinematics(nn.Module):
    """Shared time encoder: CTC token stream + attention-pooled speed bins.

    Speed is a classifier, not a regressor, so L2-to-the-mean is not in the graph.
    """

    def __init__(self, n_mels: int = 128, n_class: int = N_CLASS, n_speed: int = N_SPEED):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        hidden = 384
        self.rnn = nn.GRU(
            128,
            hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.proj = nn.Linear(hidden, n_class)
        self.attn = nn.Linear(hidden, 1)
        self.speed_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_speed),
        )
        self.xy_head = nn.Linear(hidden, 2)
        self.bin_emb = nn.Embedding(n_speed, 32)
        self.vel_mlp = nn.Sequential(
            nn.Linear(hidden + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )
        self.register_buffer(
            "bin_centers",
            torch.tensor([float(v) for v in SPEED_BINS_MPS], dtype=torch.float32),
        )

    def _velocity_from_bins(self, h: torch.Tensor, speed_idx: torch.Tensor) -> torch.Tensor:
        """v = (bin_center + residual) * unit_dir. Residual is tanh-scaled to ±2.5 m/s (half bin)."""
        b, t, _ = h.shape
        emb = self.bin_emb(speed_idx).unsqueeze(1).expand(b, t, -1)
        raw = self.vel_mlp(torch.cat([h, emb], dim=-1))
        resid = torch.tanh(raw[..., 0]) * 2.5
        mag = self.bin_centers[speed_idx].unsqueeze(1) + resid
        mag = mag.clamp(min=0.5)
        direction = raw[..., 1:3]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return mag.unsqueeze(-1) * direction

    def forward(
        self,
        x: torch.Tensor,
        raw_lengths: torch.Tensor | None = None,
        speed_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.transpose(1, 2)
        h, _ = self.rnn(h)
        ctc_logits = self.proj(h)
        scores = self.attn(h).squeeze(-1)
        if raw_lengths is not None:
            enc = self.encoded_lengths(raw_lengths)
            t = h.size(1)
            ar = torch.arange(t, device=h.device).unsqueeze(0)
            scores = scores.masked_fill(ar >= enc.unsqueeze(1), -1e9)
        w = torch.softmax(scores, dim=1)
        pooled = torch.sum(h * w.unsqueeze(-1), dim=1)
        speed_logits = self.speed_head(pooled)
        if speed_idx is None:
            speed_idx = speed_logits.argmax(dim=-1)
        xy = self.xy_head(h)
        vel = self._velocity_from_bins(h, speed_idx)
        return ctc_logits, speed_logits, xy, vel

    @staticmethod
    def encoded_lengths(raw_lengths: torch.Tensor) -> torch.Tensor:
        t = raw_lengths // 2
        t = t // 2
        return torch.clamp(t, min=1)


def greedy_decode(logits: torch.Tensor) -> list[int]:
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    out: list[int] = []
    prev = None
    for i in ids:
        if i == BLANK_ID:
            prev = i
            continue
        if i != prev:
            out.append(int(i))
        prev = i
    return out
