"""CNN-GRU encoders for CTC tokens and for frame-wise position."""

from __future__ import annotations

import torch
from torch import nn

from position_ctc.alphabet import N_CLASS
from position_ctc.features import N_MELS


class FrameEncoder(nn.Module):
    """Shared CNN-GRU. Each output frame is a 256-d embedding on a 4x coarser grid."""

    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3 * n_mels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.GRU(96, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(x).transpose(1, 2)
        hidden, _ = self.rnn(hidden)
        return self.drop(hidden)


class MotionCTC(nn.Module):
    def __init__(self, n_class: int = N_CLASS, n_mels: int = N_MELS):
        super().__init__()
        n_in = 3 * n_mels
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.GRU(
            96,
            128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(0.15)
        self.proj = nn.Linear(256, n_class)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[0] = -2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(B, C, T)``. Returns logits ``(B, T_enc, n_class)``."""
        hidden = self.conv(x).transpose(1, 2)
        hidden, _ = self.rnn(hidden)
        return self.proj(self.drop(hidden))

    @staticmethod
    def encoded_lengths(raw_lengths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(raw_lengths // 4, min=1)
