"""CNN and bidirectional LSTM conditioned on the supplied initial position.

Each axis has its own framewise velocity classifier and its own CTC head.
CTC blank is class 0 and is not a physical velocity bin. There is no
position head and no bearing/range head. The trajectory is produced by
integrating velocity from the same initial position the encoder receives.
That position is the canonical-frame ``(x, y)`` in meters, with the
microphone at the origin. Predicted position is not fed back in.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from position_ctc.bins import N_VEL
from position_ctc.features import N_MELS

# Pass-bys in this set stay inside about 90 m. Dividing by this scale puts
# the conditioning vector near order 1 without changing the meter frame
# used by the labels or the integrator.
ORIGIN_SCALE_M = 50.0


def initial_position(state: torch.Tensor) -> torch.Tensor:
    """First-frame ``(x, y)`` in meters. Columns of ``state`` are ``x, vx, y, vy``."""
    return torch.stack((state[:, 0, 0], state[:, 0, 2]), dim=-1)


def _upsample_valid(hidden: torch.Tensor, enc_lengths: torch.Tensor, out_lengths: torch.Tensor) -> torch.Tensor:
    """Linear upsample of each item's valid encoder frames onto its audio frames."""
    batch, _, channels = hidden.shape
    out_time = int(out_lengths.max())
    out = hidden.new_zeros(batch, out_time, channels)
    for i in range(batch):
        n_enc = int(enc_lengths[i])
        n_out = int(out_lengths[i])
        if n_enc < 1 or n_out < 1:
            continue
        piece = hidden[i, :n_enc].transpose(0, 1).unsqueeze(0)
        up = F.interpolate(piece, size=n_out, mode="linear", align_corners=True)
        out[i, :n_out] = up.squeeze(0).transpose(0, 1)
    return out


class MotionNet(nn.Module):
    """Offline pass-by encoder. Output frames match the spectrogram clock.

    Two stride-2 pools make the recurrent features ``T/4`` long. They are
    upsampled back to one velocity distribution per audio frame, so the
    network cannot emit a single clip vector repeated over time.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        n_vel: int = N_VEL,
        dropout: float = 0.1,
        temporal_pool: bool = True,
    ):
        super().__init__()
        self.n_vel = n_vel
        self.temporal_pool = bool(temporal_pool)
        self.time_stride = 4 if self.temporal_pool else 1

        def _pool() -> nn.Module:
            return nn.MaxPool1d(2) if self.temporal_pool else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv1d(3 * n_mels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            _pool(),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            _pool(),
        )
        self.origin = nn.Linear(2, 96)
        self.rnn = nn.LSTM(96, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.vel_x = nn.Linear(256, n_vel)
        self.vel_y = nn.Linear(256, n_vel)
        self.ctc_x = nn.Linear(256, n_vel + 1)
        self.ctc_y = nn.Linear(256, n_vel + 1)
        for layer in (self.vel_x, self.vel_y, self.ctc_x, self.ctc_y):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.origin.weight, std=0.02)
        nn.init.zeros_(self.origin.bias)
        # Prefer blank at initialization so CTC starts from a short alignment.
        self.ctc_x.bias.data[0] = 2.0
        self.ctc_y.bias.data[0] = 2.0

    def forward(
        self,
        feat: torch.Tensor,
        lengths: torch.Tensor,
        origin: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``feat`` is ``(B, C, T)``. ``origin`` is ``(B, 2)`` in meters.

        ``origin`` is the supplied vehicle position at the first valid frame,
        in the same canonical frame as the state labels. Lengths count real
        frames, not the pad. Returns per-frame logits. Padded positions stay
        at the zero-embedding bias.
        """
        if origin.ndim != 2 or origin.shape[-1] != 2 or origin.shape[0] != feat.shape[0]:
            raise ValueError("origin must be (batch, 2) meters in the canonical frame")
        lengths = lengths.to(device=feat.device, dtype=torch.long)
        hidden = self.conv(feat).transpose(1, 2)
        if hidden.size(1) < 1:
            raise RuntimeError("spectrogram shorter than the encoder")
        shift = self.origin(origin.to(device=hidden.device, dtype=hidden.dtype) / ORIGIN_SCALE_M)
        hidden = hidden + shift[:, None, :]
        enc_len = torch.clamp(lengths // self.time_stride, min=1)
        enc_len = torch.minimum(enc_len, torch.full_like(enc_len, hidden.size(1)))
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden,
            enc_len.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=hidden.size(1),
        )
        hidden = self.drop(hidden)
        frame = _upsample_valid(hidden, enc_len, lengths)
        return {
            "vel_x": self.vel_x(frame),
            "vel_y": self.vel_y(frame),
            "ctc_x": self.ctc_x(frame),
            "ctc_y": self.ctc_y(frame),
            "enc_lengths": enc_len,
        }
