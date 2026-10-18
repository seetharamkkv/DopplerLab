"""CNN and bidirectional LSTM conditioned on the supplied initial position.

Each axis has its own framewise velocity classifier and its own CTC head.
CTC blank is class 0 and is not a physical velocity bin. There is no
position head and no bearing/range head. The trajectory is produced by
integrating velocity from the same initial position the encoder receives.
That position is the canonical-frame ``(x, y)`` in meters, with the
microphone at the origin. Predicted position is not fed back in.
"""

from __future__ import annotations

from pathlib import Path

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


class FullMotionNet(nn.Module):
    """Shared encoder at the audio frame rate, conditioned on the initial position.

    Frequency is downsampled and time is not. ``p0`` is concatenated onto every
    frame before the bidirectional LSTM. Heads read the supervised frames; a
    trailing audio tail may still sit in the LSTM state.
    """

    def __init__(self, n_mels: int = N_MELS, n_vel: int = N_VEL, dropout: float = 0.1, origin_dim: int = 16):
        super().__init__()
        self.n_mels = int(n_mels)
        self.n_vel = int(n_vel)
        self.origin_dim = int(origin_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=(3, 5), stride=(2, 1), padding=(1, 2)),
            nn.ReLU(),
        )
        self.proj = nn.Linear(64 * 6, 96)
        self.origin = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, origin_dim),
        )
        self.rnn = nn.LSTM(96 + origin_dim, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.vel_x = nn.Linear(256, n_vel)
        self.vel_y = nn.Linear(256, n_vel)
        self.ctc_x = nn.Linear(256, n_vel + 1)
        self.ctc_y = nn.Linear(256, n_vel + 1)
        self.pos = nn.Linear(256, 2)
        self.acc = nn.Linear(256, 2)
        self.register_buffer("origin_scale", torch.tensor(50.0))
        self.reset_new_layers()

    def reset_new_layers(self) -> None:
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)
        for module in self.origin.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        for layer in (self.vel_x, self.vel_y, self.ctc_x, self.ctc_y, self.pos, self.acc):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
        self.ctc_x.bias.data[0] = 2.0
        self.ctc_y.bias.data[0] = 2.0

    def forward(self, feat: torch.Tensor, lengths: torch.Tensor, origin: torch.Tensor) -> dict[str, torch.Tensor]:
        if feat.shape[1] != 3 * self.n_mels:
            raise ValueError("expected log-mel and phase channels")
        if origin.shape != (feat.shape[0], 2):
            raise ValueError("origin must be (batch, 2) meters")
        batch, _, time = feat.shape
        image = feat.reshape(batch, 3, self.n_mels, time)
        hidden = self.cnn(image)
        if hidden.shape[-1] != time:
            raise RuntimeError(f"CNN changed the frame count from {time} to {hidden.shape[-1]}")
        flat = hidden.permute(0, 3, 1, 2).reshape(batch, time, -1)
        frames = self.proj(flat)
        scale = self.origin_scale.to(device=origin.device, dtype=frames.dtype).clamp(min=1e-3)
        cond = self.origin(origin.to(dtype=frames.dtype) / scale)
        packed_in = torch.cat((frames, cond[:, None, :].expand(-1, time, -1)), dim=-1)
        lengths = lengths.to(device=feat.device, dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(
            packed_in, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=time)
        hidden = self.drop(hidden)
        return {
            "vel_x": self.vel_x(hidden),
            "vel_y": self.vel_y(hidden),
            "ctc_x": self.ctc_x(hidden),
            "ctc_y": self.ctc_y(hidden),
            "disp": self.pos(hidden),
            "acc": self.acc(hidden),
        }


class CoreMotionNet(nn.Module):
    """Audio features, a temporal 1D CNN, a BiLSTM, and separate velocity heads.

    Frequency bins are the convolution channels. Two stride-2 pools shorten
    the sequence the LSTM reads; its states are linearly restored to the
    spectrogram clock, so each supervised frame keeps its own distribution.
    There is no position, acceleration, bearing, or initial-position head.
    The supplied ``(x0, y0)`` is used only when the decoded velocities are
    integrated.
    """

    def __init__(self, n_mels: int = N_MELS, n_vel: int = N_VEL, dropout: float = 0.1):
        super().__init__()
        self.n_vel = int(n_vel)
        self.time_stride = 4
        self.conv = nn.Sequential(
            nn.Conv1d(3 * n_mels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.LSTM(96, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.vel_x = nn.Linear(256, n_vel)
        self.vel_y = nn.Linear(256, n_vel)
        self.ctc_x = nn.Linear(256, n_vel + 1)
        self.ctc_y = nn.Linear(256, n_vel + 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.conv.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        for layer in (self.vel_x, self.vel_y, self.ctc_x, self.ctc_y):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
        self.ctc_x.bias.data[0] = 2.0
        self.ctc_y.bias.data[0] = 2.0

    def forward(self, feat: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """``feat`` is ``(B, C, T)``. ``lengths`` count real audio frames, including any tail."""
        if feat.ndim != 3:
            raise ValueError("features must be (batch, channels, time)")
        lengths = lengths.to(device=feat.device, dtype=torch.long)
        hidden = self.conv(feat).transpose(1, 2)
        if hidden.size(1) < 1:
            raise RuntimeError("spectrogram shorter than the encoder")
        enc_len = torch.clamp(lengths // self.time_stride, min=1)
        enc_len = torch.minimum(enc_len, torch.full_like(enc_len, hidden.size(1)))
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden, enc_len.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=hidden.size(1))
        frame = _upsample_valid(self.drop(hidden), enc_len, lengths)
        return {
            "vel_x": self.vel_x(frame),
            "vel_y": self.vel_y(frame),
            "ctc_x": self.ctc_x(frame),
            "ctc_y": self.ctc_y(frame),
        }


def load_epoch38(model: FullMotionNet, path: str | Path, origin_scale: float) -> dict[str, list[str]]:
    """Copy shape-compatible epoch-38 weights. New layers stay at their explicit init."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    parent = saved["model"]
    report: dict[str, list[str]] = {"copied": [], "partial": [], "new": [], "skipped": []}
    own = model.state_dict()
    for key in ("vel_x.weight", "vel_x.bias", "vel_y.weight", "vel_y.bias", "ctc_x.weight", "ctc_x.bias", "ctc_y.weight", "ctc_y.bias", "pos.weight", "pos.bias", "acc.weight", "acc.bias"):
        if own[key].shape != parent[key].shape:
            raise RuntimeError(f"{key} shape {tuple(own[key].shape)} != epoch 38 {tuple(parent[key].shape)}")
        own[key].copy_(parent[key])
        report["copied"].append(key)
    for suffix in ("weight_hh_l0", "weight_hh_l0_reverse", "bias_ih_l0", "bias_hh_l0", "bias_ih_l0_reverse", "bias_hh_l0_reverse"):
        key = f"rnn.{suffix}"
        if own[key].shape != parent[key].shape:
            raise RuntimeError(key)
        own[key].copy_(parent[key])
        report["copied"].append(key)
    for suffix in ("weight_ih_l0", "weight_ih_l0_reverse"):
        key = f"rnn.{suffix}"
        src = parent[key]
        dst = own[key]
        if src.shape[0] != dst.shape[0] or src.shape[1] > dst.shape[1]:
            raise RuntimeError(key)
        dst[:, : src.shape[1]].copy_(src)
        nn.init.normal_(dst[:, src.shape[1] :], std=0.02)
        report["partial"].append(f"{key} columns 0:{src.shape[1]} copied; new position columns initialized N(0, 0.02)")
    model.load_state_dict(own)
    model.origin_scale.fill_(float(origin_scale))
    report["new"] = [
        "cnn.*",
        "proj.*",
        "origin.*",
        f"origin inputs are p0 / {float(origin_scale):.4f} m",
    ]
    report["skipped"] = [
        "epoch 38 conv.* temporally pooled mel-mixed features and does not match the frequency CNN",
    ]
    return report
