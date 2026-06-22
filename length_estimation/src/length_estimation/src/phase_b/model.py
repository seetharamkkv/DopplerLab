"""CNN regressor: pass-by spectrogram (+ speed) -> vehicle length (m)."""

from __future__ import annotations

from length_estimation.config import PhaseBConfig


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Phase B requires PyTorch. Install: pip install -r length_estimation/requirements.txt"
        ) from exc
    return torch, nn


class PassByLengthCNN:
    """
    2-D CNN on (1, n_mels, T) log-mel or SSQ spectrogram.

    Architecture tuned for CPA-centred pass-by clips:
    - Pool mostly along time (vehicle extent is temporal)
    - Optional speed channel fused before regression head
    """

    @staticmethod
    def build(cfg: PhaseBConfig):
        _, nn = _require_torch()
        channels = cfg.hidden_channels

        layers: list = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            # Shrink time axis faster than frequency (pass-by structure is time-dominant)
            layers.append(nn.MaxPool2d(kernel_size=(1, 2) if i < len(channels) - 1 else 2))
            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool2d(1))

        conv = nn.Sequential(*layers)
        fc_in = channels[-1] + (1 if cfg.include_speed else 0)
        head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(64, 1),
        )

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.include_speed = cfg.include_speed
                self.conv = conv
                self.head = head

            def forward(self, x, speed=None):
                import torch

                h = self.conv(x).flatten(1)
                if self.include_speed and speed is not None:
                    h = torch.cat([h, speed.unsqueeze(1)], dim=1)
                return self.head(h).squeeze(-1)

        return _Net()
