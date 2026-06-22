"""Small CNN for direction-of-travel classification."""

from __future__ import annotations

from idmt_experiments.config import DirectionConfig


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Direction CNN requires PyTorch. Install: pip install -r IDMT_experiments/requirements.txt"
        ) from exc
    return torch, nn


class DirectionCNN:
    @staticmethod
    def build(cfg: DirectionConfig):
        _, nn = _require_torch()
        channels = cfg.hidden_channels
        in_ch = 2 if cfg.feature_type == "stereo_mel" else 1

        layers: list = []
        for i, out_ch in enumerate(channels):
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d(1))

        conv = nn.Sequential(*layers)
        head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(64, cfg.n_classes),
        )

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = conv
                self.head = head

            def forward(self, x):
                return self.head(self.conv(x))

        return _Net()
