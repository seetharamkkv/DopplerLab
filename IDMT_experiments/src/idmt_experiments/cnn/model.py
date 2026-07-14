"""Small CNN for direction-of-travel classification.

REPRODUCIBILITY BASELINE (CNN direction: mel_3class, mel_3class_left, mel_3class_right)
---------------------------------------------------------------------------------
Critical path for published monoaural metrics (vehicle bal. acc 81.5% / 79.3% / 73.6%).
Do not change default behaviour, numerics, or evaluation outputs without re-benchmarking
all three reference runs. Refactoring for maintainability is OK only if metrics stay
bit-identical. New work: separate --run-name or new modules.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

from idmt_experiments.config import DirectionConfig, feature_input_channels


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
        in_ch = feature_input_channels(cfg.feature_type)

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
