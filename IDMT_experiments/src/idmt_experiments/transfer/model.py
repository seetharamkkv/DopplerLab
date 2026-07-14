"""Deep mel CNN (PANNs-inspired backbone for Phase B transfer experiments)."""

from __future__ import annotations

from idmt_experiments.config import DirectionConfig, feature_input_channels


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Requires PyTorch.") from exc
    return torch, nn


def build_deep_mel_cnn(cfg: DirectionConfig):
    """Deeper residual mel CNN (~PANNs-scale depth, trainable from scratch on CPU)."""
    torch, nn = _require_torch()
    in_ch = feature_input_channels(cfg.feature_type)
    channels = (64, 128, 256, 512)

    class ResBlock(nn.Module):
        def __init__(self, ch: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            r = x
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return self.relu(x + r)

    layers: list = []
    for i, out_ch in enumerate(channels):
        layers.extend(
            [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        )
        layers.append(ResBlock(out_ch))
        if i < len(channels) - 1:
            layers.append(nn.MaxPool2d(2))
        in_ch = out_ch
    layers.append(nn.AdaptiveAvgPool2d(1))

    conv = nn.Sequential(*layers)
    embed = channels[-1]
    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(embed, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(cfg.dropout),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(cfg.dropout * 0.5),
        nn.Linear(128, cfg.n_classes),
    )

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = conv
            self.head = head

        def forward(self, x):
            return self.head(self.conv(x))

        def embed(self, x):
            return self.conv(x).flatten(1)

    return _Net()


def build_model(cfg: DirectionConfig, backbone: str = "deep_mel_cnn"):
    if backbone == "deep_mel_cnn":
        return build_deep_mel_cnn(cfg)
    raise ValueError(f"Unknown backbone: {backbone!r}")
