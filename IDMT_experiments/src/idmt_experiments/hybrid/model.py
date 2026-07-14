"""CNN mel backbone with physics feature conditioning (late fusion at conv embedding)."""

from __future__ import annotations

from idmt_experiments.config import HybridConfig
from idmt_experiments.physics.features import feature_names


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Hybrid model requires PyTorch. Install: pip install -r IDMT_experiments/requirements.txt"
        ) from exc
    return torch, nn


def n_physics_features(cfg: HybridConfig) -> int:
    return len(feature_names(cfg.to_physics_config()))


class HybridDirectionCNN:
    @staticmethod
    def build(cfg: HybridConfig):
        torch, nn = _require_torch()
        channels = cfg.hidden_channels
        in_ch = 2 if cfg.feature_type == "stereo_mel" else 1
        n_phys = n_physics_features(cfg)
        embed_dim = channels[-1]

        conv_layers: list = []
        for out_ch in channels:
            conv_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            conv_layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        conv_layers.append(nn.AdaptiveAvgPool2d(1))
        conv = nn.Sequential(*conv_layers)

        physics_mlp = nn.Sequential(
            nn.Linear(n_phys, cfg.physics_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.physics_dropout),
        )

        fused_dim = embed_dim + cfg.physics_embed_dim
        head = nn.Sequential(
            nn.Linear(fused_dim, 128),
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
                self.physics_mlp = physics_mlp
                self.head = head

            def forward(self, x_mel, x_physics):
                z_cnn = self.conv(x_mel).flatten(1)
                z_phys = self.physics_mlp(x_physics)
                return self.head(torch.cat([z_cnn, z_phys], dim=1))

        return _Net()
