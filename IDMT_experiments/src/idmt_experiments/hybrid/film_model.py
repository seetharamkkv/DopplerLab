"""FiLM-conditioned hybrid model (Phase D)."""

from __future__ import annotations

from idmt_experiments.config import HybridConfig
from idmt_experiments.physics.features import feature_names


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Requires PyTorch.") from exc
    return torch, nn


def n_physics_features(cfg: HybridConfig) -> int:
    return len(feature_names(cfg.to_physics_config()))


class FiLMHybridDirectionCNN:
    @staticmethod
    def build(cfg: HybridConfig):
        torch, nn = _require_torch()
        channels = cfg.hidden_channels
        in_ch = 1
        n_phys = n_physics_features(cfg)

        layers: list = []
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        conv = nn.Sequential(*layers)
        embed_dim = channels[-1]

        film = nn.Sequential(
            nn.Linear(n_phys, cfg.physics_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.physics_embed_dim, 2 * embed_dim),
        )

        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 128),
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
                self.film = film
                self.head = head

            def forward(self, x_mel, x_physics):
                h = self.conv(x_mel)
                gamma, beta = self.film(x_physics).chunk(2, dim=-1)
                b, c, _, _ = h.shape
                h = h * (1.0 + gamma.view(b, c, 1, 1)) + beta.view(b, c, 1, 1)
                return self.head(h)

        return _Net()
