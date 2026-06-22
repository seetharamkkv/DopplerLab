"""Configuration for IDMT-Traffic direction experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "IDMT_Traffic"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"
DEFAULT_CHECKPOINT_DIR = PACKAGE_ROOT / "checkpoints"
DEFAULT_ANNOTATION_DIR = DEFAULT_DATA_DIR / "annotation"

# IDMT native audio (EUSIPCO paper)
SR_NATIVE = 48000
SR_MEL = 22050
CLIP_DURATION_S = 2.0

# Mel (EUSIPCO vehicle-type / our direction CNN)
N_FFT = 2048
WIN_LENGTH = 1024
HOP_LENGTH = 512
N_MELS = 64

# Cross-correlation stack (EUSIPCO direction experiment)
CC_BLOCK_S = 0.200
CC_HOP_S = 0.025
CC_MARGIN = 25  # lags each side of zero -> 51 bins

DIRECTION_LABELS = ("L2R", "R2L", "no_vehicle")
VEHICLE_TYPE_LABELS = ("bus", "car", "motorcycle", "truck", "none")
VEHICLE_CODE_TO_IDX = {"B": 0, "C": 1, "M": 2, "T": 3}
# Back-compat alias
VEHICLE_LABELS = VEHICLE_TYPE_LABELS

# Paper default for published benchmarks
DEFAULT_MIC = "SE"
DEFAULT_CHANNEL = "CH34"


def resolve_class_labels(cfg: "DirectionConfig") -> tuple[str, ...]:
    if cfg.task == "vehicle":
        return VEHICLE_TYPE_LABELS
    if cfg.n_classes == 2:
        return DIRECTION_LABELS[:2]
    return DIRECTION_LABELS


def checkpoint_subdir(cfg: "DirectionConfig") -> str:
    return cfg.task if cfg.task in ("direction", "vehicle") else "direction"


@dataclass
class DirectionConfig:
    """IDMT-Traffic classifier (direction-of-travel or vehicle type)."""

    task: str = "direction"  # direction | vehicle
    feature_type: str = "mel"  # mel | cc | stereo_mel
    n_classes: int = 3  # direction: 2|3; vehicle: 5 (bus/car/motorcycle/truck/none)
    batch_size: int = 32
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 10
    preempt: bool = False
    hidden_channels: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.3
    n_mels: int = N_MELS
    mic_filter: str = DEFAULT_MIC
    channel_filter: str = DEFAULT_CHANNEL
    split_name: str = "eusipco"  # eusipco | location_loo | weather_holdout
    val_fraction: float = 0.1
    split_seed: int = 42
    norm_fit_max_samples: int | None = 512  # None = all train clips; leak-safe subsample for speed

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "feature_type": self.feature_type,
            "n_classes": self.n_classes,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "preempt": self.preempt,
            "hidden_channels": list(self.hidden_channels),
            "dropout": self.dropout,
            "n_mels": self.n_mels,
            "mic_filter": self.mic_filter,
            "channel_filter": self.channel_filter,
            "split_name": self.split_name,
            "val_fraction": self.val_fraction,
            "split_seed": self.split_seed,
            "norm_fit_max_samples": self.norm_fit_max_samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DirectionConfig:
        hc = d.get("hidden_channels", (32, 64, 128, 256))
        if isinstance(hc, list):
            hc = tuple(hc)
        task = d.get("task", "direction")
        default_n = 5 if task == "vehicle" else 3
        return cls(
            task=task,
            feature_type=d.get("feature_type", "mel"),
            n_classes=int(d.get("n_classes", default_n)),
            batch_size=int(d.get("batch_size", 32)),
            epochs=int(d.get("epochs", 40)),
            lr=float(d.get("lr", 1e-4)),
            weight_decay=float(d.get("weight_decay", 1e-4)),
            patience=int(d.get("patience", 10)),
            preempt=bool(d.get("preempt", False)),
            hidden_channels=hc,
            dropout=float(d.get("dropout", 0.3)),
            n_mels=int(d.get("n_mels", N_MELS)),
            mic_filter=d.get("mic_filter", DEFAULT_MIC),
            channel_filter=d.get("channel_filter", DEFAULT_CHANNEL),
            split_name=d.get("split_name", "eusipco"),
            val_fraction=float(d.get("val_fraction", 0.1)),
            split_seed=int(d.get("split_seed", 42)),
            norm_fit_max_samples=d.get("norm_fit_max_samples", 512),
        )


@dataclass
class NormStats:
    """Train-fitted per-bin normalization (no val/test leakage)."""

    mel_mean: list[float] | None = None
    mel_std: list[float] | None = None
    cc_mean: list[float] | None = None
    cc_std: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "mel_mean": self.mel_mean,
            "mel_std": self.mel_std,
            "cc_mean": self.cc_mean,
            "cc_std": self.cc_std,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> NormStats | None:
        if not d:
            return None
        return cls(
            mel_mean=d.get("mel_mean"),
            mel_std=d.get("mel_std"),
            cc_mean=d.get("cc_mean"),
            cc_std=d.get("cc_std"),
        )
