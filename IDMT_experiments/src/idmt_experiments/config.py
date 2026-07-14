"""Configuration for IDMT-Traffic direction experiments.

REPRODUCIBILITY BASELINE — shared CNN dependency (mel_3class / mel_3class_left / mel_3class_right)
---------------------------------------------------------------------------------
``DirectionConfig`` defaults, mel constants, and ``NormStats`` are on the CNN baseline
critical path. Do not change default behaviour, numerics, or evaluation outputs without
re-benchmarking all three reference runs. Refactoring is OK only if metrics stay
bit-identical. New work: separate --run-name or new config types.
Verified: outputs/_repro/REPRODUCTION.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


DEFAULT_DATA_DIR = _env_path("IDMT_DATA_DIR", PACKAGE_ROOT / "IDMT_Traffic")
DEFAULT_OUTPUT_DIR = _env_path("IDMT_OUTPUT_DIR", PACKAGE_ROOT / "outputs")
DEFAULT_SHARED_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "shared"
DEFAULT_CHECKPOINT_DIR = _env_path("IDMT_CHECKPOINT_DIR", PACKAGE_ROOT / "checkpoints")
DEFAULT_ANNOTATION_DIR = DEFAULT_DATA_DIR / "annotation"

# Model-family layout: checkpoints/{cnn|physics}/{task}/<run_name>/
MODEL_CNN = "cnn"
MODEL_PHYSICS = "physics"
MODEL_HYBRID = "hybrid"

# IDMT native audio (EUSIPCO paper)
SR_NATIVE = 48000
SR_MEL = 22050
CLIP_DURATION_S = 2.0

# REPRODUCIBILITY BASELINE — mel spectrogram constants for CNN direction models.
N_FFT = 2048
WIN_LENGTH = 1024
HOP_LENGTH = 512
N_MELS = 64
FMAX_HZ = 8000.0  # complex STFT freq crop (matches physics / diffusion tracks)

# Cross-correlation stack (EUSIPCO direction experiment)
CC_BLOCK_S = 0.200
CC_HOP_S = 0.025
CC_MARGIN = 25  # lags each side of zero -> 51 bins

DIRECTION_LABELS = ("L2R", "R2L", "no_vehicle")
VEHICLE_TYPE_LABELS = ("bus", "car", "motorcycle", "truck", "none")
WEATHER_LABELS = ("dry", "wet")
VEHICLE_CODE_TO_IDX = {"B": 0, "C": 1, "M": 2, "T": 3}
WEATHER_CODE_TO_IDX = {"D": 0, "W": 1}
# Back-compat alias
VEHICLE_LABELS = VEHICLE_TYPE_LABELS

# Paper default for published benchmarks
DEFAULT_MIC = "SE"
DEFAULT_CHANNEL = "CH34"

# IDMT wet recordings exist only at this site (dry-only at the other two locations).
WEATHER_EVAL_SITE = "Schleusinger-Allee"


def resolve_class_labels(cfg: "DirectionConfig") -> tuple[str, ...]:
    if cfg.task == "vehicle":
        return VEHICLE_TYPE_LABELS
    if cfg.task == "weather":
        return WEATHER_LABELS
    if cfg.n_classes == 2:
        return DIRECTION_LABELS[:2]
    return DIRECTION_LABELS


def _task_subdir(cfg: "DirectionConfig") -> str:
    return cfg.task if cfg.task in ("direction", "vehicle", "weather") else "direction"


def checkpoint_subdir(cfg: "DirectionConfig", *, model_family: str = MODEL_CNN) -> str:
    """Relative path under checkpoints/ or outputs/, e.g. cnn/direction."""
    return f"{model_family}/{_task_subdir(cfg)}"


def legacy_checkpoint_subdir(cfg: "DirectionConfig") -> str:
    """Pre-layout paths: direction/, weather/, vehicle/ directly under checkpoints/."""
    return _task_subdir(cfg)


def resolve_run_dir(
    cfg: "DirectionConfig",
    run_name: str,
    *,
    root: Path,
    model_family: str = MODEL_CNN,
) -> Path:
    """Resolve run directory; fall back to legacy layout if the new path is absent."""
    primary = root / checkpoint_subdir(cfg, model_family=model_family) / run_name
    if primary.exists():
        return primary
    legacy = root / legacy_checkpoint_subdir(cfg) / run_name
    return legacy if legacy.exists() else primary


def resolve_checkpoint_file(
    cfg: "DirectionConfig",
    run_name: str,
    *,
    checkpoint_dir: Path | None = None,
    filename: str = "best.pt",
    model_family: str = MODEL_CNN,
) -> Path:
    """Resolve ``best.pt`` for a run (REPRODUCIBILITY BASELINE — mel_3class* checkpoints)."""
    run_dir = resolve_run_dir(
        cfg,
        run_name,
        root=checkpoint_dir or DEFAULT_CHECKPOINT_DIR,
        model_family=model_family,
    )
    return run_dir / filename


def physics_checkpoint_subdir(cfg: "PhysicsConfig") -> str:
    return f"{MODEL_PHYSICS}/{cfg.task}"


def resolve_physics_run_dir(
    cfg: "PhysicsConfig",
    run_name: str,
    *,
    root: Path | None = None,
) -> Path:
    return (root or DEFAULT_CHECKPOINT_DIR) / physics_checkpoint_subdir(cfg) / run_name


def hybrid_checkpoint_subdir(cfg: "HybridConfig") -> str:
    return f"{MODEL_HYBRID}/{cfg.task}"


def resolve_hybrid_run_dir(
    cfg: "HybridConfig",
    run_name: str,
    *,
    root: Path | None = None,
) -> Path:
    return (root or DEFAULT_CHECKPOINT_DIR) / hybrid_checkpoint_subdir(cfg) / run_name


def feature_input_channels(feature_type: str) -> int:
    """CNN first-conv input channels.

    - ``mel`` / ``cc``: 1
    - ``stereo_mel``: 2 (left + right **microphone** mels — true stereo input)
    - ``complex_stft``: 2 (real + imag **spectral** components from mono audio)
    """
    if feature_type in ("stereo_mel", "complex_stft"):
        return 2
    return 1


def complex_stft_n_freq_bins(
    *,
    sr: int = SR_MEL,
    n_fft: int = N_FFT,
    fmax_hz: float = FMAX_HZ,
) -> int:
    import librosa
    import numpy as np

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return int(np.sum(freqs <= fmax_hz))


@dataclass
class DirectionConfig:
    """IDMT-Traffic classifier (direction-of-travel or vehicle type).

    REPRODUCIBILITY BASELINE — default field values define mel_3class / mel_3class_left /
    mel_3class_right training and eval. Do not change defaults without re-benchmarking.
    """

    task: str = "direction"  # direction | vehicle | weather
    feature_type: str = "mel"  # mel | cc | stereo_mel | complex_stft
    mono_source: str = "mean"  # mean | left | right — mel/complex_stft mono path
    n_classes: int = 3  # direction: 2|3; vehicle: 5; weather: 2 (dry/wet)
    batch_size: int = 32
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 10
    preempt: bool = False
    min_epochs: int = 0  # with preempt: do not early-stop before this epoch (0 = no floor)
    hidden_channels: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.3
    n_mels: int = N_MELS
    mic_filter: str = DEFAULT_MIC
    channel_filter: str = DEFAULT_CHANNEL
    split_name: str = "eusipco"  # eusipco | location_loo | weather_holdout | weather_site | weather_stratified
    val_fraction: float = 0.1
    split_seed: int = 42
    norm_fit_max_samples: int | None = 512  # None = all train clips; leak-safe subsample for speed

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "feature_type": self.feature_type,
            "mono_source": self.mono_source,
            "n_classes": self.n_classes,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "preempt": self.preempt,
            "min_epochs": self.min_epochs,
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
        if task == "vehicle":
            default_n = 5
        elif task == "weather":
            default_n = 2
        else:
            default_n = 3
        return cls(
            task=task,
            feature_type=d.get("feature_type", "mel"),
            mono_source=d.get("mono_source", "mean"),
            n_classes=int(d.get("n_classes", default_n)),
            batch_size=int(d.get("batch_size", 32)),
            epochs=int(d.get("epochs", 40)),
            lr=float(d.get("lr", 1e-4)),
            weight_decay=float(d.get("weight_decay", 1e-4)),
            patience=int(d.get("patience", 10)),
            preempt=bool(d.get("preempt", False)),
            min_epochs=int(d.get("min_epochs", 0)),
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


# Physics direction: L2R / R2L only (vehicle clips by default).
PHYSICS_DIRECTION_LABELS = DIRECTION_LABELS[:2]


@dataclass
class PhysicsConfig:
    """Physics-informed direction classifier (kinematic features + sklearn)."""

    task: str = "direction"
    mono_source: str = "left"  # pure mono: left | right (not mean)
    n_classes: int = 2  # L2R / R2L only
    include_no_vehicle: bool = False  # when False, drop background clips
    classifier: str = "logistic"  # logistic | rules (rules: later)
    feature_set: str = "kinematic_v2"
    cpa_mode: str = "envelope_peak"  # clip_center | envelope_peak — kinematic_v2 forces clip_center
    use_speed_scaled_features: bool = True
    mic_filter: str = DEFAULT_MIC
    channel_filter: str = DEFAULT_CHANNEL
    split_name: str = "eusipco"
    val_fraction: float = 0.1
    split_seed: int = 42
    max_iter: int = 2000

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "mono_source": self.mono_source,
            "n_classes": self.n_classes,
            "include_no_vehicle": self.include_no_vehicle,
            "classifier": self.classifier,
            "feature_set": self.feature_set,
            "cpa_mode": self.cpa_mode,
            "use_speed_scaled_features": self.use_speed_scaled_features,
            "mic_filter": self.mic_filter,
            "channel_filter": self.channel_filter,
            "split_name": self.split_name,
            "val_fraction": self.val_fraction,
            "split_seed": self.split_seed,
            "max_iter": self.max_iter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PhysicsConfig:
        return cls(
            task=d.get("task", "direction"),
            mono_source=d.get("mono_source", "left"),
            n_classes=int(d.get("n_classes", 2)),
            include_no_vehicle=bool(d.get("include_no_vehicle", False)),
            classifier=d.get("classifier", "logistic"),
            feature_set=d.get("feature_set", "kinematic_v1"),
            cpa_mode=d.get("cpa_mode", "envelope_peak"),
            use_speed_scaled_features=bool(d.get("use_speed_scaled_features", True)),
            mic_filter=d.get("mic_filter", DEFAULT_MIC),
            channel_filter=d.get("channel_filter", DEFAULT_CHANNEL),
            split_name=d.get("split_name", "eusipco"),
            val_fraction=float(d.get("val_fraction", 0.1)),
            split_seed=int(d.get("split_seed", 42)),
            max_iter=int(d.get("max_iter", 2000)),
        )


@dataclass
class PhysicsScaler:
    """Train-fitted z-score for physics feature vectors (no val/test leakage)."""

    mean: list[float]
    std: list[float]
    feature_names: list[str]

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> PhysicsScaler | None:
        if not d:
            return None
        return cls(
            mean=list(d["mean"]),
            std=list(d["std"]),
            feature_names=list(d["feature_names"]),
        )


@dataclass
class HybridConfig:
    """CNN mel backbone + physics feature conditioning (PINN-style late fusion)."""

    task: str = "direction"
    feature_type: str = "mel"
    mono_source: str = "left"
    n_classes: int = 3
    batch_size: int = 32
    epochs: int = 60
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 10
    preempt: bool = False
    hidden_channels: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.3
    n_mels: int = N_MELS
    mic_filter: str = DEFAULT_MIC
    channel_filter: str = DEFAULT_CHANNEL
    split_name: str = "eusipco"
    val_fraction: float = 0.1
    split_seed: int = 42
    norm_fit_max_samples: int | None = 512
    feature_set: str = "kinematic_v3"
    use_speed_scaled_features: bool = True
    include_no_vehicle: bool = True
    physics_embed_dim: int = 32
    physics_dropout: float = 0.2

    def to_direction_config(self) -> DirectionConfig:
        return DirectionConfig(
            task=self.task,
            feature_type=self.feature_type,
            mono_source=self.mono_source,
            n_classes=self.n_classes,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            preempt=self.preempt,
            hidden_channels=self.hidden_channels,
            dropout=self.dropout,
            n_mels=self.n_mels,
            mic_filter=self.mic_filter,
            channel_filter=self.channel_filter,
            split_name=self.split_name,
            val_fraction=self.val_fraction,
            split_seed=self.split_seed,
            norm_fit_max_samples=self.norm_fit_max_samples,
        )

    def to_physics_config(self) -> PhysicsConfig:
        return PhysicsConfig(
            mono_source=self.mono_source,
            n_classes=2,
            include_no_vehicle=self.include_no_vehicle,
            feature_set=self.feature_set,
            use_speed_scaled_features=self.use_speed_scaled_features,
            mic_filter=self.mic_filter,
            channel_filter=self.channel_filter,
            split_name=self.split_name,
            val_fraction=self.val_fraction,
            split_seed=self.split_seed,
        )

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "feature_type": self.feature_type,
            "mono_source": self.mono_source,
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
            "feature_set": self.feature_set,
            "use_speed_scaled_features": self.use_speed_scaled_features,
            "include_no_vehicle": self.include_no_vehicle,
            "physics_embed_dim": self.physics_embed_dim,
            "physics_dropout": self.physics_dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HybridConfig:
        hc = d.get("hidden_channels", (32, 64, 128, 256))
        if isinstance(hc, list):
            hc = tuple(hc)
        return cls(
            task=d.get("task", "direction"),
            feature_type=d.get("feature_type", "mel"),
            mono_source=d.get("mono_source", "left"),
            n_classes=int(d.get("n_classes", 3)),
            batch_size=int(d.get("batch_size", 32)),
            epochs=int(d.get("epochs", 60)),
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
            feature_set=d.get("feature_set", "kinematic_v3"),
            use_speed_scaled_features=bool(d.get("use_speed_scaled_features", True)),
            include_no_vehicle=bool(d.get("include_no_vehicle", True)),
            physics_embed_dim=int(d.get("physics_embed_dim", 32)),
            physics_dropout=float(d.get("physics_dropout", 0.2)),
        )


@dataclass
class NormStats:
    """Train-fitted per-bin normalization (no val/test leakage).

    REPRODUCIBILITY BASELINE — mel_mean/mel_std in saved checkpoints must reload identically.
    """

    mel_mean: list[float] | None = None
    mel_std: list[float] | None = None
    cc_mean: list[float] | None = None
    cc_std: list[float] | None = None
    cpx_mean: list[list[float]] | None = None  # [real, imag] per-frequency means
    cpx_std: list[list[float]] | None = None

    def to_dict(self) -> dict:
        return {
            "mel_mean": self.mel_mean,
            "mel_std": self.mel_std,
            "cc_mean": self.cc_mean,
            "cc_std": self.cc_std,
            "cpx_mean": self.cpx_mean,
            "cpx_std": self.cpx_std,
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
            cpx_mean=d.get("cpx_mean"),
            cpx_std=d.get("cpx_std"),
        )
