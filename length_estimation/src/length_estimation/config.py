"""Configuration for VS13 vehicle length estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root: length_estimation/ (parent of src/length_estimation package)
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data" / "vs13"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"
DEFAULT_SPECS_PATH = PACKAGE_ROOT / "data" / "vehicle_specs.csv"
DEFAULT_CHECKPOINT_DIR = PACKAGE_ROOT / "checkpoints"

# VS13 native audio
SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
WINDOW = "hann"

# CPA-centred crop (seconds)
CROP_T_MIN = -4.0
CROP_T_MAX = 4.0

# Synchrosqueezing (DopplerSim convention)
SSQ_TARGET_SR = 11025
SSQ_N_SCALES = 64
SSQ_SCALE_MIN = 2
SSQ_SCALE_MAX = 128

# Mel / CNN
N_MELS = 128
FMAX_HZ = 8000.0

# Sub-band splits for cross-correlation (Hz)
SUBBAND_EDGES = (50, 500, 2000, 6000)

# Envelope thresholds (dB relative to peak)
ENV_DB_THRESHOLDS = (-3.0, -10.0)

# VS13 vehicle folder names (must match dataset directory names)
VS13_VEHICLE_DIRS = (
    "CitroenC4Picasso",
    "KiaSportage",
    "Mazda3",
    "MercedesAMG550",
    "MercedesGLA",
    "NissanQashqai",
    "OpelInsignia",
    "Peugeot208",
    "Peugeot3008",
    "Peugeot307",
    "RenaultCaptur",
    "RenaultScenic",
    "VWPassat",
)


@dataclass
class StftConfig:
    sr: int = SR
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    window: str = WINDOW
    fmax_hz: float = FMAX_HZ


@dataclass
class FeatureConfig:
    stft: StftConfig = field(default_factory=StftConfig)
    crop_t_min: float = CROP_T_MIN
    crop_t_max: float = CROP_T_MAX
    subband_edges: tuple[float, ...] = SUBBAND_EDGES
    env_db_thresholds: tuple[float, ...] = ENV_DB_THRESHOLDS
    compute_reassigned: bool = True
    compute_ssq: bool = True


@dataclass
class PhaseBConfig:
    """CNN length regressor — VS13 pass-by spectrograms, target length_m only."""

    spec_type: str = "mel"  # mel | ssq
    target: str = "length_m"
    batch_size: int = 16
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    huber_delta: float = 0.15
    patience: int = 12
    preempt: bool = False  # True = early-stop on val MAE; False = run all epochs (still save best.pt)
    include_speed: bool = True  # auxiliary input — disambiguates speed vs geometry
    hidden_channels: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.25
    # Fixed mel/ssq time axis for batching (CPA-centred ±4 s ≈ 689 frames @ hop 512)
    target_time_frames: int = 704
    n_mels: int = N_MELS
    # Speed auxiliary: normalise km/h to [0, 1] using VS13 range
    speed_kmh_max: float = 105.0

    def to_dict(self) -> dict:
        return {
            "spec_type": self.spec_type,
            "target": self.target,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "huber_delta": self.huber_delta,
            "patience": self.patience,
            "preempt": self.preempt,
            "include_speed": self.include_speed,
            "hidden_channels": list(self.hidden_channels),
            "dropout": self.dropout,
            "target_time_frames": self.target_time_frames,
            "n_mels": self.n_mels,
            "speed_kmh_max": self.speed_kmh_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PhaseBConfig:
        hc = d.get("hidden_channels", (32, 64, 128, 256))
        if isinstance(hc, list):
            hc = tuple(hc)
        return cls(
            spec_type=d.get("spec_type", "mel"),
            target=d.get("target", "length_m"),
            batch_size=int(d.get("batch_size", 16)),
            epochs=int(d.get("epochs", 60)),
            lr=float(d.get("lr", 3e-4)),
            weight_decay=float(d.get("weight_decay", 1e-4)),
            huber_delta=float(d.get("huber_delta", 0.15)),
            patience=int(d.get("patience", 12)),
            preempt=bool(d.get("preempt", False)),
            include_speed=bool(d.get("include_speed", True)),
            hidden_channels=hc,
            dropout=float(d.get("dropout", 0.25)),
            target_time_frames=int(d.get("target_time_frames", 704)),
            n_mels=int(d.get("n_mels", N_MELS)),
            speed_kmh_max=float(d.get("speed_kmh_max", 105.0)),
        )
