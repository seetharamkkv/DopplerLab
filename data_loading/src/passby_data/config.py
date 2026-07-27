"""Default paths and mel / audio constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # data_loading/
DOPPLERLAB_ROOT = PACKAGE_ROOT.parent

# Default local pass-by audio root. Override with --real_root if needed.
DEFAULT_REAL_ROOT = DOPPLERLAB_ROOT / "speed_estimation" / "passby"
DEFAULT_SPLITS_DIR = PACKAGE_ROOT / "splits"


@dataclass(frozen=True)
class AudioConfig:
    """Mel pipeline matching the common SE-ResNet audio config."""

    sample_rate: int = 16_000
    duration_seconds: float = 10.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512

    @property
    def audio_length_samples(self) -> int:
        return int(self.sample_rate * self.duration_seconds)

    @property
    def n_frames(self) -> int:
        import math

        return int(math.ceil(self.audio_length_samples / self.hop_length))
