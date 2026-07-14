"""Configuration for engine acoustic modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"

# Audio defaults aligned with DopplerLab mel pipelines
DEFAULT_SR = 22050
DEFAULT_DURATION_S = 4.0


class CylinderLayout(str, Enum):
    """Common reciprocating-engine layouts."""

    INLINE = "inline"
    V = "v"
    FLAT = "flat"
    BOXER = "boxer"


class StrokeType(str, Enum):
    FOUR_STROKE = "four_stroke"
    TWO_STROKE = "two_stroke"


@dataclass
class EngineConfig:
    """Mechanical description of a reciprocating engine."""

    num_cylinders: int = 4
    stroke_type: StrokeType = StrokeType.FOUR_STROKE
    layout: CylinderLayout = CylinderLayout.INLINE
    # Firing order as 1-based cylinder indices (e.g. inline-4: 1-3-4-2).
    firing_order: tuple[int, ...] = (1, 3, 4, 2)
    # Bank angle for V/flat engines (degrees). Inline uses 0.
    bank_angle_deg: float = 0.0
    # Per-order relative amplitudes (order -> gain). Orders are multiples of crank Hz.
    order_amplitudes: dict[float, float] = field(default_factory=dict)
    # Combustion pulse width as fraction of one firing interval (0–1).
    pulse_width_fraction: float = 0.12
    # Mechanical imbalance / reciprocating mass contribution at 1× and 2× crank.
    reciprocating_imbalance: float = 0.15

    def __post_init__(self) -> None:
        if self.num_cylinders < 1:
            raise ValueError("num_cylinders must be >= 1")
        if len(self.firing_order) != self.num_cylinders:
            raise ValueError("firing_order length must match num_cylinders")
        if sorted(self.firing_order) != list(range(1, self.num_cylinders + 1)):
            raise ValueError("firing_order must be a permutation of 1..N")
        if not 0.0 < self.pulse_width_fraction < 1.0:
            raise ValueError("pulse_width_fraction must be in (0, 1)")

    @property
    def strokes_per_power_cycle(self) -> int:
        return 4 if self.stroke_type == StrokeType.FOUR_STROKE else 2

    @property
    def firings_per_crank_rev(self) -> float:
        """Number of power strokes per crank revolution."""
        if self.stroke_type == StrokeType.FOUR_STROKE:
            return self.num_cylinders / 2.0
        return float(self.num_cylinders)

    @property
    def primary_firing_order(self) -> float:
        """Dominant acoustic order of the firing fundamental (× crank frequency)."""
        return self.firings_per_crank_rev

    def default_order_amplitudes(self) -> dict[float, float]:
        """Typical relative spectrum if none supplied."""
        if self.order_amplitudes:
            return dict(self.order_amplitudes)
        primary = self.primary_firing_order
        spectrum: dict[float, float] = {
            0.5: 0.05 * self.reciprocating_imbalance,
            1.0: 0.25 + self.reciprocating_imbalance,
            2.0: 0.35 if self.num_cylinders == 4 else 0.2,
        }
        spectrum[primary] = 1.0
        for harmonic in range(2, 7):
            order = primary * harmonic
            spectrum[order] = 0.6 ** harmonic
        # Layout-specific weak orders (e.g. V6 half-orders).
        if self.layout == CylinderLayout.V and self.num_cylinders % 2 == 0:
            half = primary / 2.0
            spectrum[half] = spectrum.get(half, 0.0) + 0.12
        return spectrum


# RPM as a function of time (seconds) -> revolutions per minute
RpmProfile = Callable[[float], float]


@dataclass
class SynthesisConfig:
    """Digital synthesis parameters."""

    sample_rate: int = DEFAULT_SR
    duration_s: float = DEFAULT_DURATION_S
    # Load fraction 0–1 scales combustion amplitude (throttle / torque proxy).
    load: float = 0.7
    # Broadband exhaust / intake noise floor (linear RMS scale).
    broadband_noise: float = 0.02
    # Random jitter on pulse amplitudes (cycle-to-cycle variation).
    cycle_variation: float = 0.08
    # Fade in/out to avoid clicks (seconds).
    edge_fade_s: float = 0.02
    seed: int | None = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.load <= 1.0:
            raise ValueError("load must be in [0, 1]")
