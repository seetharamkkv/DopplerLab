"""Four-stroke engine acoustics: physics, synthesis, and order tracking."""

from engine_acoustics.config import EngineConfig, SynthesisConfig
from engine_acoustics.physics import (
    crank_frequency_hz,
    firing_frequency_hz,
    order_frequency_hz,
    cylinder_firing_angles_rad,
)
from engine_acoustics.synthesis import EngineSynthesizer
from engine_acoustics.order_tracking import OrderTracker, order_spectrum

__all__ = [
    "EngineConfig",
    "SynthesisConfig",
    "EngineSynthesizer",
    "OrderTracker",
    "crank_frequency_hz",
    "firing_frequency_hz",
    "order_frequency_hz",
    "cylinder_firing_angles_rad",
    "order_spectrum",
]
