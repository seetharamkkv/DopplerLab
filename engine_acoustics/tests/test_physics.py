"""Basic tests for engine acoustics physics and synthesis."""

from __future__ import annotations

import numpy as np

from engine_acoustics.config import EngineConfig
from engine_acoustics.physics import (
    crank_frequency_hz,
    cylinder_firing_angles_rad,
    firing_frequency_hz,
    order_frequency_hz,
)
from engine_acoustics.synthesis import EngineSynthesizer, constant_rpm


def test_firing_frequency_inline4() -> None:
    assert firing_frequency_hz(3000, 4) == 100.0
    assert crank_frequency_hz(3000) == 50.0
    assert order_frequency_hz(3000, 2.0) == 100.0


def test_cylinder_angles_span_cycle() -> None:
    engine = EngineConfig(num_cylinders=4, firing_order=(1, 3, 4, 2))
    angles = cylinder_firing_angles_rad(engine)
    assert len(angles) == 4
  # Evenly spaced at 0, π, 2π, 3π crank radians (720° cycle / 4 cylinders).
    assert np.allclose(np.sort(angles), [0.0, np.pi, 2 * np.pi, 3 * np.pi])


def test_synthesis_produces_finite_audio() -> None:
    engine = EngineConfig(num_cylinders=4)
    synth = EngineSynthesizer(engine)
    audio, meta = synth.synthesize(constant_rpm(2000.0), mode="additive")
    assert audio.dtype == np.float32
    assert len(audio) == len(meta["rpm"])
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) <= 1.0
