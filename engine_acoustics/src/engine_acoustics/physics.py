"""Reciprocating-engine acoustics: firing frequencies, orders, and cylinder phasing."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from engine_acoustics.config import EngineConfig, StrokeType

TWO_PI = 2.0 * math.pi
FOUR_STROKE_CYCLE_RAD = 2.0 * TWO_PI  # 720° crank = one complete 4-stroke cycle per cylinder


def crank_frequency_hz(rpm: float | np.ndarray) -> float | np.ndarray:
    """Shaft (1×) frequency in Hz from RPM."""
    return np.asarray(rpm, dtype=np.float64) / 60.0


def firing_frequency_hz(rpm: float | np.ndarray, num_cylinders: int, *, four_stroke: bool = True) -> float | np.ndarray:
    """
    Fundamental firing rate in Hz.

    Four-stroke: f_fire = (RPM/60) × (N/2) — one power stroke every two crank revolutions
    per cylinder, with N cylinders evenly distributed in crank angle.

    Two-stroke: f_fire = (RPM/60) × N.
    """
    factor = (num_cylinders / 2.0) if four_stroke else float(num_cylinders)
    return crank_frequency_hz(rpm) * factor


def order_frequency_hz(rpm: float | np.ndarray, order: float) -> float | np.ndarray:
    """
    Frequency of engine order `order` in Hz.

    Automotive convention: order n → f = n × (RPM/60).
    Examples at 3000 RPM (50 Hz crank):
      - 1×  = 50 Hz   (shaft / reciprocating)
      - 2×  = 100 Hz  (inline-4 firing fundamental)
      - 4×  = 200 Hz  (2nd harmonic of firing)
    """
    return crank_frequency_hz(rpm) * order


def firing_order_of_frequency(
    freq_hz: float | np.ndarray,
    rpm: float | np.ndarray,
) -> float | np.ndarray:
    """Map a frequency to its engine order at the given RPM."""
    fc = crank_frequency_hz(rpm)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(freq_hz, dtype=np.float64) / fc


def cylinder_firing_angles_rad(engine: EngineConfig) -> np.ndarray:
    """
    Crank angles (radians) at which each physical cylinder fires, ordered by cylinder index 0..N-1.

  For a four-stroke engine the crank must rotate 4π rad between successive firings of the
  same cylinder. Cylinders that fire earlier in the firing order are assigned proportionally
  spaced angles within that 4π window, then mapped back to physical cylinder indices.
    """
    n = engine.num_cylinders
    cycle_rad = FOUR_STROKE_CYCLE_RAD if engine.stroke_type == StrokeType.FOUR_STROKE else TWO_PI
    spacing = cycle_rad / n

    # Position in firing sequence → crank angle.
    sequence_angles = np.array(
        [(rank * spacing) for rank in range(n)],
        dtype=np.float64,
    )

    # firing_order[k] is the physical cylinder (1-based) at sequence position k.
    physical_angles = np.zeros(n, dtype=np.float64)
    for seq_idx, cyl_one_based in enumerate(engine.firing_order):
        physical_angles[cyl_one_based - 1] = sequence_angles[seq_idx]
    return physical_angles


def layout_phase_offsets_rad(engine: EngineConfig) -> np.ndarray:
    """
    Additional acoustic phase offsets from bank geometry (V/flat).

    Inline engines: all zeros. V engines: left/right bank cylinders receive opposing
    phase to model asymmetric exhaust routing (simplified first-order model).
    """
    n = engine.num_cylinders
    offsets = np.zeros(n, dtype=np.float64)
    if engine.layout.value in ("v", "flat", "boxer") and n % 2 == 0:
        half = n // 2
        bank_phase = math.radians(engine.bank_angle_deg) * 0.15
        offsets[:half] = -bank_phase
        offsets[half:] = bank_phase
    return offsets


def instantaneous_firing_frequency(
    rpm: np.ndarray,
    engine: EngineConfig,
) -> np.ndarray:
    """Time-varying firing fundamental (Hz) for an RPM trace."""
    return firing_frequency_hz(
        rpm,
        engine.num_cylinders,
        four_stroke=(engine.stroke_type == StrokeType.FOUR_STROKE),
    )


def rpm_from_crank_derivative(crank_rad_per_s: np.ndarray) -> np.ndarray:
    """Recover RPM from crank angular velocity (rad/s)."""
    return crank_rad_per_s * 60.0 / TWO_PI


def integrate_rpm_to_crank_angle(
    rpm: np.ndarray,
    sample_rate: int,
    initial_angle_rad: float = 0.0,
) -> np.ndarray:
    """
    Integrate RPM(t) to crank angle θ(t) in radians.

    θ(t) = θ₀ + ∫ 2π·RPM(τ)/60 dτ
    """
    omega = TWO_PI * rpm / 60.0
    dt = 1.0 / sample_rate
    return initial_angle_rad + np.cumsum(omega) * dt


def dominant_orders(engine: EngineConfig, max_order: float = 12.0) -> list[float]:
    """List of mechanically significant orders up to max_order."""
    primary = engine.primary_firing_order
    orders: set[float] = {1.0, 2.0, primary}
    if engine.layout.value == "v" and engine.num_cylinders % 2 == 0:
        orders.add(primary / 2.0)
    harmonic = primary
    while harmonic <= max_order:
        orders.add(harmonic)
        harmonic += primary
    for half in (0.5, 1.5, 2.5):
        if half <= max_order:
            orders.add(half)
    return sorted(orders)


def order_amplitude_vector(
    engine: EngineConfig,
    orders: Iterable[float],
) -> np.ndarray:
    """Amplitude weights for a list of orders using engine config."""
    table = engine.default_order_amplitudes()
    return np.array([table.get(float(o), 0.0) for o in orders], dtype=np.float64)
