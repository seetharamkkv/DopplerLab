"""Time-domain engine acoustic synthesis from RPM profiles."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy import signal

from engine_acoustics.config import EngineConfig, RpmProfile, SynthesisConfig
from engine_acoustics.physics import (
    FOUR_STROKE_CYCLE_RAD,
    TWO_PI,
    cylinder_firing_angles_rad,
    integrate_rpm_to_crank_angle,
    layout_phase_offsets_rad,
)

SynthesisMode = Literal["pulse", "additive", "hybrid"]


class EngineSynthesizer:
    """
    Synthesize engine audio from RPM(t), cylinder count, and mechanical layout.

    Three modes:
      - pulse:   crank-angle impulse train with per-cylinder phasing (most physical)
      - additive: order-based sinusoidal bank tracked to instantaneous RPM
      - hybrid:  pulse excitation filtered by order-shaped resonances
    """

    def __init__(self, engine: EngineConfig, config: SynthesisConfig | None = None) -> None:
        self.engine = engine
        self.config = config or SynthesisConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def synthesize(
        self,
        rpm_profile: RpmProfile | np.ndarray,
        *,
        mode: SynthesisMode = "hybrid",
    ) -> tuple[np.ndarray, dict]:
        """
        Generate engine audio.

        Parameters
        ----------
        rpm_profile:
            Callable RPM(t) or precomputed RPM array aligned to output samples.
        mode:
            'pulse', 'additive', or 'hybrid'.

        Returns
        -------
        audio:
            Mono float32 waveform, peak-normalized to 0.95.
        meta:
            Sidecar dict with crank angle, RPM trace, sample times.
        """
        sr = self.config.sample_rate
        n = int(self.config.duration_s * sr)
        t = np.arange(n, dtype=np.float64) / sr
        rpm = self._resolve_rpm(rpm_profile, t)

        if mode == "pulse":
            audio = self._synthesize_pulse(rpm, t)
        elif mode == "additive":
            audio = self._synthesize_additive(rpm, t)
        elif mode == "hybrid":
            audio = self._synthesize_hybrid(rpm, t)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        audio = self._apply_load_and_noise(audio)
        audio = self._apply_edge_fade(audio, sr)
        peak = np.max(np.abs(audio)) or 1.0
        audio = (0.95 * audio / peak).astype(np.float32)

        crank = integrate_rpm_to_crank_angle(rpm, sr)
        meta = {
            "time_s": t,
            "rpm": rpm,
            "crank_angle_rad": crank,
            "sample_rate": sr,
            "mode": mode,
            "engine": {
                "num_cylinders": self.engine.num_cylinders,
                "firing_order": self.engine.firing_order,
                "primary_firing_order": self.engine.primary_firing_order,
            },
        }
        return audio, meta

    def _resolve_rpm(self, rpm_profile: RpmProfile | np.ndarray, t: np.ndarray) -> np.ndarray:
        if callable(rpm_profile):
            rpm = np.array([float(rpm_profile(float(ti))) for ti in t], dtype=np.float64)
        else:
            rpm = np.asarray(rpm_profile, dtype=np.float64)
            if rpm.shape != t.shape:
                raise ValueError("rpm_profile array must match synthesized time length")
        return np.clip(rpm, 0.0, None)

    def _synthesize_pulse(self, rpm: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Crank-synchronous combustion pulses with cylinder phasing."""
        sr = self.config.sample_rate
        n = len(t)
        crank = integrate_rpm_to_crank_angle(rpm, sr)
        firing_angles = cylinder_firing_angles_rad(self.engine)
        layout_phase = layout_phase_offsets_rad(self.engine)
        cycle = (
            FOUR_STROKE_CYCLE_RAD
            if self.engine.stroke_type.value == "four_stroke"
            else TWO_PI
        )

        pulse_len = max(8, int(sr * 0.004))  # ~4 ms kernel
        kernel = self._combustion_kernel(pulse_len)

        audio = np.zeros(n, dtype=np.float64)
        prev_wrapped = np.mod(crank[0], cycle)

        for i in range(1, n):
            wrapped = np.mod(crank[i], cycle)
            # Detect forward wrap → one full cycle completed.
            if wrapped < prev_wrapped:
                delta = (cycle - prev_wrapped) + wrapped
            else:
                delta = wrapped - prev_wrapped

            # Any cylinder whose firing angle was crossed this sample?
            for cyl_idx, alpha in enumerate(firing_angles):
                # Crossings of alpha within [prev, curr] modulo cycle.
                if self._angle_crossed(prev_wrapped, wrapped, cycle, alpha):
                    amp = 1.0 + self.config.cycle_variation * self._rng.standard_normal()
                    start = i
                    end = min(n, start + pulse_len)
                    seg_len = end - start
                    audio[start:end] += amp * kernel[:seg_len] * math.cos(layout_phase[cyl_idx])

            prev_wrapped = wrapped

        return audio

    @staticmethod
    def _angle_crossed(prev: float, curr: float, cycle: float, alpha: float) -> bool:
        """True if crank angle crossed alpha going forward (handles wrap)."""
        if curr >= prev:
            return prev <= alpha < curr
        # Wrapped through zero.
        return alpha >= prev or alpha < curr

    def _combustion_kernel(self, length: int) -> np.ndarray:
        """Band-limited asymmetric pulse (rapid rise, slower decay)."""
        t = np.linspace(0.0, 1.0, length, endpoint=False)
        rise = np.exp(-((t - 0.05) ** 2) / 0.0008)
        decay = np.exp(-4.5 * t)
        k = rise * decay
        return k / (np.max(np.abs(k)) or 1.0)

    def _synthesize_additive(self, rpm: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Order-tracked sinusoids — efficient, smooth under RPM sweeps."""
        sr = self.config.sample_rate
        orders = sorted(self.engine.default_order_amplitudes().keys())
        amplitudes = np.array([self.engine.default_order_amplitudes()[o] for o in orders])
        amplitudes /= np.max(amplitudes) or 1.0

        audio = np.zeros_like(t)
        crank_phase = integrate_rpm_to_crank_angle(rpm, sr, initial_angle_rad=0.0)

        for order, amp in zip(orders, amplitudes):
            if amp <= 0.0:
                continue
            # Phase = order × crank angle (order is multiple of shaft frequency).
            phase = order * crank_phase
            audio += amp * np.sin(phase)

        return audio

    def _synthesize_hybrid(self, rpm: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Pulse excitation shaped by order-dependent resonances (intake/exhaust coloration)."""
        excitation = self._synthesize_pulse(rpm, t)
        sr = self.config.sample_rate

        # Parallel resonators at dominant orders (time-varying via FM from RPM).
        resonant = self._synthesize_additive(rpm, t)

        # High-frequency aspiration noise modulated by firing rate.
        n = len(t)
        noise = self._rng.standard_normal(n)
        # Bandpass 200–2500 Hz, amplitude follows load.
        sos = signal.butter(4, [200.0, 2500.0], btype="bandpass", fs=sr, output="sos")
        colored = signal.sosfilt(sos, noise)
        firing_env = self._smooth_envelope(excitation)

        mix = 0.55 * excitation + 0.40 * resonant + 0.05 * colored * firing_env
        return mix

    def _smooth_envelope(self, x: np.ndarray, window_ms: float = 10.0) -> np.ndarray:
        sr = self.config.sample_rate
        w = max(3, int(sr * window_ms / 1000.0))
        if w % 2 == 0:
            w += 1
        return signal.medfilt(np.abs(x), kernel_size=w) / (np.max(np.abs(x)) or 1.0)

    def _apply_load_and_noise(self, audio: np.ndarray) -> np.ndarray:
        load = self.config.load
        # Nonlinear load curve: idle is quiet; mid-load loudest; near WOT slightly compressed.
        load_gain = 0.15 + 0.85 * (load ** 0.7)
        audio = audio * load_gain

        if self.config.broadband_noise > 0:
            noise = self._rng.standard_normal(len(audio))
            audio = audio + self.config.broadband_noise * load_gain * noise
        return audio

    def _apply_edge_fade(self, audio: np.ndarray, sr: int) -> np.ndarray:
        n_fade = int(self.config.edge_fade_s * sr)
        if n_fade <= 0 or len(audio) < 2 * n_fade:
            return audio
        ramp = np.linspace(0.0, 1.0, n_fade)
        audio = audio.copy()
        audio[:n_fade] *= ramp
        audio[-n_fade:] *= ramp[::-1]
        return audio


def constant_rpm(rpm_value: float) -> RpmProfile:
    """Constant RPM profile factory."""

    def profile(_t: float) -> float:
        return rpm_value

    return profile


def ramp_rpm(rpm_start: float, rpm_end: float, duration_s: float) -> RpmProfile:
    """Linear RPM sweep."""

    def profile(t: float) -> float:
        frac = min(1.0, max(0.0, t / duration_s))
        return rpm_start + frac * (rpm_end - rpm_start)

    return profile


def step_rpm(steps: list[tuple[float, float]]) -> RpmProfile:
    """
    Piecewise-constant RPM schedule.

    steps: [(t_start_seconds, rpm), ...] sorted by time.
  """

    def profile(t: float) -> float:
        value = steps[0][1]
        for t_start, rpm in steps:
            if t >= t_start:
                value = rpm
        return value

    return profile
