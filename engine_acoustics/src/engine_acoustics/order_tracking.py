"""Engine order tracking and order-domain spectral analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from engine_acoustics.config import EngineConfig
from engine_acoustics.physics import (
    crank_frequency_hz,
    dominant_orders,
    integrate_rpm_to_crank_angle,
    order_amplitude_vector,
    order_frequency_hz,
)


@dataclass
class OrderSpectrumResult:
    """Order-domain spectrum at a reference RPM."""

    orders: np.ndarray
    magnitudes: np.ndarray
    reference_rpm: float
    sample_rate: int


class OrderTracker:
    """
    Computed order tracking (angular resampling) for engine audio.

    Given a known or estimated RPM trace, resamples the waveform at uniform
    crank-angle increments and FFTs each order slice. This is the standard
    DSP approach for separating shaft/firing/harmonic content under varying speed.
    """

    def __init__(
        self,
        engine: EngineConfig,
        sample_rate: int,
        samples_per_rev: int = 256,
    ) -> None:
        self.engine = engine
        self.sample_rate = sample_rate
        self.samples_per_rev = samples_per_rev

    def angular_resample(
        self,
        audio: np.ndarray,
        rpm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resample audio to uniform crank-angle domain.

        Returns
        -------
        angle_signal:
            Audio sampled every Δθ = 2π / samples_per_rev radians.
        theta_axis:
            Crank angle (rad) for each sample in angle_signal.
        """
        crank = integrate_rpm_to_crank_angle(rpm, self.sample_rate)
        if crank[-1] <= crank[0]:
            raise ValueError("Crank angle must increase over the segment")

        n_revs = (crank[-1] - crank[0]) / (2.0 * np.pi)
        n_angle = max(2, int(n_revs * self.samples_per_rev))
        theta_uniform = np.linspace(crank[0], crank[-1], n_angle)

        interpolator = interp1d(
            crank,
            audio,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return interpolator(theta_uniform), theta_uniform

    def order_spectrum(
        self,
        audio: np.ndarray,
        rpm: np.ndarray,
        *,
        max_order: float = 16.0,
        reference_rpm: float | None = None,
    ) -> OrderSpectrumResult:
        """
        Magnitude spectrum in the order domain via angular resampling + FFT.

        The FFT of the angle-domain signal reveals content at integer and
        fractional multiples of crank frequency when referenced to reference_rpm.
        """
        angle_sig, _ = self.angular_resample(audio, rpm)
        angle_sig = angle_sig - np.mean(angle_sig)
        window = signal.windows.hann(len(angle_sig))
        spectrum = np.fft.rfft(angle_sig * window)
        mag = np.abs(spectrum)

        # Bin index k ↔ order k at the mean shaft speed over the segment.
        mean_rpm = float(np.mean(rpm))
        ref = mean_rpm if reference_rpm is None else reference_rpm
        n_bins = len(mag)
        orders = np.arange(n_bins, dtype=np.float64)

        # Normalize magnitudes to peak 1 for shape comparison.
        peak = np.max(mag) or 1.0
        mag_norm = mag / peak

        mask = orders <= max_order
        return OrderSpectrumResult(
            orders=orders[mask],
            magnitudes=mag_norm[mask],
            reference_rpm=ref,
            sample_rate=self.sample_rate,
        )

    def extract_order_tracks(
        self,
        audio: np.ndarray,
        rpm: np.ndarray,
        orders: list[float] | None = None,
        *,
        filter_q: float = 30.0,
    ) -> dict[float, np.ndarray]:
        """
        Vold-Kalman–style simplified tracking: narrow bandpass per order, centered at order×RPM(t).

        Returns time-domain bandpassed signals, one per requested order.
        """
        if orders is None:
            orders = dominant_orders(self.engine, max_order=12.0)

        sr = self.sample_rate
        n = len(audio)
        t = np.arange(n, dtype=np.float64) / sr
        tracks: dict[float, np.ndarray] = {}

        for order in orders:
            center = order_frequency_hz(rpm, order)
            # Time-varying center: use short STFT chunks with fixed center per chunk.
            chunk = max(256, sr // 20)
            out = np.zeros(n, dtype=np.float64)
            for start in range(0, n - chunk, chunk // 2):
                end = start + chunk
                seg = audio[start:end]
                f0 = float(np.median(center[start:end]))
                if f0 < 5.0 or f0 > sr / 2.5:
                    continue
                bw = max(2.0, f0 / filter_q)
                low = max(1.0, f0 - bw)
                high = min(sr / 2 - 1.0, f0 + bw)
                sos = signal.butter(2, [low, high], btype="band", fs=sr, output="sos")
                filtered = signal.sosfiltfilt(sos, seg)
                out[start:end] += filtered * signal.windows.hann(chunk)
            tracks[float(order)] = out
        return tracks

    def compare_to_model(
        self,
        audio: np.ndarray,
        rpm: np.ndarray,
        *,
        max_order: float = 12.0,
    ) -> dict[str, np.ndarray | float]:
        """
        Compare measured order spectrum to the engine's configured order amplitudes.
        """
        measured = self.order_spectrum(audio, rpm, max_order=max_order)
        model_orders = np.array(dominant_orders(self.engine, max_order=max_order))
        model_amp = order_amplitude_vector(self.engine, model_orders)
        model_amp /= np.max(model_amp) or 1.0

        # Interpolate measured onto model order grid.
        measured_interp = np.interp(
            model_orders,
            measured.orders,
            measured.magnitudes,
            left=0.0,
            right=0.0,
        )
        residual = measured_interp - model_amp
        mse = float(np.mean(residual ** 2))
        return {
            "orders": model_orders,
            "measured": measured_interp,
            "model": model_amp,
            "residual": residual,
            "mse": mse,
            "reference_rpm": measured.reference_rpm,
        }


def order_spectrum(
    audio: np.ndarray,
    rpm: np.ndarray,
    engine: EngineConfig,
    sample_rate: int,
    **kwargs,
) -> OrderSpectrumResult:
    """Convenience wrapper around OrderTracker.order_spectrum."""
    return OrderTracker(engine, sample_rate).order_spectrum(audio, rpm, **kwargs)


def stft_order_ridge(
    audio: np.ndarray,
    rpm: np.ndarray,
    order: float,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT magnitude slice along a theoretical order ridge f(t) = order × RPM(t)/60.

    Useful for visualizing how a specific order (e.g. 2× firing on inline-4) evolves
    during RPM transients.
    """
    f, t_stft, Z = signal.stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
    rpm_interp = np.interp(t_stft, np.arange(len(rpm)) / sample_rate, rpm)
    target_f = order_frequency_hz(rpm_interp, order)
    ridge_mag = np.zeros(len(t_stft), dtype=np.float64)
    for i, ft in enumerate(target_f):
        bin_idx = int(np.argmin(np.abs(f - ft)))
        ridge_mag[i] = np.abs(Z[bin_idx, i])
    return t_stft, target_f, ridge_mag
