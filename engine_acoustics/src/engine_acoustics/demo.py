#!/usr/bin/env python3
"""Demonstrate engine acoustic synthesis and order tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from engine_acoustics.config import DEFAULT_OUTPUT_DIR, CylinderLayout, EngineConfig, SynthesisConfig
from engine_acoustics.order_tracking import OrderTracker, stft_order_ridge
from engine_acoustics.physics import (
    crank_frequency_hz,
    firing_frequency_hz,
    order_frequency_hz,
)
from engine_acoustics.synthesis import EngineSynthesizer, ramp_rpm, step_rpm


def run_demo(output_dir: Path, *, show: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    engine_i4 = EngineConfig(
        num_cylinders=4,
        firing_order=(1, 3, 4, 2),
        layout=CylinderLayout.INLINE,
    )
    syn_cfg = SynthesisConfig(duration_s=3.0, load=0.75, sample_rate=22050)
    synth = EngineSynthesizer(engine_i4, syn_cfg)

    rpm_fn = ramp_rpm(800.0, 4500.0, duration_s=2.5)
    audio, meta = synth.synthesize(rpm_fn, mode="hybrid")

    wav_path = output_dir / "inline4_ramp_hybrid.wav"
    try:
        import soundfile as sf

        sf.write(wav_path, audio, syn_cfg.sample_rate)
        print(f"Wrote {wav_path}")
    except ImportError:
        print("soundfile not installed — skipping WAV export")

    tracker = OrderTracker(engine_i4, syn_cfg.sample_rate)
    comparison = tracker.compare_to_model(audio, meta["rpm"])

    t_ridge, _f_ridge, ridge_mag = stft_order_ridge(
        audio, meta["rpm"], order=2.0, sample_rate=syn_cfg.sample_rate
    )

    engine_v6 = EngineConfig(
        num_cylinders=6,
        firing_order=(1, 4, 2, 5, 3, 6),
        layout=CylinderLayout.V,
        bank_angle_deg=60.0,
    )
    EngineSynthesizer(engine_v6, syn_cfg).synthesize(step_rpm([(0.0, 2500.0)]), mode="hybrid")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Four-stroke engine acoustic model — inline-4 RPM ramp")

    ax = axes[0, 0]
    excerpt = slice(0, int(0.15 * syn_cfg.sample_rate))
    ax.plot(meta["time_s"][excerpt], audio[excerpt], lw=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform (first 150 ms)")

    ax = axes[0, 1]
    ax.plot(meta["time_s"], meta["rpm"], color="tab:orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.set_title("RPM profile")

    ax = axes[1, 0]
    plt.specgram(audio, NFFT=1024, Fs=syn_cfg.sample_rate, noverlap=768, cmap="magma")
    ax.set_ylim(0, 400)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram (0–400 Hz)")

    ax = axes[1, 1]
    ax.plot(comparison["orders"], comparison["measured"], "o-", label="synthesized")
    ax.plot(comparison["orders"], comparison["model"], "s--", label="model template")
    ax.set_xlabel("Order (× crank)")
    ax.set_ylabel("Normalized magnitude")
    ax.set_title(f"Order spectrum @ {comparison['reference_rpm']:.0f} RPM mean")
    ax.legend()

    ax = axes[2, 0]
    ax.plot(t_ridge, ridge_mag)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ridge magnitude")
    ax.set_title("STFT ridge along 2× order (I4 firing fundamental)")

    ax = axes[2, 1]
    rpm_ref = np.linspace(500, 5000, 50)
    ax.plot(rpm_ref, firing_frequency_hz(rpm_ref, 4), label="I4 firing Hz")
    ax.plot(rpm_ref, firing_frequency_hz(rpm_ref, 6), label="V6 firing Hz")
    ax.plot(rpm_ref, crank_frequency_hz(rpm_ref), "--", label="1× crank Hz", alpha=0.6)
    ax.set_xlabel("RPM")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Firing vs crank frequency")
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / "engine_acoustics_demo.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print("\n--- Frequency reference at 3000 RPM ---")
    rpm0 = 3000.0
    print(f"  Crank (1×):     {order_frequency_hz(rpm0, 1.0):.1f} Hz")
    print(f"  I4 firing (2×): {order_frequency_hz(rpm0, 2.0):.1f} Hz")
    print(f"  V6 firing (3×): {order_frequency_hz(rpm0, 3.0):.1f} Hz")
    print(f"  I4 4th order:   {order_frequency_hz(rpm0, 4.0):.1f} Hz")
    print(f"\nOrder tracking MSE (synth vs template): {comparison['mse']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "demo",
        help="Directory for WAV and PNG outputs",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()
    run_demo(args.output, show=args.show)


if __name__ == "__main__":
    main()
