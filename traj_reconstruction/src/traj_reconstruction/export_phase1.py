"""Write DopplerSim-compatible Phase 1 sample packages (simulated)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from traj_reconstruction.contract import (
    ACOUSTIC_PRIMARY_RELPATH,
    CANONICAL_STATE_FRAMES_RELPATH,
    FRAME_TIMES_RELPATH,
    PATH_POLYLINE_RELPATH,
    PATH_TYPE_FREE_2D,
    PATH_TYPE_STRAIGHT,
    SCHEMA_RELPATH,
    SPEC_SR_HZ,
    STATE_FRAMES_RELPATH,
    STFT_HOP_LENGTH,
    STFT_N_FFT,
    TIER1,
)
from traj_reconstruction.kinematics import (
    canonical_state_frames,
    compute_stft_db,
    derived_from_state,
    interpolate_state,
    stft_frame_times,
    stft_n_frames,
)
from traj_reconstruction.path_families import PathSpec
from traj_reconstruction.synthesize import synthesize_tone_on_path


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    try:
        import soundfile as sf

        sf.write(str(path), audio.astype(np.float32), int(sr))
    except ImportError:
        # Minimal WAV writer (float32 PCM via wave + struct fallback as int16).
        import wave

        pcm = np.clip(audio, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(pcm_i16.tobytes())


def export_phase1_sample(
    sample_dir: Path,
    spec: PathSpec,
    *,
    sample_id: str,
    seed: int,
    tier: str = TIER1,
    f0_hz: float = 500.0,
    sr: int = SPEC_SR_HZ,
    pad_s: float = 0.25,
) -> dict[str, Any]:
    """Synthesize Tier-1 tone on ``spec`` and write a Phase 1 package."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "spectrograms").mkdir(exist_ok=True)
    (sample_dir / "metadata").mkdir(exist_ok=True)

    synth = synthesize_tone_on_path(
        spec.polyline,
        speed_mps=spec.speed_mps,
        f0_hz=f0_hz,
        sr=sr,
        pad_s=pad_s,
    )
    audio = synth["audio"]
    traj = synth["trajectory"]
    n = len(audio)
    n_frames = stft_n_frames(n, STFT_HOP_LENGTH)
    frame_times = stft_frame_times(n_frames, sr=sr, hop_length=STFT_HOP_LENGTH)
    state_frames = interpolate_state(traj["t"], traj["state"], frame_times)
    canonical, canon_meta = canonical_state_frames(state_frames)
    derived = derived_from_state(state_frames, frame_times)
    stft_db = compute_stft_db(audio, sr=sr, n_fft=STFT_N_FFT, hop_length=STFT_HOP_LENGTH)
    if stft_db.shape[1] != state_frames.shape[0]:
        raise RuntimeError(
            f"STFT T={stft_db.shape[1]} != state T={state_frames.shape[0]}"
        )

    path_type = PATH_TYPE_STRAIGHT if spec.family == "straight" else PATH_TYPE_FREE_2D
    wav_name = f"{sample_id}_{spec.family}_{spec.speed_mps:.1f}mps.wav"
    _write_wav(sample_dir / wav_name, audio, sr)

    np.save(sample_dir / ACOUSTIC_PRIMARY_RELPATH, stft_db.astype(np.float32))
    np.save(sample_dir / STATE_FRAMES_RELPATH, state_frames.astype(np.float32))
    np.save(sample_dir / CANONICAL_STATE_FRAMES_RELPATH, canonical.astype(np.float32))
    np.save(sample_dir / FRAME_TIMES_RELPATH, frame_times.astype(np.float32))
    np.save(sample_dir / PATH_POLYLINE_RELPATH, spec.polyline.astype(np.float32))
    np.save(sample_dir / "metadata/state.npy", traj["state"].astype(np.float32))
    np.save(sample_dir / "metadata/state_times.npy", traj["t"].astype(np.float32))
    np.save(sample_dir / "metadata/speed_series_mps.npy", derived["speed_mps"])
    np.save(sample_dir / "metadata/range_m.npy", derived["range_m"])
    np.save(sample_dir / "metadata/radial_velocity_mps.npy", derived["radial_velocity_mps"])
    np.save(
        sample_dir / "metadata/cpa_time.npy",
        np.array([derived["cpa_time_sec"]], dtype=np.float64),
    )
    np.save(
        sample_dir / "metadata/cpa_distance_m.npy",
        np.array([derived["cpa_distance_m"]], dtype=np.float64),
    )

    schema = {
        "schema_version": 1,
        "data_scope": "simulated_tier1_tone",
        "path_type": path_type,
        "path_family": spec.family,
        "tier": tier,
        "pipeline": "traj_reconstruction_tier1",
        "acoustics": {
            "type": "pure_tone",
            "f0_hz": float(f0_hz),
            "spreading": "1/r",
            "retarded_time": True,
        },
        "kinematics": {
            "speed_profile": "constant along polyline",
            "speed_mps": float(spec.speed_mps),
            "pad_s": float(pad_s),
            "notes": spec.notes,
        },
        "stft": {
            "sr_hz": int(sr),
            "n_fft": int(STFT_N_FFT),
            "hop_length": int(STFT_HOP_LENGTH),
            "window": "hann",
        },
        "canonical_gauge": canon_meta,
        "sample_id": sample_id,
        "seed": int(seed),
    }
    (sample_dir / SCHEMA_RELPATH).write_text(json.dumps(schema, indent=2))

    return {
        "sample_id": sample_id,
        "path_type": path_type,
        "path_family": spec.family,
        "tier": tier,
        "speed_mps": float(spec.speed_mps),
        "cpa_time_sec": float(derived["cpa_time_sec"]),
        "cpa_distance_m": float(derived["cpa_distance_m"]),
        "n_frames": int(state_frames.shape[0]),
        "wav": wav_name,
        "seed": int(seed),
        "notes": spec.notes,
    }
