"""Discover VS13 and DopplerSim clips. No audio copy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ctc_exp.audio import CROP_HALF_S, HOP, SR, frame_times
from ctc_exp.tokens import token_string


@dataclass
class Clip:
    clip_id: str
    dataset: str
    vehicle: str
    wav_path: Path
    speed_mps: float
    cpa_s: float
    cpa_m: float | None
    tokens: list[str]
    log_mel_path: Path | None = None
    range_m: np.ndarray | None = None
    times_s: np.ndarray | None = None
    xy_m: np.ndarray | None = None
    vel_mps: np.ndarray | None = None


def _parse_vs13_txt(txt: Path) -> tuple[float, float]:
    text = txt.read_text(encoding="utf-8").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1:
        parts = lines[0].split()
        return float(parts[0]), float(parts[1])
    return float(lines[0]), float(lines[1])


def _wav_duration_s(wav: Path) -> float | None:
    import wave

    try:
        with wave.open(str(wav), "rb") as handle:
            return float(handle.getnframes()) / float(handle.getframerate())
    except Exception:
        return None


def load_vs13(root: Path) -> list[Clip]:
    clips: list[Clip] = []
    for vdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for wav in sorted(vdir.glob("*.wav")):
            if wav.stem.lower().startswith("train_valid"):
                continue
            txt = wav.with_suffix(".txt")
            if not txt.is_file():
                continue
            speed_kmh, cpa_s = _parse_vs13_txt(txt)
            speed_mps = float(speed_kmh) / 3.6
            duration_s = _wav_duration_s(wav) or max(8.0, float(cpa_s) + CROP_HALF_S)
            t0 = max(0.0, float(cpa_s) - CROP_HALF_S)
            t1 = min(duration_s, float(cpa_s) + CROP_HALF_S)
            n = max(int(round((t1 - t0) * SR / HOP)), 8)
            times = frame_times(n, t0)
            toks = token_string(
                speed_mps=speed_mps,
                cpa_s=cpa_s,
                times_s=times,
                include_range=False,
            )
            clips.append(
                Clip(
                    clip_id=wav.stem,
                    dataset="vs13",
                    vehicle=vdir.name,
                    wav_path=wav,
                    speed_mps=speed_mps,
                    cpa_s=cpa_s,
                    cpa_m=None,
                    tokens=toks,
                    times_s=times,
                )
            )
    if not clips:
        raise FileNotFoundError(f"no VS13 wav+txt under {root}")
    return clips


def _row_speed_mps(row) -> float:
    if hasattr(row, "speed_mps"):
        val = row.speed_mps
        if val is not None and val == val:
            return float(val)
    return float(row.speed_kmph) / 3.6


def _load_sim_like(root: Path, *, dataset: str, with_state: bool) -> list[Clip]:
    csv_path = root / "dataset.csv"
    df = pd.read_csv(csv_path)
    clips: list[Clip] = []
    for row in df.itertuples(index=False):
        sample_id = str(row.sample_id)
        sample_dir = root / "audio_clips" / sample_id
        wav = sample_dir / str(row.filename)
        if not wav.is_file():
            found = list(sample_dir.glob("*.wav"))
            if not found:
                continue
            wav = found[0]
        times = np.asarray(np.load(sample_dir / "metadata" / "frame_times.npy"), dtype=np.float64)
        range_m = np.asarray(np.load(sample_dir / "metadata" / "range_m.npy"), dtype=np.float64)
        speed = _row_speed_mps(row)
        cpa_s = float(row.cpa_time_sec)
        cpa_m = float(row.cpa_distance_m)
        toks = token_string(
            speed_mps=speed,
            cpa_s=cpa_s,
            times_s=times,
            range_m=range_m,
            include_range=True,
        )
        mel = sample_dir / "spectrograms" / "log_mel.npy"
        xy = vel = None
        if with_state:
            canon = sample_dir / "metadata" / "canonical_state_frames.npy"
            state_path = canon if canon.is_file() else sample_dir / "metadata" / "state_frames.npy"
            state = np.asarray(np.load(state_path), dtype=np.float64)
            xy = np.column_stack([state[:, 0], state[:, 2]])
            vel = np.column_stack([state[:, 1], state[:, 3]])
        clips.append(
            Clip(
                clip_id=sample_id,
                dataset=dataset,
                vehicle=str(row.vehicle_class),
                wav_path=wav,
                speed_mps=speed,
                cpa_s=cpa_s,
                cpa_m=cpa_m,
                tokens=toks,
                log_mel_path=mel if mel.is_file() else None,
                range_m=range_m,
                times_s=times,
                xy_m=xy,
                vel_mps=vel,
            )
        )
    if not clips:
        raise FileNotFoundError(f"no clips under {root}")
    return clips


def load_sim(root: Path) -> list[Clip]:
    return _load_sim_like(root, dataset="sim", with_state=False)


def load_straight_audio_only(root: Path) -> list[Clip]:
    """Straight-path clips from wav + dataset.csv kinematics only.

    Does not read state_frames, range_m, or frame_times. Position labels
    are derived later from speed, CPA, and spectrogram times.
    """
    csv_path = root / "dataset.csv"
    df = pd.read_csv(csv_path)
    clips: list[Clip] = []
    seen: set[str] = set()
    for row in df.itertuples(index=False):
        sample_id = str(row.sample_id)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        traj = str(getattr(row, "trajectory_type", "straight"))
        if traj and traj != "straight":
            continue
        sample_dir = root / "audio_clips" / sample_id
        wav = sample_dir / str(row.filename)
        if not wav.is_file():
            found = list(sample_dir.glob("*.wav"))
            if not found:
                continue
            wav = found[0]
        mel = sample_dir / "spectrograms" / "log_mel.npy"
        clips.append(
            Clip(
                clip_id=sample_id,
                dataset="sim_straight",
                vehicle=str(row.vehicle_class),
                wav_path=wav,
                speed_mps=_row_speed_mps(row),
                cpa_s=float(row.cpa_time_sec),
                cpa_m=float(row.cpa_distance_m),
                tokens=[],
                log_mel_path=mel if mel.is_file() else None,
            )
        )
    if not clips:
        raise FileNotFoundError(f"no straight clips under {root}")
    return clips


def load_unconstrained(root: Path) -> list[Clip]:
    return _load_sim_like(root, dataset="unconstrained", with_state=True)
