"""Read DopplerSim Phase 1 sample folders (simulated data only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from traj_reconstruction.contract import (
    ACOUSTIC_PRIMARY_RELPATH,
    CANONICAL_STATE_FRAMES_RELPATH,
    DATA_SCOPE,
    FRAME_TIMES_RELPATH,
    INFERENCE_ALLOWED_RELPATHS,
    INFERENCE_FORBIDDEN_RELPATHS,
    PATH_POLYLINE_RELPATH,
    PATH_TYPE_STRAIGHT,
    SCHEMA_RELPATH,
    STATE_FRAMES_RELPATH,
)
from traj_reconstruction.paths import resolve_phase1_root


class DatasetError(ValueError):
    """Phase 1 layout or A/s alignment is invalid."""


@dataclass(frozen=True)
class Phase1Sample:
    """One simulated Phase 1 clip.

    ``stft_db`` is inference-allowed. State arrays are train/eval GT only —
    never pass them into a model at inference.
    """

    root: Path
    stft_db: np.ndarray
    state_frames: np.ndarray
    frame_times: np.ndarray
    canonical_state_frames: np.ndarray | None
    path_polyline: np.ndarray | None
    schema: dict[str, Any] | None
    wav_path: Path | None
    path_type: str
    path_family: str
    tier: str
    sample_id: str
    data_scope: str = DATA_SCOPE

    @property
    def n_frames(self) -> int:
        return int(self.state_frames.shape[0])


@dataclass(frozen=True)
class Phase1Batch:
    """Manifest-backed iterator over a simulated freehand / Tier-1 batch."""

    root: Path
    rows: tuple[dict[str, str], ...]

    @classmethod
    def from_dir(cls, batch_dir: Path | str) -> Phase1Batch:
        import csv

        batch_dir = Path(batch_dir).expanduser().resolve()
        manifest = batch_dir / "dataset.csv"
        if not manifest.is_file():
            raise DatasetError(f"missing dataset.csv under {batch_dir}")
        with manifest.open(newline="") as f:
            rows = tuple(csv.DictReader(f))
        if not rows:
            raise DatasetError(f"empty dataset.csv under {batch_dir}")
        return cls(root=batch_dir, rows=rows)

    def __len__(self) -> int:
        return len(self.rows)

    def sample_dir(self, sample_id: str) -> Path:
        return self.root / "audio_clips" / sample_id

    def load(self, index: int) -> Phase1Sample:
        row = self.rows[index]
        return load_phase1_sample(self.sample_dir(row["sample_id"]))

    def iter_samples(self) -> Iterator[Phase1Sample]:
        for row in self.rows:
            yield load_phase1_sample(self.sample_dir(row["sample_id"]))


@dataclass(frozen=True)
class InferenceBundle:
    """Audio-only payload safe for model inference (simulated or later real)."""

    root: Path
    stft_db: np.ndarray | None
    wav_path: Path | None

    def __post_init__(self) -> None:
        if self.stft_db is None and self.wav_path is None:
            raise DatasetError("InferenceBundle needs stft_db and/or wav_path")


def _load_npy(root: Path, rel: str) -> np.ndarray:
    path = root / rel
    if not path.is_file():
        raise DatasetError(f"missing {rel} under {root}")
    return np.load(path)


def _find_wav(root: Path) -> Path | None:
    wavs = sorted(root.glob("*.wav"))
    return wavs[0] if wavs else None


def _path_type_from_schema(schema: dict[str, Any] | None) -> str:
    if not schema:
        return PATH_TYPE_STRAIGHT
    for key in ("path_type", "trajectory_type", "pipeline"):
        val = schema.get(key)
        if isinstance(val, str) and val:
            if val in ("path2d", "free_path_2d"):
                return "free_path_2d"
            if val in ("path3d", "free_path_3d"):
                return "free_path_3d"
            return val
    kin = schema.get("kinematics")
    if isinstance(kin, dict):
        pt = kin.get("path_type") or kin.get("trajectory_type")
        if isinstance(pt, str) and pt:
            return pt
    return PATH_TYPE_STRAIGHT


def load_phase1_sample(path: Path | str) -> Phase1Sample:
    """Load a simulated Phase 1 package (render phase1/ or batch sample_*)."""
    root = resolve_phase1_root(Path(path))
    stft = _load_npy(root, ACOUSTIC_PRIMARY_RELPATH)
    state = _load_npy(root, STATE_FRAMES_RELPATH)
    times = _load_npy(root, FRAME_TIMES_RELPATH)

    if state.shape[0] != times.shape[0]:
        raise DatasetError(
            f"state_frames T={state.shape[0]} vs frame_times T={times.shape[0]}"
        )
    # STFT is often (F, T) or (T, F); align on time axis vs state.
    if stft.ndim != 2:
        raise DatasetError(f"stft must be 2-D, got {stft.shape}")
    t_stft = stft.shape[1] if stft.shape[0] != state.shape[0] else stft.shape[0]
    # Prefer (freq, time) convention from DopplerSim export.
    if stft.shape[1] == state.shape[0]:
        t_stft = stft.shape[1]
    elif stft.shape[0] == state.shape[0]:
        t_stft = stft.shape[0]
    else:
        raise DatasetError(
            f"STFT shape {stft.shape} not aligned with state T={state.shape[0]}"
        )
    if t_stft != state.shape[0]:
        raise DatasetError("internal STFT/state alignment error")

    canonical = None
    can_path = root / CANONICAL_STATE_FRAMES_RELPATH
    if can_path.is_file():
        canonical = np.load(can_path)
        if canonical.shape[0] != state.shape[0]:
            raise DatasetError(
                f"canonical_state_frames T={canonical.shape[0]} "
                f"!= state T={state.shape[0]}"
            )

    polyline = None
    poly_path = root / PATH_POLYLINE_RELPATH
    if poly_path.is_file():
        polyline = np.load(poly_path)

    schema = None
    schema_path = root / SCHEMA_RELPATH
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text())

    return Phase1Sample(
        root=root,
        stft_db=stft,
        state_frames=state,
        frame_times=times,
        canonical_state_frames=canonical,
        path_polyline=polyline,
        schema=schema,
        wav_path=_find_wav(root),
        path_type=_path_type_from_schema(schema),
        path_family=str((schema or {}).get("path_family") or _path_type_from_schema(schema)),
        tier=str((schema or {}).get("tier") or "unknown"),
        sample_id=str((schema or {}).get("sample_id") or root.name),
    )


def to_inference_bundle(sample: Phase1Sample) -> InferenceBundle:
    """Strip all GT — only audio/STFT remain."""
    return InferenceBundle(
        root=sample.root,
        stft_db=sample.stft_db,
        wav_path=sample.wav_path,
    )


def assert_inference_safe(relpaths: list[str] | tuple[str, ...]) -> None:
    """Raise if any provided relative path is forbidden or otherwise disallowed."""
    forbidden = set(INFERENCE_FORBIDDEN_RELPATHS)
    allowed = set(INFERENCE_ALLOWED_RELPATHS)
    for p in relpaths:
        if p in forbidden or p.startswith("metadata/"):
            raise DatasetError(f"inference-forbidden path: {p}")
        if p in allowed or p.endswith(".wav"):
            continue
        raise DatasetError(f"disallowed inference path: {p}")


def iter_batch_samples(batch_dir: Path | str) -> Iterator[Path]:
    """Yield ``audio_clips/sample_*`` directories under a DopplerSim batch export."""
    batch_dir = Path(batch_dir).expanduser().resolve()
    clips = batch_dir / "audio_clips"
    if not clips.is_dir():
        raise DatasetError(f"no audio_clips/ under {batch_dir}")
    for sample in sorted(clips.glob("sample_*")):
        if sample.is_dir() and (sample / "spectrograms" / "stft.npy").is_file():
            yield sample
