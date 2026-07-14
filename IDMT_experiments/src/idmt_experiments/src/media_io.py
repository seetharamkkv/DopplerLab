"""Load monoaural audio from wav or video containers (avi, mp4, ...)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from idmt_experiments.config import SR_NATIVE

VIDEO_EXTENSIONS = frozenset({".avi", ".mp4", ".mkv", ".mov", ".webm", ".wmv", ".m4v"})


def resolve_ffmpeg_exe() -> str:
    """System ffmpeg on PATH, else bundled binary from imageio-ffmpeg."""
    import shutil

    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "Video clips (.avi, etc.) need ffmpeg. Either install ffmpeg on PATH or run:\n"
            "  pip install imageio-ffmpeg"
        ) from exc


def load_mono_ffmpeg(path: Path | str, sr: int = SR_NATIVE) -> tuple[np.ndarray, int]:
    """Decode audio track to mono float32 via ffmpeg (full file, no trim)."""
    path = Path(path)
    ffmpeg = resolve_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-sn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {path.name}: {err or 'unknown error'}")
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg returned no audio for {path.name}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return y.astype(np.float32), sr


def is_video_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS
