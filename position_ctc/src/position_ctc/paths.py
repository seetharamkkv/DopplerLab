"""Resolve clip directories without baking in a local username or home path."""

from __future__ import annotations

import os
from pathlib import Path

ENV_AUDIO_CLIPS = "DOPPLER_AUDIO_CLIPS"
ENV_PATH2D_CLIPS = "DOPPLER_PATH2D_CLIPS"


def resolve_audio_clips(
    explicit: Path | str | None = None,
    *,
    env: str = ENV_AUDIO_CLIPS,
) -> Path:
    """Return the audio_clips directory from --data or an environment variable."""
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser()
    raw = os.environ.get(env, "").strip()
    if raw:
        return Path(raw).expanduser()
    raise SystemExit(
        f"Pass --data /path/to/audio_clips or set {env} to that directory."
    )
