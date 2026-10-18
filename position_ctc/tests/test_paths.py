import os
from pathlib import Path

import pytest

from position_ctc.paths import ENV_AUDIO_CLIPS, ENV_PATH2D_CLIPS, resolve_audio_clips


def test_resolve_prefers_explicit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_AUDIO_CLIPS, str(tmp_path / "env"))
    chosen = tmp_path / "explicit"
    assert resolve_audio_clips(chosen) == chosen


def test_resolve_uses_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_AUDIO_CLIPS, str(tmp_path))
    assert resolve_audio_clips() == tmp_path


def test_resolve_path2d_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_AUDIO_CLIPS, raising=False)
    monkeypatch.setenv(ENV_PATH2D_CLIPS, str(tmp_path))
    assert resolve_audio_clips(env=ENV_PATH2D_CLIPS) == tmp_path


def test_resolve_requires_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_AUDIO_CLIPS, raising=False)
    with pytest.raises(SystemExit, match=ENV_AUDIO_CLIPS):
        resolve_audio_clips()
