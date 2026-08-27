"""Path helpers for the traj_reconstruction track."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


def resolve_phase1_root(path: Path) -> Path:
    """Accept a Phase 1 dir, a sample_* dir, or a render dir containing phase1/."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if (path / "spectrograms" / "stft.npy").is_file():
        return path
    if (path / "phase1" / "spectrograms" / "stft.npy").is_file():
        return path / "phase1"
    if path.name.startswith("sample_") and (path / "spectrograms" / "stft.npy").is_file():
        return path
    raise FileNotFoundError(
        f"No Phase 1 package under {path} "
        "(expected spectrograms/stft.npy or phase1/spectrograms/stft.npy)"
    )
