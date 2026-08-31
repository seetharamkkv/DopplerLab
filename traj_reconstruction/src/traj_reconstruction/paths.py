"""Path helpers for the traj_reconstruction track."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# Live best weights the frontend / infer scripts load while training continues.
DEFAULT_ORBIT_MLP_BEST = PACKAGE_ROOT / "checkpoints" / "orbit_mlp_path2d_1000.npz"
DEFAULT_ORBIT_CNN_BEST = PACKAGE_ROOT / "checkpoints" / "orbit_cnn_path2d_1000.npz"
DEFAULT_ORBIT_SEQ_BEST = PACKAGE_ROOT / "checkpoints" / "orbit_seq_path2d_1000.npz"


def default_learned_checkpoint() -> Path:
    """Prefer the learned checkpoint with the best recorded val orbit RMS."""
    import json

    def _rms(path: Path) -> float:
        if not path.is_file():
            return float("inf")
        status = path.with_name(path.stem + ".status.json")
        if status.is_file():
            try:
                val = json.loads(status.read_text()).get("best_val_orbit_rms")
                if val is not None:
                    return float(val)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        return float("inf")

    ranked = sorted(
        (DEFAULT_ORBIT_CNN_BEST, DEFAULT_ORBIT_SEQ_BEST, DEFAULT_ORBIT_MLP_BEST),
        key=_rms,
    )
    for path in ranked:
        if path.is_file():
            return path
    return DEFAULT_ORBIT_CNN_BEST


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
