"""Phase B: CNN length regression (legacy import path — use length_estimation.train)."""

from pathlib import Path

from length_estimation.config import PhaseBConfig
from length_estimation.src.phase_b.eval import run_eval
from length_estimation.src.phase_b.train import train_lovo, train_split

__all__ = ["PhaseBConfig", "train_split", "train_lovo", "run_eval"]


def run_phase_b_lovo(records, output_dir, cfg=None, target="length_m", device="cpu"):
    """Deprecated: use `python -m length_estimation.train --mode lovo`."""
    if target != "length_m":
        raise ValueError("Phase B is length-only; wheelbase target is not supported.")
    run_name = Path(output_dir).name

    return train_lovo(
        checkpoint_dir=Path(output_dir).parent,
        run_name=run_name,
        cfg=cfg or PhaseBConfig(),
        device=device,
    )
