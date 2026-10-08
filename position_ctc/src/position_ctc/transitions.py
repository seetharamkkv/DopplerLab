"""Graded velocity transitions for one axis.

A small change is cheap, a larger change costs the square of its size, and a
change past the jump limit is forbidden. The limit is ``a_max * dt`` in m/s,
not a fixed "one bin per frame" rule.

On this dataset's 86 Hz grid a passenger-car acceleration (about 5 m/s²)
moves a component by ~0.06 m/s per frame, which is far below the 2 m/s bin
width, so that literal bound would freeze every quantized path. The default
limit is three bins (6 m/s per frame). In a sample of these polylines, 99.86%
of component steps are within three bins; the rest are the corner spikes.
"""

from __future__ import annotations

import numpy as np

from position_ctc.alphabet import VEL_STEP_MPS
from position_ctc.bins import CENTERS_MPS

DEFAULT_JUMP_BINS = 3
DEFAULT_JUMP_MPS = DEFAULT_JUMP_BINS * VEL_STEP_MPS
DEFAULT_SCALE_MPS = VEL_STEP_MPS
DEFAULT_LAMBDA = 1.0


def jump_limit_mps(dt: float, a_max_mps2: float, slack_mps: float = 0.0) -> float:
    """Largest |Δv| allowed from one frame to the next, in m/s."""
    if dt <= 0:
        raise ValueError("dt")
    return float(a_max_mps2) * float(dt) + float(slack_mps)


def a_max_for_jump(jump_mps: float, dt: float) -> float:
    """Acceleration whose one-frame change equals ``jump_mps``."""
    if dt <= 0:
        raise ValueError("dt")
    return float(jump_mps) / float(dt)


def transition_matrix(
    dt: float,
    a_max_mps2: float,
    *,
    lam: float = DEFAULT_LAMBDA,
    scale_mps: float = DEFAULT_SCALE_MPS,
    slack_mps: float = 0.0,
    centers: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``ψ[i, j]`` for a move from bin ``i`` to bin ``j``.

    ``ψ = -λ (d / scale)²`` when ``d = |c_j - c_i|`` is within the jump limit,
    and ``-inf`` otherwise. Staying put scores 0. With ``scale`` equal to the
    bin width and a three-bin limit, the costs are ``0, -λ, -4λ, -9λ, -inf``.
    """
    if scale_mps <= 0:
        raise ValueError("scale_mps")
    centers = CENTERS_MPS if centers is None else np.asarray(centers, dtype=np.float64)
    limit = jump_limit_mps(dt, a_max_mps2, slack_mps)
    delta = np.abs(centers[None, :] - centers[:, None])
    psi = -float(lam) * (delta / float(scale_mps)) ** 2
    psi = psi.astype(np.float64)
    psi[delta > limit + 1e-6] = -np.inf
    return psi


def component_jump_fraction(velocity: np.ndarray, jump_mps: float) -> float:
    """Fraction of consecutive component steps with ``|Δv|`` above ``jump_mps``."""
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.shape[0] < 2:
        return 0.0
    delta = np.abs(np.diff(velocity, axis=0))
    return float(np.mean(delta > float(jump_mps) + 1e-6))


def forbidden_bin_fraction(velocity: np.ndarray, psi: np.ndarray) -> float:
    """Fraction of consecutive frames whose bins are illegal under ``psi``.

    ``velocity`` is one component in m/s. A three-bin mask still allows a
    6 m/s jump in one 12 ms frame, which is not a smoothness guarantee.
    """
    from position_ctc.bins import velocity_bin_index

    idx = velocity_bin_index(np.asarray(velocity, dtype=np.float64).reshape(-1))
    if idx.shape[0] < 2:
        return 0.0
    scores = np.asarray(psi, dtype=np.float64)[idx[:-1], idx[1:]]
    return float(np.mean(~np.isfinite(scores)))
