"""Best velocity-bin path under a framewise score plus a transition score.

Viterbi keeps, for every bin, the best partial path that ends there. That is
the dynamic program for this chain: once two histories share a bin, the one
with the worse score can never win, because future scores depend only on the
current bin. A greedy choice commits to one bin and cannot revise it.
"""

from __future__ import annotations

import numpy as np


def viterbi(logp: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Return the best bin index at each frame.

    ``logp`` is ``(T, K)`` and ``psi`` is ``(K, K)`` with ``psi[i, j]`` the
    score of moving from ``i`` to ``j``. Forbidden moves are ``-inf``.
    """
    logp = np.asarray(logp, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if logp.ndim != 2:
        raise ValueError("logp")
    frames, n_bins = logp.shape
    if psi.shape != (n_bins, n_bins):
        raise ValueError("psi")
    if frames == 0:
        raise ValueError("empty logp")
    back = np.zeros((frames, n_bins), dtype=np.int64)
    dp = np.empty((frames, n_bins), dtype=np.float64)
    dp[0] = logp[0]
    for t in range(1, frames):
        score = dp[t - 1][:, None] + psi
        back[t] = np.argmax(score, axis=0)
        chosen = score[back[t], np.arange(n_bins)]
        dp[t] = logp[t] + chosen
    path = np.empty(frames, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def constrained_greedy(logp: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Pick the best legal successor of the bin chosen at the previous frame."""
    logp = np.asarray(logp, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    frames, n_bins = logp.shape
    if frames == 0 or n_bins < 1:
        raise ValueError("logp")
    path = np.empty(frames, dtype=np.int64)
    path[0] = int(np.argmax(logp[0]))
    for t in range(1, frames):
        path[t] = int(np.argmax(logp[t] + psi[path[t - 1]]))
    return path
