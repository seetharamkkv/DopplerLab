"""Viterbi velocity, then trapezoid integration from the supplied position.

The anchored integral is the only trajectory. Framewise argmax is integrated
from the same anchor so decoding can be compared with the raw classifier.
There is no direct position head and no bearing/range head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from position_ctc.bins import CENTERS_MPS
from position_ctc.integrate import integrate_trapezoid
from position_ctc.motion_model import MotionNet
from position_ctc.viterbi import viterbi


@dataclass(frozen=True)
class Trajectory:
    vx: np.ndarray
    vy: np.ndarray
    xy: np.ndarray
    vx_argmax: np.ndarray
    vy_argmax: np.ndarray
    xy_argmax: np.ndarray
    log_px: np.ndarray
    log_py: np.ndarray


def _axis(logp: np.ndarray, psi: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = viterbi(logp, psi)
    greedy = np.argmax(logp, axis=1)
    return centers[path], centers[greedy]


def reconstruct(
    log_px: np.ndarray,
    log_py: np.ndarray,
    dt: float,
    x0: float,
    y0: float,
    psi: np.ndarray,
    centers: np.ndarray | None = None,
) -> Trajectory:
    """Decode each axis and integrate bin centers from ``(x0, y0)``.

    ``log_px`` and ``log_py`` are ``(T, K)`` log-probabilities. ``(x0, y0)``
    is the supplied initial position in meters, canonical frame. Both the
    Viterbi path and the framewise argmax are integrated from that point.
    ``xy`` is the Viterbi integral and is the trajectory to report.
    """
    centers = CENTERS_MPS if centers is None else np.asarray(centers, dtype=np.float64)
    vx, vx_argmax = _axis(log_px, psi, centers)
    vy, vy_argmax = _axis(log_py, psi, centers)
    xy = integrate_trapezoid(vx, vy, dt, x0=x0, y0=y0)
    xy_argmax = integrate_trapezoid(vx_argmax, vy_argmax, dt, x0=x0, y0=y0)
    xy[0, 0] = float(x0)
    xy[0, 1] = float(y0)
    xy_argmax[0, 0] = float(x0)
    xy_argmax[0, 1] = float(y0)
    return Trajectory(
        vx=vx,
        vy=vy,
        xy=xy,
        vx_argmax=vx_argmax,
        vy_argmax=vy_argmax,
        xy_argmax=xy_argmax,
        log_px=np.asarray(log_px, dtype=np.float64),
        log_py=np.asarray(log_py, dtype=np.float64),
    )


@torch.no_grad()
def predict_trajectory(
    model: MotionNet,
    feat: np.ndarray,
    state: np.ndarray,
    dt: float,
    psi: np.ndarray,
    device: torch.device,
) -> Trajectory:
    """Run one clip. The network and the integrator share ``state[0]``'s position."""
    model.eval()
    frames = int(feat.shape[-1])
    n = min(frames, int(state.shape[0]))
    batch = torch.from_numpy(np.ascontiguousarray(feat[:, :n])).unsqueeze(0).to(device)
    lengths = torch.tensor([n], dtype=torch.long, device=device)
    origin = torch.tensor([[float(state[0, 0]), float(state[0, 2])]], dtype=torch.float32, device=device)
    out = model(batch, lengths, origin)
    log_px = out["vel_x"][0, :n].log_softmax(-1).cpu().numpy()
    log_py = out["vel_y"][0, :n].log_softmax(-1).cpu().numpy()
    return reconstruct(
        log_px,
        log_py,
        dt,
        x0=float(state[0, 0]),
        y0=float(state[0, 2]),
        psi=psi,
    )
