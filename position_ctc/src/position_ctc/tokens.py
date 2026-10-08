"""CTC target strings from a mic-centric state trajectory.

State columns follow DopplerSim Phase 1: ``[x_m, vx_mps, y_m, vy_mps]``.

The string is the initial velocity pair, then an acceleration pair at each
quantized velocity-bin change. Position is not a token. Frame indices of those
changes stay beside the string so Phase 0 can integrate them; CTC training
uses the collapsed string only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from position_ctc.alphabet import acc_token, quantize_velocity, vel_token
from position_ctc.integrate import hold_velocity_bins, integrate_velocity

MAX_TARGET_TOKENS = 40
MIN_TARGET_TOKENS = 2


@dataclass(frozen=True)
class MotionTokens:
    tokens: list[str]
    change_frames: list[int]
    vx_bins: np.ndarray
    vy_bins: np.ndarray
    xy_from_bins: np.ndarray

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


def motion_tokens(state: np.ndarray, dt: float) -> MotionTokens:
    """Build the CTC string and the position profile of the held velocity bins."""
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] < 4 or state.shape[0] < 2:
        raise ValueError("state must be (T, 4+) with T >= 2")
    vx = np.array([quantize_velocity(v) for v in state[:, 1]], dtype=np.float64)
    vy = np.array([quantize_velocity(v) for v in state[:, 3]], dtype=np.float64)
    held_x, held_y, changes = hold_velocity_bins(vx, vy)

    # Acceleration is the bin step divided by the time since the previous bin,
    # not the one-frame spike. A 2 m/s step in a single 10 ms frame is ~200 m/s²
    # and would clamp every token to the edge of the alphabet.
    tokens = [vel_token("VX", held_x[0]), vel_token("VY", held_y[0])]
    prev_frame = 0
    step = float(dt)
    for frame in changes:
        dur = max((frame - prev_frame) * step, step)
        tokens.append(acc_token("AX", (held_x[frame] - held_x[prev_frame]) / dur))
        tokens.append(acc_token("AY", (held_y[frame] - held_y[prev_frame]) / dur))
        prev_frame = frame

    xy = integrate_velocity(held_x, held_y, dt, x0=float(state[0, 0]), y0=float(state[0, 2]))
    return MotionTokens(
        tokens=tokens,
        change_frames=changes,
        vx_bins=held_x,
        vy_bins=held_y,
        xy_from_bins=xy,
    )


def string_length_ok(n_tokens: int) -> bool:
    return MIN_TARGET_TOKENS <= int(n_tokens) <= MAX_TARGET_TOKENS
