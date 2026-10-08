"""Quantized velocity and acceleration alphabet.

Bin widths are the Phase 0 proposal from the position-detection plan. Lock
them after a histogram on a real DopplerSim batch; do not treat them as final.
"""

from __future__ import annotations

import numpy as np

BLANK = "_"

VEL_STEP_MPS = 2.0
VEL_MIN_MPS = -30.0
VEL_MAX_MPS = 30.0

ACC_STEP_MPS2 = 0.5
ACC_MIN_MPS2 = -10.0
ACC_MAX_MPS2 = 10.0


def _grid(step: float, lo: float, hi: float) -> tuple[float, ...]:
    n = int(round((hi - lo) / step))
    return tuple(float(lo + i * step) for i in range(n + 1))


VEL_CENTERS_MPS = _grid(VEL_STEP_MPS, VEL_MIN_MPS, VEL_MAX_MPS)
ACC_CENTERS_MPS2 = _grid(ACC_STEP_MPS2, ACC_MIN_MPS2, ACC_MAX_MPS2)


def quantize(value: float, step: float, lo: float, hi: float) -> float:
    k = int(np.round(float(value) / step))
    return float(np.clip(k * step, lo, hi))


def quantize_velocity(mps: float) -> float:
    return quantize(mps, VEL_STEP_MPS, VEL_MIN_MPS, VEL_MAX_MPS)


def quantize_acceleration(mps2: float) -> float:
    return quantize(mps2, ACC_STEP_MPS2, ACC_MIN_MPS2, ACC_MAX_MPS2)


def vel_token(axis: str, mps: float) -> str:
    if axis not in ("VX", "VY"):
        raise ValueError(axis)
    return f"{axis}{quantize_velocity(mps):+.0f}"


def acc_token(axis: str, mps2: float) -> str:
    if axis not in ("AX", "AY"):
        raise ValueError(axis)
    return f"{axis}{quantize_acceleration(mps2):+.1f}"


def parse_vel_token(token: str) -> tuple[str, float]:
    axis = token[:2]
    if axis not in ("VX", "VY"):
        raise ValueError(token)
    return axis, float(token[2:])


def parse_acc_token(token: str) -> tuple[str, float]:
    axis = token[:2]
    if axis not in ("AX", "AY"):
        raise ValueError(token)
    return axis, float(token[2:])


def build_alphabet() -> list[str]:
    tokens = [BLANK]
    for center in VEL_CENTERS_MPS:
        tokens.append(f"VX{center:+.0f}")
        tokens.append(f"VY{center:+.0f}")
    for center in ACC_CENTERS_MPS2:
        tokens.append(f"AX{center:+.1f}")
        tokens.append(f"AY{center:+.1f}")
    return tokens


ALPHABET = build_alphabet()
TOKEN_TO_ID = {token: i for i, token in enumerate(ALPHABET)}
ID_TO_TOKEN = {i: token for token, i in TOKEN_TO_ID.items()}
BLANK_ID = 0
N_CLASS = len(ALPHABET)
