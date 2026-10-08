"""Shared CTC alphabet: blank, phase, speed bins, range bins."""

from __future__ import annotations

BLANK = "_"
PHASES = ("SIL", "APR", "CPA", "REC")
SPEED_BINS_MPS = tuple(range(10, 55, 5))  # V10 .. V50
SPEED_TOKENS = tuple(f"V{v:02d}" for v in SPEED_BINS_MPS)
N_SPEED = len(SPEED_TOKENS)
RANGE_BINS_M = (5, 10, 20, 40, 80)


def speed_token(speed_mps: float) -> str:
    snapped = int(round(float(speed_mps) / 5.0) * 5)
    snapped = min(SPEED_BINS_MPS[-1], max(SPEED_BINS_MPS[0], snapped))
    return f"V{snapped:02d}"


def range_token(range_m: float) -> str:
    r = float(range_m)
    for edge in RANGE_BINS_M:
        if r <= edge:
            return f"R{edge:02d}"
    return f"R{RANGE_BINS_M[-1]:02d}"


def build_alphabet() -> list[str]:
    tokens = [BLANK, *PHASES]
    tokens.extend(f"V{v:02d}" for v in SPEED_BINS_MPS)
    tokens.extend(f"R{e:02d}" for e in RANGE_BINS_M)
    return tokens


ALPHABET = build_alphabet()
TOKEN_TO_ID = {t: i for i, t in enumerate(ALPHABET)}
ID_TO_TOKEN = {i: t for t, i in TOKEN_TO_ID.items()}
N_CLASS = len(ALPHABET)
BLANK_ID = 0


def ids_to_tokens(ids: list[int]) -> list[str]:
    return [ID_TO_TOKEN[i] for i in ids if i in ID_TO_TOKEN]


def tokens_to_ids(tokens: list[str]) -> list[int]:
    return [TOKEN_TO_ID[t] for t in tokens]


def speed_bin_index(speed_mps: float) -> int:
    return SPEED_TOKENS.index(speed_token(speed_mps))


def speed_bin_center_mps(index: int) -> float:
    return float(SPEED_BINS_MPS[int(index)])
