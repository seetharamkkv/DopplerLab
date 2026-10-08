"""Greedy CTC decode and token error rate."""

from __future__ import annotations

from position_ctc.alphabet import BLANK_ID, ID_TO_TOKEN


def greedy_decode(logits) -> list[str]:
    """``logits`` is ``(T, C)`` on any device."""
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    out: list[str] = []
    prev = None
    for idx in ids:
        if idx == BLANK_ID:
            prev = idx
            continue
        if idx != prev:
            out.append(ID_TO_TOKEN[int(idx)])
        prev = idx
    return out


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    row = list(range(m + 1))
    for i in range(1, n + 1):
        prev, row[0] = row[0], i
        for j in range(1, m + 1):
            cur = row[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            row[j] = min(row[j] + 1, row[j - 1] + 1, prev + cost)
            prev = cur
    return row[m]


def token_error_rate(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def initial_pair(tokens: list[str]) -> tuple[str | None, str | None]:
    vx = next((tok for tok in tokens if tok.startswith("VX")), None)
    vy = next((tok for tok in tokens if tok.startswith("VY")), None)
    return vx, vy
