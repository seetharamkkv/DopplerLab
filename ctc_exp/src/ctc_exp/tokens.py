"""Phase-0 token strings from kinematics (no network)."""

from __future__ import annotations

import numpy as np

from ctc_exp.alphabet import range_token, speed_token

CPA_HALF_S = 0.4
SIL_BEYOND_S = 3.5
# ~186 ms at 22050 Hz / hop 512. Keeps CTC targets much shorter than T/4.
DOWNSAMPLE = 8


def collapse_repeats(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if not out or out[-1] != tok:
            out.append(tok)
    return out


def phase_at(t_s: float, cpa_s: float) -> str:
    dt = float(t_s) - float(cpa_s)
    if abs(dt) > SIL_BEYOND_S:
        return "SIL"
    if abs(dt) <= CPA_HALF_S:
        return "CPA"
    return "APR" if dt < 0 else "REC"


def token_string(
    *,
    speed_mps: float,
    cpa_s: float,
    times_s: np.ndarray,
    range_m: np.ndarray | None = None,
    include_range: bool = True,
) -> list[str]:
    """Compact CTC target: one speed token, then collapsed phase (/ range) runs.

    Do not emit ``phase, V, R`` on every hop — that cycle never collapses and
    is longer than the encoder timeline, which makes CTCLoss return 0.
    """
    times = np.asarray(times_s, dtype=np.float64)
    if times.size == 0:
        raise ValueError("empty time axis")
    idx = np.arange(0, len(times), DOWNSAMPLE)
    rng = None if range_m is None else np.asarray(range_m, dtype=np.float64)
    events: list[tuple[str, str | None]] = []
    for i in idx:
        ph = phase_at(float(times[i]), cpa_s)
        rtok = None
        if include_range and rng is not None and rng.size:
            j = min(int(i), len(rng) - 1)
            rtok = range_token(float(rng[j]))
        events.append((ph, rtok))
    collapsed: list[tuple[str, str | None]] = []
    for ev in events:
        if not collapsed or collapsed[-1] != ev:
            collapsed.append(ev)
    seq = [speed_token(speed_mps)]
    for ph, rtok in collapsed:
        seq.append(ph)
        if rtok is not None:
            seq.append(rtok)
    return collapse_repeats(seq)


def parse_speed_token(tokens: list[str]) -> str | None:
    vs = [t for t in tokens if t.startswith("V") and t[1:].isdigit()]
    if not vs:
        return None
    return max(set(vs), key=vs.count)


def parse_min_range_token(tokens: list[str]) -> str | None:
    rs = [t for t in tokens if t.startswith("R") and t[1:].isdigit()]
    if not rs:
        return None
    return min(rs, key=lambda t: int(t[1:]))


def token_error_rate(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[n, m]) / float(n)


def has_apr_cpa_rec(tokens: list[str]) -> bool:
    try:
        i = tokens.index("APR")
        j = tokens.index("CPA")
        k = tokens.index("REC")
    except ValueError:
        return False
    return i < j < k
