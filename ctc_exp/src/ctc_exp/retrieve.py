"""Phase-2 string retrieval."""

from __future__ import annotations

from ctc_exp.alphabet import speed_token
from ctc_exp.tokens import parse_min_range_token, parse_speed_token, token_error_rate


def recall_at_k(
    queries: list[dict],
    *,
    k: int,
    key: str,
) -> float:
    """Each query dict: gallery_keys, query_key, exclude self already."""
    if not queries:
        return float("nan")
    hits = 0
    for q in queries:
        qk = q[key]
        ranked = q["ranked_ids"][:k]
        gallery = q["gallery"]
        if any(gallery[i][key] == qk for i in ranked):
            hits += 1
    return hits / len(queries)


def rank_by_token_distance(query_tokens: list[str], gallery: list[list[str]]) -> list[int]:
    scores = [token_error_rate(query_tokens, g) for g in gallery]
    return sorted(range(len(gallery)), key=lambda i: scores[i])


def random_recall_at_k(n_gallery: int, k: int) -> float:
    if n_gallery <= 0:
        return float("nan")
    return min(1.0, k / n_gallery)


def clip_bin_key(speed_mps: float, range_token: str | None) -> tuple[str, str | None]:
    return speed_token(speed_mps), range_token


def decoded_speed_key(tokens: list[str], fallback_speed_mps: float) -> str:
    parsed = parse_speed_token(tokens)
    return parsed if parsed is not None else speed_token(fallback_speed_mps)


def decoded_range_key(tokens: list[str]) -> str | None:
    return parse_min_range_token(tokens)
