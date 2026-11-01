"""Melody comparison for copyright analysis: pitch, rhythm, contour similarity."""
from __future__ import annotations
import numpy as np


def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[m, n])


def pitch_sequence_similarity(seq1: list[int], seq2: list[int]) -> float:
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0
    pc1 = [p % 12 for p in seq1]
    pc2 = [p % 12 for p in seq2]
    dist = _edit_distance(pc1, pc2)
    max_len = max(len(pc1), len(pc2))
    return 1.0 - dist / max_len


def rhythm_sequence_similarity(seq1: list[float], seq2: list[float]) -> float:
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0
    def quantize(d: float) -> int:
        return round(d * 4)
    q1 = [quantize(d) for d in seq1]
    q2 = [quantize(d) for d in seq2]
    dist = _edit_distance(q1, q2)
    max_len = max(len(q1), len(q2))
    return 1.0 - dist / max_len


def _to_contour(pitches: list[int]) -> list[str]:
    contour = []
    for i in range(1, len(pitches)):
        diff = pitches[i] - pitches[i - 1]
        if diff > 0:
            contour.append("U")
        elif diff < 0:
            contour.append("D")
        else:
            contour.append("R")
    return contour


def interval_contour_similarity(seq1: list[int], seq2: list[int]) -> float:
    if len(seq1) < 2 and len(seq2) < 2:
        return 1.0
    if len(seq1) < 2 or len(seq2) < 2:
        return 0.0
    c1 = _to_contour(seq1)
    c2 = _to_contour(seq2)
    dist = _edit_distance(c1, c2)
    max_len = max(len(c1), len(c2))
    return 1.0 - dist / max_len


def transposition_invariant_similarity(seq1: list[int], seq2: list[int]) -> float:
    if len(seq1) < 2 and len(seq2) < 2:
        return 1.0
    if len(seq1) < 2 or len(seq2) < 2:
        return 0.0
    iv1 = [seq1[i + 1] - seq1[i] for i in range(len(seq1) - 1)]
    iv2 = [seq2[i + 1] - seq2[i] for i in range(len(seq2) - 1)]
    dist = _edit_distance(iv1, iv2)
    max_len = max(len(iv1), len(iv2))
    return 1.0 - dist / max_len
