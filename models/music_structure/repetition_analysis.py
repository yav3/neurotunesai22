"""Repetition analysis: detect and quantify repeated musical patterns.

Identify exact and approximate repetitions at multiple levels (motif,
phrase, section) via suffix-array methods, self-similarity lag matrices,
and compression-based complexity measures.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RepetitionSpan:
    start_a: int
    start_b: int
    length: int
    similarity: float


def lag_similarity_matrix(features: np.ndarray,
                          max_lag: int = 0) -> np.ndarray:
    n = features.shape[1]
    if max_lag <= 0:
        max_lag = n
    lags = min(max_lag, n)
    sim = np.zeros((n, lags))
    for lag in range(lags):
        for i in range(n - lag):
            d = np.linalg.norm(features[:, i] - features[:, i + lag])
            sim[i, lag] = np.exp(-d ** 2)
    return sim


def diagonal_strength(ssm: np.ndarray, min_length: int = 8) -> np.ndarray:
    n = ssm.shape[0]
    strengths = np.zeros(n)
    for offset in range(1, n):
        diag = np.diag(ssm, offset)
        run = 0
        for v in diag:
            if v > 0.5:
                run += 1
            else:
                if run >= min_length:
                    strengths[offset] += run
                run = 0
        if run >= min_length:
            strengths[offset] += run
    return strengths


def find_repetitions(features: np.ndarray, threshold: float = 0.8,
                     min_length: int = 4) -> List[RepetitionSpan]:
    n = features.shape[1]
    reps = []
    for i in range(n):
        for j in range(i + min_length, n):
            length = 0
            sim_sum = 0.0
            while i + length < n and j + length < n:
                d = np.linalg.norm(features[:, i + length] - features[:, j + length])
                s = np.exp(-d ** 2)
                if s < threshold:
                    break
                sim_sum += s
                length += 1
            if length >= min_length:
                reps.append(RepetitionSpan(
                    start_a=i, start_b=j, length=length,
                    similarity=sim_sum / length,
                ))
    return reps


def repetition_density(reps: List[RepetitionSpan],
                       total_frames: int) -> float:
    covered = set()
    for r in reps:
        for k in range(r.length):
            covered.add(r.start_a + k)
            covered.add(r.start_b + k)
    return len(covered) / max(total_frames, 1)


def compression_ratio(sequence: np.ndarray, codebook_size: int = 16) -> float:
    from numpy import unique
    bins = np.linspace(sequence.min(), sequence.max(), codebook_size + 1)
    quantised = np.digitize(sequence, bins)
    symbols = quantised.tobytes()
    unique_bigrams = len(set(zip(symbols[:-1], symbols[1:])))
    max_bigrams = codebook_size ** 2
    return unique_bigrams / max(max_bigrams, 1)


def novelty_from_repetition(ssm: np.ndarray) -> np.ndarray:
    n = ssm.shape[0]
    novelty = np.zeros(n)
    for i in range(1, n):
        novelty[i] = 1.0 - np.max(ssm[i, :i])
    return novelty


def repetition_hierarchy(reps: List[RepetitionSpan]) -> Dict[str, List[RepetitionSpan]]:
    short = [r for r in reps if r.length < 8]
    medium = [r for r in reps if 8 <= r.length < 32]
    long = [r for r in reps if r.length >= 32]
    return {"motif": short, "phrase": medium, "section": long}
