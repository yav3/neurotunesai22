"""Harmonic language evolution: Markov models of chord progressions per era."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .periods import MusicalPeriod


ROMAN_NUMERALS = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]

ERA_TRANSITION_MATRICES: dict[MusicalPeriod, np.ndarray] = {}


def _build_common_practice() -> np.ndarray:
    n = len(ROMAN_NUMERALS)
    T = np.zeros((n, n))
    idx = {r: i for i, r in enumerate(ROMAN_NUMERALS)}
    progressions = [
        ("I", "IV", 0.20), ("I", "V", 0.25), ("I", "vi", 0.10), ("I", "ii", 0.10),
        ("ii", "V", 0.50), ("ii", "vii°", 0.10), ("ii", "IV", 0.15),
        ("iii", "vi", 0.30), ("iii", "IV", 0.25),
        ("IV", "V", 0.40), ("IV", "I", 0.20), ("IV", "ii", 0.10),
        ("V", "I", 0.60), ("V", "vi", 0.15), ("V", "IV", 0.05),
        ("vi", "ii", 0.30), ("vi", "IV", 0.25), ("vi", "V", 0.15),
        ("vii°", "I", 0.55), ("vii°", "iii", 0.15),
    ]
    for src, dst, w in progressions:
        T[idx[src], idx[dst]] = w
    for i in range(n):
        row_sum = T[i].sum()
        if row_sum > 0:
            deficit = 1.0 - row_sum
            if deficit > 0:
                zeros = T[i] == 0
                n_zeros = zeros.sum()
                if n_zeros > 0:
                    T[i, zeros] = deficit / n_zeros
            T[i] /= T[i].sum()
        else:
            T[i] = 1.0 / n
    return T


ERA_TRANSITION_MATRICES[MusicalPeriod.CLASSICAL] = _build_common_practice()
ERA_TRANSITION_MATRICES[MusicalPeriod.BAROQUE] = _build_common_practice() * 0.85 + 0.15 / len(ROMAN_NUMERALS)
for p in [MusicalPeriod.BAROQUE]:
    ERA_TRANSITION_MATRICES[p] /= ERA_TRANSITION_MATRICES[p].sum(axis=1, keepdims=True)

_romantic = _build_common_practice() * 0.7 + 0.3 / len(ROMAN_NUMERALS)
_romantic /= _romantic.sum(axis=1, keepdims=True)
ERA_TRANSITION_MATRICES[MusicalPeriod.ROMANTIC] = _romantic

_modern = np.ones((len(ROMAN_NUMERALS), len(ROMAN_NUMERALS))) * 0.6 / len(ROMAN_NUMERALS)
_modern += _build_common_practice() * 0.4
_modern /= _modern.sum(axis=1, keepdims=True)
ERA_TRANSITION_MATRICES[MusicalPeriod.MODERN] = _modern


@dataclass
class HarmonicProfile:
    period: MusicalPeriod
    entropy: float
    vocabulary_size: int
    dominant_resolution_rate: float
    transition_matrix: np.ndarray


@dataclass
class ProgressionAnalysis:
    chords: list[str]
    log_likelihood: float
    perplexity: float
    surprise_curve: np.ndarray
    most_surprising_idx: int


def harmonic_entropy(transition_matrix: np.ndarray) -> float:
    n = transition_matrix.shape[0]
    stationary = _stationary_distribution(transition_matrix)
    H = 0.0
    for i in range(n):
        for j in range(n):
            p = transition_matrix[i, j]
            if p > 1e-15:
                H -= stationary[i] * p * math.log2(p)
    return H


def _stationary_distribution(T: np.ndarray, max_iter: int = 200) -> np.ndarray:
    n = T.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_new = pi @ T
        if np.allclose(pi, pi_new, atol=1e-12):
            break
        pi = pi_new
    return pi


def vocabulary_usage(transition_matrix: np.ndarray, threshold: float = 0.01) -> int:
    pi = _stationary_distribution(transition_matrix)
    return int(np.sum(pi > threshold))


def dominant_resolution_rate(transition_matrix: np.ndarray) -> float:
    idx = {r: i for i, r in enumerate(ROMAN_NUMERALS)}
    if "V" in idx and "I" in idx:
        return float(transition_matrix[idx["V"], idx["I"]])
    return 0.0


def analyze_progression(
    chords: list[str],
    period: MusicalPeriod = MusicalPeriod.CLASSICAL,
) -> ProgressionAnalysis:
    T = ERA_TRANSITION_MATRICES.get(period)
    if T is None:
        T = _build_common_practice()
    idx = {r: i for i, r in enumerate(ROMAN_NUMERALS)}
    valid = [c for c in chords if c in idx]
    if len(valid) < 2:
        return ProgressionAnalysis(
            chords=valid, log_likelihood=0.0, perplexity=1.0,
            surprise_curve=np.array([0.0]), most_surprising_idx=0,
        )
    ll = 0.0
    surprises = []
    for k in range(1, len(valid)):
        i, j = idx[valid[k - 1]], idx[valid[k]]
        p = max(T[i, j], 1e-15)
        ll += math.log2(p)
        surprises.append(-math.log2(p))
    surprise_arr = np.array(surprises)
    avg_surprise = surprise_arr.mean() if len(surprises) > 0 else 0.0
    perplexity = 2.0 ** avg_surprise
    return ProgressionAnalysis(
        chords=valid,
        log_likelihood=ll,
        perplexity=perplexity,
        surprise_curve=surprise_arr,
        most_surprising_idx=int(np.argmax(surprise_arr)) if len(surprises) > 0 else 0,
    )


def profile_era(period: MusicalPeriod) -> HarmonicProfile:
    T = ERA_TRANSITION_MATRICES.get(period)
    if T is None:
        T = _build_common_practice()
    return HarmonicProfile(
        period=period,
        entropy=harmonic_entropy(T),
        vocabulary_size=vocabulary_usage(T),
        dominant_resolution_rate=dominant_resolution_rate(T),
        transition_matrix=T.copy(),
    )


def era_entropy_trajectory(
    periods: list[MusicalPeriod] | None = None,
) -> list[tuple[MusicalPeriod, float]]:
    if periods is None:
        periods = [
            MusicalPeriod.BAROQUE, MusicalPeriod.CLASSICAL,
            MusicalPeriod.ROMANTIC, MusicalPeriod.MODERN,
        ]
    return [(p, profile_era(p).entropy) for p in periods]


def generate_progression(
    period: MusicalPeriod = MusicalPeriod.CLASSICAL,
    length: int = 8,
    seed: int | None = None,
) -> list[str]:
    T = ERA_TRANSITION_MATRICES.get(period, _build_common_practice())
    rng = np.random.default_rng(seed)
    state = 0
    result = [ROMAN_NUMERALS[state]]
    for _ in range(length - 1):
        state = rng.choice(len(ROMAN_NUMERALS), p=T[state])
        result.append(ROMAN_NUMERALS[state])
    return result


def transition_divergence(
    period_a: MusicalPeriod,
    period_b: MusicalPeriod,
) -> float:
    Ta = ERA_TRANSITION_MATRICES.get(period_a, _build_common_practice())
    Tb = ERA_TRANSITION_MATRICES.get(period_b, _build_common_practice())
    n = Ta.shape[0]
    kl = 0.0
    for i in range(n):
        for j in range(n):
            p = Ta[i, j]
            q = max(Tb[i, j], 1e-15)
            if p > 1e-15:
                kl += p * math.log2(p / q)
    return kl / n
