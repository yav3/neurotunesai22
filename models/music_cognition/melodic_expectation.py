"""Melodic expectation: Narmour implication-realization, information-theoretic surprise."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "NarmourAnalysis", "ExpectationProfile",
    "narmour_implication", "registral_direction_score",
    "intervallic_difference_score", "registral_return_score",
    "proximity_score", "closure_score",
    "markov_pitch_expectation", "information_content",
    "expectation_violation", "melodic_surprise_curve",
    "build_pitch_transition_matrix",
]


@dataclass
class NarmourAnalysis:
    registral_direction: float
    intervallic_difference: float
    registral_return: float
    proximity: float
    closure: float
    combined: float


def registral_direction_score(interval1: float, interval2: float) -> float:
    if abs(interval1) <= 5:
        expected_same_dir = True
    else:
        expected_same_dir = False
    actual_same = (interval1 > 0 and interval2 > 0) or (interval1 < 0 and interval2 < 0)
    if expected_same_dir:
        return 1.0 if actual_same else 0.0
    else:
        return 1.0 if not actual_same else 0.0


def intervallic_difference_score(interval1: float, interval2: float) -> float:
    if abs(interval1) <= 5:
        expected = abs(interval1)
    else:
        expected = max(0, abs(interval1) - 3)
    actual = abs(interval2)
    diff = abs(actual - expected)
    return float(np.exp(-0.3 * diff))


def registral_return_score(
    pitch1: float, pitch2: float, pitch3: float,
) -> float:
    if abs(pitch2 - pitch1) > 5:
        target = pitch1
        deviation = abs(pitch3 - target)
        return float(np.exp(-0.2 * deviation))
    return 0.5


def proximity_score(interval: float, threshold: float = 5.0) -> float:
    return float(np.exp(-abs(interval) / threshold))


def closure_score(
    interval1: float, interval2: float,
    pitch2: float, pitch3: float,
) -> float:
    direction_change = (interval1 > 0) != (interval2 > 0)
    smaller_interval = abs(interval2) < abs(interval1)
    if direction_change and smaller_interval:
        return 1.0
    elif direction_change or smaller_interval:
        return 0.5
    return 0.0


def narmour_implication(pitches: np.ndarray) -> List[NarmourAnalysis]:
    p = np.asarray(pitches, dtype=float)
    if len(p) < 3:
        return []
    results = []
    for i in range(len(p) - 2):
        i1 = p[i + 1] - p[i]
        i2 = p[i + 2] - p[i + 1]
        rd = registral_direction_score(i1, i2)
        id_s = intervallic_difference_score(i1, i2)
        rr = registral_return_score(p[i], p[i + 1], p[i + 2])
        pr = proximity_score(i2)
        cl = closure_score(i1, i2, p[i + 1], p[i + 2])
        combined = 0.25 * rd + 0.25 * id_s + 0.2 * rr + 0.15 * pr + 0.15 * cl
        results.append(NarmourAnalysis(
            registral_direction=rd, intervallic_difference=id_s,
            registral_return=rr, proximity=pr, closure=cl,
            combined=combined,
        ))
    return results


def build_pitch_transition_matrix(
    melodies: List[np.ndarray],
    n_pitch_classes: int = 12,
) -> np.ndarray:
    trans = np.zeros((n_pitch_classes, n_pitch_classes))
    for melody in melodies:
        pcs = np.asarray(melody, dtype=int) % n_pitch_classes
        for i in range(len(pcs) - 1):
            trans[pcs[i], pcs[i + 1]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return trans / row_sums


def markov_pitch_expectation(
    context_pitch: int,
    transition_matrix: np.ndarray,
    n_pitch_classes: int = 12,
) -> np.ndarray:
    pc = context_pitch % n_pitch_classes
    return transition_matrix[pc].copy()


def information_content(
    pitch: int,
    context_pitch: int,
    transition_matrix: np.ndarray,
    n_pitch_classes: int = 12,
) -> float:
    pc = pitch % n_pitch_classes
    probs = markov_pitch_expectation(context_pitch, transition_matrix, n_pitch_classes)
    p = probs[pc]
    if p < 1e-15:
        return 20.0
    return float(-np.log2(p))


@dataclass
class ExpectationProfile:
    pitches: np.ndarray
    narmour_scores: np.ndarray
    ic_values: np.ndarray
    surprise_curve: np.ndarray
    mean_surprise: float
    max_surprise_idx: int


def expectation_violation(
    pitches: np.ndarray,
    transition_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    p = np.asarray(pitches, dtype=float)
    n = len(p)
    if n < 3:
        return np.zeros(max(n - 2, 0))
    narmour = narmour_implication(p)
    violations = np.array([1.0 - a.combined for a in narmour])
    return violations


def melodic_surprise_curve(
    pitches: np.ndarray,
    transition_matrix: Optional[np.ndarray] = None,
    n_pitch_classes: int = 12,
) -> ExpectationProfile:
    p = np.asarray(pitches, dtype=float)
    n = len(p)
    narmour = narmour_implication(p)
    narmour_scores = np.array([a.combined for a in narmour]) if narmour else np.array([])
    if transition_matrix is None:
        transition_matrix = np.ones((n_pitch_classes, n_pitch_classes)) / n_pitch_classes
    ic_vals = np.zeros(max(n - 1, 0))
    for i in range(1, n):
        ic_vals[i - 1] = information_content(
            int(p[i]), int(p[i - 1]), transition_matrix, n_pitch_classes
        )
    min_len = min(len(narmour_scores), len(ic_vals) - 1) if len(ic_vals) > 1 else 0
    if min_len > 0:
        surprise = 0.5 * (1.0 - narmour_scores[:min_len]) + 0.5 * (ic_vals[1:min_len + 1] / 20.0)
    else:
        surprise = ic_vals / 20.0 if len(ic_vals) > 0 else np.array([])
    mean_s = float(surprise.mean()) if len(surprise) > 0 else 0.0
    max_idx = int(np.argmax(surprise)) if len(surprise) > 0 else 0
    return ExpectationProfile(
        pitches=p, narmour_scores=narmour_scores,
        ic_values=ic_vals, surprise_curve=surprise,
        mean_surprise=mean_s, max_surprise_idx=max_idx,
    )
