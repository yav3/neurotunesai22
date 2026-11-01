"""Derivative work analysis for music copyright.

Quantifies the degree to which one work derives from another by
measuring structural, melodic, harmonic, and rhythmic overlap.
Uses interval sequences and pitch-class profiles rather than
raw audio (symbolic comparison).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Sequence, Tuple

import numpy as np


class DerivativeCategory(Enum):
    IDENTICAL = "identical"
    SUBSTANTIALLY_SIMILAR = "substantially_similar"
    DERIVATIVE = "derivative"
    INSPIRED_BY = "inspired_by"
    INDEPENDENT = "independent"


@dataclass
class OverlapResult:
    melodic_overlap: float
    harmonic_overlap: float
    rhythmic_overlap: float
    structural_overlap: float
    composite_score: float
    category: DerivativeCategory


def pitch_class_profile(pitches: Sequence[int]) -> np.ndarray:
    hist = np.zeros(12)
    for p in pitches:
        hist[p % 12] += 1
    total = hist.sum()
    return hist / total if total > 0 else hist


def interval_sequence(pitches: Sequence[int]) -> List[int]:
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


def longest_common_subsequence(a: Sequence[int], b: Sequence[int]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def melodic_overlap(pitches_a: Sequence[int], pitches_b: Sequence[int]) -> float:
    if len(pitches_a) < 2 or len(pitches_b) < 2:
        return 0.0
    int_a = interval_sequence(pitches_a)
    int_b = interval_sequence(pitches_b)
    lcs = longest_common_subsequence(int_a, int_b)
    return lcs / max(len(int_a), len(int_b))


def harmonic_overlap(
    pc_profile_a: np.ndarray,
    pc_profile_b: np.ndarray,
) -> float:
    norm_a = np.linalg.norm(pc_profile_a)
    norm_b = np.linalg.norm(pc_profile_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(pc_profile_a, pc_profile_b) / (norm_a * norm_b))


def rhythmic_overlap(
    onsets_a: Sequence[float],
    onsets_b: Sequence[float],
    tolerance: float = 0.05,
) -> float:
    if len(onsets_a) == 0 or len(onsets_b) == 0:
        return 0.0
    ioi_a = [onsets_a[i + 1] - onsets_a[i] for i in range(len(onsets_a) - 1)]
    ioi_b = [onsets_b[i + 1] - onsets_b[i] for i in range(len(onsets_b) - 1)]
    if len(ioi_a) == 0 or len(ioi_b) == 0:
        return 0.0
    ratio_a = np.array(ioi_a) / (sum(ioi_a) / len(ioi_a)) if len(ioi_a) > 0 else np.array([])
    ratio_b = np.array(ioi_b) / (sum(ioi_b) / len(ioi_b)) if len(ioi_b) > 0 else np.array([])
    lcs = longest_common_subsequence(
        [round(r, 2) for r in ratio_a],
        [round(r, 2) for r in ratio_b],
    )
    return lcs / max(len(ratio_a), len(ratio_b))


def structural_overlap(
    form_a: Sequence[str],
    form_b: Sequence[str],
) -> float:
    if len(form_a) == 0 or len(form_b) == 0:
        return 0.0
    lcs = longest_common_subsequence(list(form_a), list(form_b))
    return lcs / max(len(form_a), len(form_b))


def classify_derivative(score: float) -> DerivativeCategory:
    if score >= 0.95:
        return DerivativeCategory.IDENTICAL
    if score >= 0.75:
        return DerivativeCategory.SUBSTANTIALLY_SIMILAR
    if score >= 0.50:
        return DerivativeCategory.DERIVATIVE
    if score >= 0.25:
        return DerivativeCategory.INSPIRED_BY
    return DerivativeCategory.INDEPENDENT


def analyze_derivative(
    pitches_a: Sequence[int],
    pitches_b: Sequence[int],
    onsets_a: Sequence[float] | None = None,
    onsets_b: Sequence[float] | None = None,
    form_a: Sequence[str] | None = None,
    form_b: Sequence[str] | None = None,
    weights: Tuple[float, float, float, float] = (0.40, 0.25, 0.20, 0.15),
) -> OverlapResult:
    mel = melodic_overlap(pitches_a, pitches_b)
    har = harmonic_overlap(
        pitch_class_profile(pitches_a),
        pitch_class_profile(pitches_b),
    )
    rhy = rhythmic_overlap(onsets_a or [], onsets_b or [])
    stru = structural_overlap(form_a or [], form_b or [])
    w = np.array(weights)
    w = w / w.sum()
    composite = float(w[0] * mel + w[1] * har + w[2] * rhy + w[3] * stru)
    return OverlapResult(
        melodic_overlap=mel,
        harmonic_overlap=har,
        rhythmic_overlap=rhy,
        structural_overlap=stru,
        composite_score=composite,
        category=classify_derivative(composite),
    )


def transposition_invariant_overlap(
    pitches_a: Sequence[int],
    pitches_b: Sequence[int],
) -> Tuple[float, int]:
    best_score = 0.0
    best_shift = 0
    for shift in range(12):
        shifted = [p + shift for p in pitches_b]
        score = melodic_overlap(pitches_a, shifted)
        if score > best_score:
            best_score = score
            best_shift = shift
    return best_score, best_shift


def self_quotation_ratio(
    excerpts: List[Sequence[int]],
    full_work: Sequence[int],
    threshold: float = 0.6,
) -> float:
    if len(excerpts) == 0:
        return 0.0
    count = sum(
        1 for ex in excerpts
        if melodic_overlap(list(ex), list(full_work)) >= threshold
    )
    return count / len(excerpts)
