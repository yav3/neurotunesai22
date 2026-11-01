"""Consonance/dissonance modeling: periodicity, harmonicity, familiarity, cultural weighting."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class ConsonanceModel(str, Enum):
    HELMHOLTZ = "helmholtz"
    PLOMP_LEVELT = "plomp_levelt"
    HARMONICITY = "harmonicity"
    PERIODICITY = "periodicity"
    COMPOSITE = "composite"


@dataclass
class IntervalRating:
    interval_cents: float
    consonance: float
    model_used: ConsonanceModel
    components: Dict[str, float] = field(default_factory=dict)


def frequency_ratio_simplicity(p: int, q: int) -> float:
    if p <= 0 or q <= 0:
        return 0.0
    return 1.0 / (p + q)


def harmonicity_score(f1: float, f2: float, n_harmonics: int = 8,
                      tolerance_cents: float = 15.0) -> float:
    if f1 <= 0 or f2 <= 0:
        return 0.0
    tol_ratio = 2.0 ** (tolerance_cents / 1200.0)
    matches = 0
    total = 0
    for i in range(1, n_harmonics + 1):
        for j in range(1, n_harmonics + 1):
            total += 1
            ratio = (f1 * i) / (f2 * j)
            if 1.0 / tol_ratio <= ratio <= tol_ratio:
                matches += 1
    return matches / total if total > 0 else 0.0


def periodicity_pitch_strength(f1: float, f2: float, sr: float = 44100.0,
                               dur: float = 0.1) -> float:
    if f1 <= 0 or f2 <= 0:
        return 0.0
    n = int(sr * dur)
    t = np.arange(n) / sr
    sig = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    acf = np.correlate(sig, sig, mode="full")
    acf = acf[n - 1:]
    acf = acf / (acf[0] + 1e-12)
    min_lag = int(sr / max(f1, f2, 1.0))
    max_lag = min(int(sr / 20.0), len(acf) - 1)
    if min_lag >= max_lag or min_lag >= len(acf):
        return 0.0
    peak = float(np.max(acf[min_lag:max_lag + 1]))
    return max(0.0, peak)


def plomp_levelt_consonance(f1: float, f2: float) -> float:
    if f1 <= 0 or f2 <= 0:
        return 1.0
    f_low, f_high = min(f1, f2), max(f1, f2)
    s = 0.24 / (0.0207 * f_low + 18.96)
    x = s * (f_high - f_low)
    d = math.exp(-3.5 * x) - math.exp(-5.75 * x)
    return max(0.0, 1.0 - d)


def composite_consonance(f1: float, f2: float,
                         weights: Optional[Dict[str, float]] = None) -> IntervalRating:
    if weights is None:
        weights = {"harmonicity": 0.35, "periodicity": 0.3,
                   "plomp_levelt": 0.25, "simplicity": 0.1}
    h = harmonicity_score(f1, f2)
    p = periodicity_pitch_strength(f1, f2)
    pl = plomp_levelt_consonance(f1, f2)
    ratio = max(f1, f2) / min(f1, f2) if min(f1, f2) > 0 else 1.0
    best_simp = 0.0
    for num in range(1, 13):
        for den in range(1, 13):
            r = num / den
            if abs(r - ratio) < 0.02:
                s = frequency_ratio_simplicity(num, den)
                best_simp = max(best_simp, s)
    norm_simp = best_simp * 6.0
    components = {"harmonicity": h, "periodicity": p,
                  "plomp_levelt": pl, "simplicity": norm_simp}
    total = sum(weights.get(k, 0.0) * v for k, v in components.items())
    total = max(0.0, min(1.0, total))
    cents = 1200.0 * math.log2(ratio) if ratio > 0 else 0.0
    return IntervalRating(
        interval_cents=cents, consonance=total,
        model_used=ConsonanceModel.COMPOSITE, components=components,
    )


JUST_INTERVALS: Dict[str, Tuple[int, int]] = {
    "unison": (1, 1), "minor_second": (16, 15), "major_second": (9, 8),
    "minor_third": (6, 5), "major_third": (5, 4), "perfect_fourth": (4, 3),
    "tritone": (45, 32), "perfect_fifth": (3, 2), "minor_sixth": (8, 5),
    "major_sixth": (5, 3), "minor_seventh": (9, 5), "major_seventh": (15, 8),
    "octave": (2, 1),
}


def just_interval_consonance_ranking() -> List[Tuple[str, float]]:
    ranking = []
    for name, (p, q) in JUST_INTERVALS.items():
        s = frequency_ratio_simplicity(p, q)
        ranking.append((name, s))
    ranking.sort(key=lambda x: -x[1])
    return ranking


@dataclass
class ChordConsonance:
    frequencies: List[float]
    total_consonance: float
    pairwise_scores: List[Tuple[int, int, float]]
    roughness: float


def chord_consonance(freqs: List[float]) -> ChordConsonance:
    if len(freqs) < 2:
        return ChordConsonance(freqs, 1.0, [], 0.0)
    pairs = []
    total = 0.0
    count = 0
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            c = plomp_levelt_consonance(freqs[i], freqs[j])
            pairs.append((i, j, c))
            total += c
            count += 1
    avg = total / count if count > 0 else 1.0
    roughness = 1.0 - avg
    return ChordConsonance(freqs, avg, pairs, roughness)


def cultural_weighting(base_consonance: float, familiarity: float = 0.5,
                       context_weight: float = 0.3) -> float:
    return (1.0 - context_weight) * base_consonance + context_weight * familiarity


@dataclass
class TensionProfile:
    times: np.ndarray
    tension: np.ndarray
    peak_tension: float
    mean_tension: float
    resolution_points: List[int]


def tension_from_dissonance_curve(freq_pairs_over_time: List[List[Tuple[float, float]]],
                                   decay: float = 0.9) -> TensionProfile:
    n = len(freq_pairs_over_time)
    if n == 0:
        return TensionProfile(np.array([]), np.array([]), 0.0, 0.0, [])
    tensions = np.zeros(n)
    for t, pairs in enumerate(freq_pairs_over_time):
        d = 0.0
        for f1, f2 in pairs:
            d += 1.0 - plomp_levelt_consonance(f1, f2)
        tensions[t] = d / max(len(pairs), 1)
    smoothed = np.zeros(n)
    smoothed[0] = tensions[0]
    for t in range(1, n):
        smoothed[t] = decay * smoothed[t - 1] + (1.0 - decay) * tensions[t]
    resolutions = []
    for t in range(1, n - 1):
        if smoothed[t - 1] > smoothed[t] < smoothed[t + 1] and smoothed[t] < 0.2:
            resolutions.append(t)
    times = np.arange(n, dtype=float)
    return TensionProfile(
        times=times, tension=smoothed,
        peak_tension=float(np.max(smoothed)),
        mean_tension=float(np.mean(smoothed)),
        resolution_points=resolutions,
    )


def interval_class_vector(pitch_classes: List[int]) -> List[int]:
    pcs = sorted(set(p % 12 for p in pitch_classes))
    icv = [0] * 6
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            diff = (pcs[j] - pcs[i]) % 12
            ic = min(diff, 12 - diff)
            if 1 <= ic <= 6:
                icv[ic - 1] += 1
    return icv


def set_dissonance(pitch_classes: List[int], base_freq: float = 261.63) -> float:
    if len(pitch_classes) < 2:
        return 0.0
    freqs = [base_freq * (2.0 ** (pc / 12.0)) for pc in pitch_classes]
    result = chord_consonance(freqs)
    return result.roughness
