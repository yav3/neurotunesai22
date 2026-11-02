"""Cross-cultural music analysis: scale comparison, rhythm alignment, tuning deviation, tradition mapping."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class Tradition(str, Enum):
    WESTERN = "western"
    ARABIC = "arabic"
    INDIAN = "indian"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    AFRICAN = "african"
    INDONESIAN = "indonesian"
    TURKISH = "turkish"
    PERSIAN = "persian"
    ETHIOPIAN = "ethiopian"
    CELTIC = "celtic"
    FLAMENCO = "flamenco"
    LATIN = "latin"


@dataclass
class ScaleProfile:
    name: str
    tradition: Tradition
    intervals_cents: List[float]
    tonic_hz: float = 440.0

    @property
    def n_notes(self) -> int:
        return len(self.intervals_cents) + 1

    def frequencies(self) -> List[float]:
        freqs = [self.tonic_hz]
        for c in self.intervals_cents:
            freqs.append(self.tonic_hz * (2.0 ** (c / 1200.0)))
        return freqs

    def total_span_cents(self) -> float:
        return self.intervals_cents[-1] if self.intervals_cents else 0.0


@dataclass
class RhythmProfile:
    name: str
    tradition: Tradition
    cycle_beats: int
    accent_pattern: List[float]
    subdivisions: int = 1

    @property
    def total_pulses(self) -> int:
        return self.cycle_beats * self.subdivisions


def scale_interval_similarity(a: ScaleProfile, b: ScaleProfile) -> float:
    ints_a = sorted(a.intervals_cents)
    ints_b = sorted(b.intervals_cents)
    if not ints_a and not ints_b:
        return 1.0
    if not ints_a or not ints_b:
        return 0.0
    n = max(len(ints_a), len(ints_b))
    cost = 0.0
    i, j = 0, 0
    matched = 0
    tolerance = 30.0
    while i < len(ints_a) and j < len(ints_b):
        diff = abs(ints_a[i] - ints_b[j])
        if diff <= tolerance:
            cost += diff
            matched += 1
            i += 1
            j += 1
        elif ints_a[i] < ints_b[j]:
            cost += tolerance
            i += 1
        else:
            cost += tolerance
            j += 1
    cost += tolerance * (len(ints_a) - i + len(ints_b) - j)
    max_cost = tolerance * n
    return max(0.0, 1.0 - cost / max_cost) if max_cost > 0 else 1.0


def tuning_deviation_12tet(intervals_cents: List[float]) -> List[float]:
    tet_steps = [100 * i for i in range(1, 13)]
    deviations = []
    for c in intervals_cents:
        nearest = min(tet_steps, key=lambda s: abs(s - c))
        deviations.append(c - nearest)
    return deviations


def microtonal_content(intervals_cents: List[float], threshold: float = 25.0) -> float:
    if not intervals_cents:
        return 0.0
    devs = tuning_deviation_12tet(intervals_cents)
    microtonal = sum(1 for d in devs if abs(d) > threshold)
    return microtonal / len(devs)


def rhythm_accent_correlation(a: RhythmProfile, b: RhythmProfile) -> float:
    la, lb = len(a.accent_pattern), len(b.accent_pattern)
    if la == 0 or lb == 0:
        return 0.0
    lcm = la * lb // math.gcd(la, lb)
    va = np.array([a.accent_pattern[i % la] for i in range(lcm)])
    vb = np.array([b.accent_pattern[i % lb] for i in range(lcm)])
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


@dataclass
class TraditionFingerprint:
    tradition: Tradition
    typical_scale_sizes: List[int]
    uses_microtones: bool
    typical_cycle_lengths: List[int]
    ornament_density: float
    typical_tempo_range: Tuple[float, float]
    polyphonic: bool
    drone_based: bool


TRADITION_FINGERPRINTS: Dict[Tradition, TraditionFingerprint] = {
    Tradition.WESTERN: TraditionFingerprint(
        Tradition.WESTERN, [7, 5, 6, 8, 12], False, [4, 3, 6, 2],
        0.3, (40.0, 240.0), True, False,
    ),
    Tradition.ARABIC: TraditionFingerprint(
        Tradition.ARABIC, [7, 8, 9], True, [4, 8, 10, 16],
        0.7, (60.0, 180.0), False, True,
    ),
    Tradition.INDIAN: TraditionFingerprint(
        Tradition.INDIAN, [5, 6, 7], True, [7, 8, 10, 12, 14, 16],
        0.8, (40.0, 200.0), False, True,
    ),
    Tradition.CHINESE: TraditionFingerprint(
        Tradition.CHINESE, [5, 7], False, [4, 2, 8],
        0.4, (60.0, 160.0), True, False,
    ),
    Tradition.JAPANESE: TraditionFingerprint(
        Tradition.JAPANESE, [5, 7], False, [4, 8, 6],
        0.5, (40.0, 140.0), False, False,
    ),
    Tradition.AFRICAN: TraditionFingerprint(
        Tradition.AFRICAN, [5, 7], False, [12, 16, 8, 6],
        0.3, (80.0, 200.0), True, False,
    ),
    Tradition.INDONESIAN: TraditionFingerprint(
        Tradition.INDONESIAN, [5, 7], True, [4, 8, 16],
        0.4, (60.0, 160.0), True, False,
    ),
    Tradition.TURKISH: TraditionFingerprint(
        Tradition.TURKISH, [7, 8, 9], True, [5, 7, 8, 9, 10],
        0.7, (60.0, 180.0), False, True,
    ),
}


def tradition_distance(a: TraditionFingerprint, b: TraditionFingerprint) -> float:
    scale_overlap = len(set(a.typical_scale_sizes) & set(b.typical_scale_sizes))
    scale_union = len(set(a.typical_scale_sizes) | set(b.typical_scale_sizes))
    scale_sim = scale_overlap / scale_union if scale_union > 0 else 0.0
    cycle_overlap = len(set(a.typical_cycle_lengths) & set(b.typical_cycle_lengths))
    cycle_union = len(set(a.typical_cycle_lengths) | set(b.typical_cycle_lengths))
    cycle_sim = cycle_overlap / cycle_union if cycle_union > 0 else 0.0
    feat_dist = 0.0
    feat_dist += 0.0 if a.uses_microtones == b.uses_microtones else 0.3
    feat_dist += abs(a.ornament_density - b.ornament_density) * 0.2
    feat_dist += 0.0 if a.polyphonic == b.polyphonic else 0.2
    feat_dist += 0.0 if a.drone_based == b.drone_based else 0.15
    tempo_a = (a.typical_tempo_range[0] + a.typical_tempo_range[1]) / 2
    tempo_b = (b.typical_tempo_range[0] + b.typical_tempo_range[1]) / 2
    feat_dist += abs(tempo_a - tempo_b) / 200.0 * 0.15
    similarity = 0.3 * scale_sim + 0.25 * cycle_sim + 0.45 * (1.0 - min(feat_dist, 1.0))
    return 1.0 - similarity


def tradition_similarity_matrix() -> Tuple[List[Tradition], np.ndarray]:
    traditions = list(TRADITION_FINGERPRINTS.keys())
    n = len(traditions)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                d = tradition_distance(
                    TRADITION_FINGERPRINTS[traditions[i]],
                    TRADITION_FINGERPRINTS[traditions[j]],
                )
                matrix[i, j] = 1.0 - d
    return traditions, matrix


@dataclass
class FusionAnalysis:
    traditions: List[Tradition]
    scale_compatibility: float
    rhythm_compatibility: float
    overall_compatibility: float
    suggestions: List[str]


def analyze_fusion_potential(trad_a: Tradition, trad_b: Tradition) -> FusionAnalysis:
    fp_a = TRADITION_FINGERPRINTS.get(trad_a)
    fp_b = TRADITION_FINGERPRINTS.get(trad_b)
    if fp_a is None or fp_b is None:
        return FusionAnalysis([trad_a, trad_b], 0.5, 0.5, 0.5, [])
    scale_sim = len(set(fp_a.typical_scale_sizes) & set(fp_b.typical_scale_sizes))
    scale_compat = min(1.0, scale_sim / 2.0)
    cycle_sim = len(set(fp_a.typical_cycle_lengths) & set(fp_b.typical_cycle_lengths))
    rhythm_compat = min(1.0, cycle_sim / 2.0)
    suggestions = []
    if fp_a.uses_microtones != fp_b.uses_microtones:
        suggestions.append("Consider quarter-tone compromises or neutral intervals")
    if fp_a.drone_based != fp_b.drone_based:
        suggestions.append("Use pedal tones to bridge drone-based and harmonic traditions")
    if fp_a.polyphonic != fp_b.polyphonic:
        suggestions.append("Layer heterophonic textures for gradual polyphonic integration")
    common_scales = set(fp_a.typical_scale_sizes) & set(fp_b.typical_scale_sizes)
    if common_scales:
        suggestions.append(f"Shared scale sizes {sorted(common_scales)} are natural meeting points")
    overall = 0.5 * scale_compat + 0.5 * rhythm_compat
    return FusionAnalysis(
        traditions=[trad_a, trad_b],
        scale_compatibility=scale_compat,
        rhythm_compatibility=rhythm_compat,
        overall_compatibility=overall,
        suggestions=suggestions,
    )


def classify_tradition(scale_cents: List[float], cycle_beats: int,
                        uses_drone: bool = False,
                        ornament_level: float = 0.5) -> List[Tuple[Tradition, float]]:
    scores: List[Tuple[Tradition, float]] = []
    for trad, fp in TRADITION_FINGERPRINTS.items():
        score = 0.0
        n_notes = len(scale_cents) + 1
        if n_notes in fp.typical_scale_sizes:
            score += 0.3
        if fp.uses_microtones == (microtonal_content(scale_cents) > 0.1):
            score += 0.15
        if cycle_beats in fp.typical_cycle_lengths:
            score += 0.25
        if uses_drone == fp.drone_based:
            score += 0.15
        score += 0.15 * (1.0 - abs(ornament_level - fp.ornament_density))
        scores.append((trad, score))
    scores.sort(key=lambda x: -x[1])
    return scores
