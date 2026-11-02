"""Computational models of musical consonance: harmonicity, periodicity, interference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ConsonanceModel", "IntervalQuality", "ConsonanceRating",
    "frequency_ratio", "harmonic_series_fit", "periodicity_pitch",
    "interference_roughness", "euler_gradus_suavitatis",
    "tenney_height", "harmonic_entropy", "consonance_rating",
    "dyad_consonance", "triad_consonance", "consonance_curve",
    "interval_ranking",
]


class ConsonanceModel(Enum):
    EULER = "euler"
    TENNEY = "tenney"
    SETHARES = "sethares"
    HARMONIC_ENTROPY = "harmonic_entropy"
    PERIODICITY = "periodicity"


class IntervalQuality(Enum):
    PERFECT = "perfect"
    IMPERFECT = "imperfect"
    DISSONANT = "dissonant"


@dataclass
class ConsonanceRating:
    interval_cents: float
    score: float
    model: ConsonanceModel
    quality: IntervalQuality


def frequency_ratio(f1: float, f2: float) -> Tuple[int, int]:
    if f1 <= 0 or f2 <= 0:
        return (0, 0)
    ratio = f2 / f1
    best_p, best_q = 1, 1
    best_err = abs(ratio - 1.0)
    for q in range(1, 32):
        p = round(ratio * q)
        if p < 1:
            continue
        err = abs(ratio - p / q)
        if err < best_err:
            best_err = err
            best_p, best_q = p, q
        if best_err < 1e-6:
            break
    return (best_p, best_q)


def harmonic_series_fit(partials: List[float], tolerance_cents: float = 50.0) -> float:
    if len(partials) < 2:
        return 1.0
    f0 = partials[0]
    if f0 <= 0:
        return 0.0
    matched = 0
    for f in partials:
        ratio = f / f0
        nearest_int = round(ratio)
        if nearest_int < 1:
            continue
        deviation_cents = abs(1200.0 * math.log2(ratio / nearest_int)) if ratio > 0 and nearest_int > 0 else float("inf")
        if deviation_cents <= tolerance_cents:
            matched += 1
    return matched / len(partials)


def periodicity_pitch(freqs: List[float]) -> float:
    if not freqs or all(f <= 0 for f in freqs):
        return 0.0
    positive = [f for f in freqs if f > 0]
    if len(positive) == 1:
        return positive[0]
    result = positive[0]
    for f in positive[1:]:
        result = _gcd_freq(result, f)
    return result


def _gcd_freq(a: float, b: float, tol: float = 0.5) -> float:
    if a < tol:
        return b
    if b < tol:
        return a
    while b > tol:
        a, b = b, a % b
    return a


def interference_roughness(f1: float, f2: float, a1: float = 1.0, a2: float = 1.0) -> float:
    if f1 <= 0 or f2 <= 0:
        return 0.0
    s = abs(f2 - f1)
    f_min = min(f1, f2)
    cb = 1.72 * (f_min ** 0.65)
    x = s / max(cb, 1e-6)
    roughness = a1 * a2 * (x * math.exp(1 - x)) if x > 0 else 0.0
    return max(0.0, roughness)


def euler_gradus_suavitatis(p: int, q: int) -> int:
    if p <= 0 or q <= 0:
        return 0
    n = p * q
    gs = 1
    d = 2
    while n > 1:
        while n % d == 0:
            gs += d - 1
            n //= d
        d += 1
        if d * d > n and n > 1:
            gs += n - 1
            break
    return gs


def tenney_height(p: int, q: int) -> float:
    if p <= 0 or q <= 0:
        return float("inf")
    return math.log2(p * q)


def harmonic_entropy(cents: float, sigma: float = 12.0) -> float:
    ratios = []
    for q in range(1, 20):
        for p in range(q, 3 * q + 1):
            r_cents = 1200.0 * math.log2(p / q) if q > 0 else 0.0
            weight = math.exp(-0.5 * ((cents - r_cents) / sigma) ** 2)
            if weight > 1e-10:
                ratios.append((p, q, weight))

    if not ratios:
        return 0.0
    total_weight = sum(w for _, _, w in ratios)
    if total_weight <= 0:
        return 0.0

    entropy = 0.0
    for _, _, w in ratios:
        prob = w / total_weight
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy


def consonance_rating(
    cents: float,
    model: ConsonanceModel = ConsonanceModel.HARMONIC_ENTROPY,
) -> ConsonanceRating:
    cents = cents % 1200.0

    if model == ConsonanceModel.HARMONIC_ENTROPY:
        he = harmonic_entropy(cents)
        max_he = harmonic_entropy(50.0)
        score = max(0.0, 1.0 - he / max(max_he, 1e-6))
    elif model == ConsonanceModel.EULER:
        ratio = 2.0 ** (cents / 1200.0)
        p, q = frequency_ratio(1.0, ratio)
        gs = euler_gradus_suavitatis(p, q)
        score = max(0.0, 1.0 - gs / 20.0)
    elif model == ConsonanceModel.TENNEY:
        ratio = 2.0 ** (cents / 1200.0)
        p, q = frequency_ratio(1.0, ratio)
        th = tenney_height(p, q)
        score = max(0.0, 1.0 - th / 10.0)
    elif model == ConsonanceModel.SETHARES:
        f1 = 440.0
        f2 = f1 * 2.0 ** (cents / 1200.0)
        r = interference_roughness(f1, f2)
        score = max(0.0, 1.0 - r)
    else:
        score = 0.5

    if score >= 0.7:
        quality = IntervalQuality.PERFECT
    elif score >= 0.4:
        quality = IntervalQuality.IMPERFECT
    else:
        quality = IntervalQuality.DISSONANT

    return ConsonanceRating(
        interval_cents=cents, score=score, model=model, quality=quality,
    )


def dyad_consonance(f1: float, f2: float, model: ConsonanceModel = ConsonanceModel.SETHARES) -> float:
    if f1 <= 0 or f2 <= 0:
        return 0.0
    cents = abs(1200.0 * math.log2(f2 / f1))
    return consonance_rating(cents, model).score


def triad_consonance(f1: float, f2: float, f3: float, model: ConsonanceModel = ConsonanceModel.SETHARES) -> float:
    d12 = dyad_consonance(f1, f2, model)
    d13 = dyad_consonance(f1, f3, model)
    d23 = dyad_consonance(f2, f3, model)
    return (d12 + d13 + d23) / 3.0


def consonance_curve(
    f_base: float = 440.0,
    cents_range: Tuple[float, float] = (0.0, 1200.0),
    step_cents: float = 10.0,
    model: ConsonanceModel = ConsonanceModel.HARMONIC_ENTROPY,
) -> List[Tuple[float, float]]:
    result = []
    c = cents_range[0]
    while c <= cents_range[1]:
        score = consonance_rating(c, model).score
        result.append((c, score))
        c += step_cents
    return result


def interval_ranking(model: ConsonanceModel = ConsonanceModel.HARMONIC_ENTROPY) -> List[Tuple[str, float, float]]:
    intervals = [
        ("unison", 0.0), ("m2", 100.0), ("M2", 200.0), ("m3", 300.0),
        ("M3", 400.0), ("P4", 500.0), ("tritone", 600.0), ("P5", 700.0),
        ("m6", 800.0), ("M6", 900.0), ("m7", 1000.0), ("M7", 1100.0),
        ("octave", 1200.0),
    ]
    ranked = []
    for name, cents in intervals:
        score = consonance_rating(cents, model).score
        ranked.append((name, cents, score))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked
