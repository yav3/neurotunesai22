"""Oral tradition analysis for ethnomusicology.

Variant clustering, archetype extraction, transmission stability,
geographic diffusion, and melodic drift rate computation.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class TransmissionType(enum.Enum):
    EXACT = "exact"
    VARIED = "varied"
    IMPROVISED = "improvised"
    RECONSTRUCTED = "reconstructed"


@dataclass
class OralVariant:
    melody_contour: list[float]
    text_syllables: list[str] = field(default_factory=list)
    performer_id: str = ""
    recording_date: int = 0
    region: str = ""
    transmission: TransmissionType = TransmissionType.VARIED


@dataclass
class VariantFamily:
    variants: list[OralVariant]
    archetype_contour: list[float] = field(default_factory=list)
    name: str = ""


def compute_contour_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    na, nb = np.array(a, dtype=float), np.array(b, dtype=float)
    n = max(len(na), len(nb))
    if len(na) < n:
        na = np.interp(np.linspace(0, len(na) - 1, n), np.arange(len(na)), na)
    if len(nb) < n:
        nb = np.interp(np.linspace(0, len(nb) - 1, n), np.arange(len(nb)), nb)
    cost = np.zeros((n + 1, n + 1))
    cost[0, 1:] = np.inf
    cost[1:, 0] = np.inf
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            d = abs(na[i - 1] - nb[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    dtw_dist = float(cost[n, n]) / n
    return float(np.exp(-dtw_dist))


def cluster_variants(
    variants: list[OralVariant],
    threshold: float = 0.7,
) -> list[VariantFamily]:
    assigned = [False] * len(variants)
    families: list[VariantFamily] = []
    for i in range(len(variants)):
        if assigned[i]:
            continue
        family_members = [variants[i]]
        assigned[i] = True
        for j in range(i + 1, len(variants)):
            if assigned[j]:
                continue
            sim = compute_contour_similarity(variants[i].melody_contour, variants[j].melody_contour)
            if sim >= threshold:
                family_members.append(variants[j])
                assigned[j] = True
        families.append(VariantFamily(variants=family_members))
    for fam in families:
        fam.archetype_contour = extract_archetype(fam).tolist()
    return families


def extract_archetype(family: VariantFamily) -> np.ndarray:
    if not family.variants:
        return np.array([])
    max_len = max(len(v.melody_contour) for v in family.variants)
    resampled = []
    for v in family.variants:
        c = np.array(v.melody_contour, dtype=float)
        if len(c) < max_len:
            c = np.interp(np.linspace(0, len(c) - 1, max_len), np.arange(len(c)), c)
        resampled.append(c)
    return np.mean(resampled, axis=0)


def transmission_stability(family: VariantFamily) -> float:
    if len(family.variants) < 2:
        return 1.0
    archetype = extract_archetype(family)
    sims = []
    for v in family.variants:
        sims.append(compute_contour_similarity(v.melody_contour, archetype.tolist()))
    return float(np.mean(sims))


def geographic_diffusion(
    variants: list[OralVariant],
) -> dict[str, int]:
    region_counts: dict[str, int] = {}
    for v in variants:
        r = v.region or "unknown"
        region_counts[r] = region_counts.get(r, 0) + 1
    return region_counts


def melodic_drift_rate(
    family: VariantFamily,
) -> float:
    dated = [(v.recording_date, v.melody_contour) for v in family.variants if v.recording_date > 0]
    if len(dated) < 2:
        return 0.0
    dated.sort(key=lambda x: x[0])
    total_dist = 0.0
    total_years = 0
    for i in range(len(dated) - 1):
        y1, c1 = dated[i]
        y2, c2 = dated[i + 1]
        dy = y2 - y1
        if dy <= 0:
            continue
        sim = compute_contour_similarity(c1, c2)
        total_dist += (1.0 - sim)
        total_years += dy
    if total_years == 0:
        return 0.0
    return total_dist / total_years


def syllable_alignment_score(a: OralVariant, b: OralVariant) -> float:
    sa, sb = a.text_syllables, b.text_syllables
    if not sa or not sb:
        return 0.0
    matches = sum(1 for x, y in zip(sa, sb) if x.lower() == y.lower())
    return matches / max(len(sa), len(sb))
