"""Historical period classification and style analysis.

Bayesian period classification from musical features, style distance
metrics, temporal evolution tracking, and influence graph construction.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class PeriodLabel(enum.Enum):
    MEDIEVAL = "medieval"
    RENAISSANCE = "renaissance"
    BAROQUE = "baroque"
    CLASSICAL = "classical"
    ROMANTIC = "romantic"
    IMPRESSIONIST = "impressionist"
    MODERN = "modern"
    CONTEMPORARY = "contemporary"


@dataclass
class StyleFeatures:
    avg_consonance: float = 0.5
    rhythmic_complexity: float = 0.5
    dynamic_range: float = 0.5
    harmonic_vocabulary_size: float = 0.5
    texture_density: float = 0.5
    ornamentation_density: float = 0.5

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.avg_consonance, self.rhythmic_complexity,
            self.dynamic_range, self.harmonic_vocabulary_size,
            self.texture_density, self.ornamentation_density,
        ])


PERIOD_PROTOTYPES: dict[PeriodLabel, StyleFeatures] = {
    PeriodLabel.MEDIEVAL: StyleFeatures(0.7, 0.2, 0.2, 0.15, 0.2, 0.1),
    PeriodLabel.RENAISSANCE: StyleFeatures(0.75, 0.3, 0.3, 0.3, 0.5, 0.15),
    PeriodLabel.BAROQUE: StyleFeatures(0.6, 0.6, 0.5, 0.5, 0.6, 0.7),
    PeriodLabel.CLASSICAL: StyleFeatures(0.7, 0.5, 0.6, 0.45, 0.4, 0.3),
    PeriodLabel.ROMANTIC: StyleFeatures(0.5, 0.6, 0.85, 0.7, 0.7, 0.5),
    PeriodLabel.IMPRESSIONIST: StyleFeatures(0.55, 0.4, 0.6, 0.8, 0.5, 0.3),
    PeriodLabel.MODERN: StyleFeatures(0.3, 0.8, 0.7, 0.9, 0.6, 0.2),
    PeriodLabel.CONTEMPORARY: StyleFeatures(0.4, 0.7, 0.8, 0.85, 0.55, 0.25),
}

PERIOD_PRIORS: dict[PeriodLabel, float] = {p: 1.0 / len(PeriodLabel) for p in PeriodLabel}


def _gaussian_likelihood(features: np.ndarray, prototype: np.ndarray, sigma: float = 0.2) -> float:
    diff = features - prototype
    return float(np.exp(-0.5 * np.sum(diff ** 2) / (sigma ** 2)))


def period_likelihood(
    features: StyleFeatures,
    sigma: float = 0.2,
    priors: dict[PeriodLabel, float] | None = None,
) -> dict[PeriodLabel, float]:
    if priors is None:
        priors = PERIOD_PRIORS
    vec = features.as_vector()
    raw: dict[PeriodLabel, float] = {}
    for period, proto in PERIOD_PROTOTYPES.items():
        raw[period] = _gaussian_likelihood(vec, proto.as_vector(), sigma) * priors.get(period, 0.0)
    total = sum(raw.values())
    if total < 1e-30:
        return {p: 1.0 / len(raw) for p in raw}
    return {p: v / total for p, v in raw.items()}


def dominant_period(features: StyleFeatures) -> PeriodLabel:
    posteriors = period_likelihood(features)
    return max(posteriors, key=posteriors.get)  # type: ignore[arg-type]


def style_distance(a: StyleFeatures, b: StyleFeatures) -> float:
    return float(np.linalg.norm(a.as_vector() - b.as_vector()))


def temporal_evolution(
    collection: list[tuple[int, StyleFeatures]],
) -> list[tuple[int, int, float]]:
    if len(collection) < 2:
        return []
    sorted_coll = sorted(collection, key=lambda x: x[0])
    drifts: list[tuple[int, int, float]] = []
    for i in range(len(sorted_coll) - 1):
        y1, f1 = sorted_coll[i]
        y2, f2 = sorted_coll[i + 1]
        drifts.append((y1, y2, style_distance(f1, f2)))
    return drifts


def influence_graph(
    threshold: float = 0.3,
) -> list[tuple[PeriodLabel, PeriodLabel, float]]:
    periods = list(PeriodLabel)
    edges: list[tuple[PeriodLabel, PeriodLabel, float]] = []
    for i in range(len(periods) - 1):
        p1 = periods[i]
        p2 = periods[i + 1]
        proto1 = PERIOD_PROTOTYPES[p1]
        proto2 = PERIOD_PROTOTYPES[p2]
        overlap = 1.0 - style_distance(proto1, proto2)
        if overlap > threshold:
            edges.append((p1, p2, overlap))
    return edges
