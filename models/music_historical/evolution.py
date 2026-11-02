"""Style evolution tracking across musical periods."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .periods import MusicalPeriod, PERIOD_PROFILES


@dataclass
class StyleFeatureVector:
    dissonance_level: float = 0.0
    rhythmic_complexity: float = 0.0
    harmonic_vocabulary_size: int = 3
    avg_phrase_length: float = 4.0
    polyphony_degree: int = 1
    dynamic_range: float = 0.5


_PERIOD_DEFAULTS: dict[MusicalPeriod, StyleFeatureVector] = {
    MusicalPeriod.MEDIEVAL: StyleFeatureVector(0.1, 0.2, 3, 8.0, 1, 0.2),
    MusicalPeriod.RENAISSANCE: StyleFeatureVector(0.15, 0.3, 5, 6.0, 4, 0.3),
    MusicalPeriod.BAROQUE: StyleFeatureVector(0.2, 0.5, 8, 4.0, 4, 0.5),
    MusicalPeriod.CLASSICAL: StyleFeatureVector(0.15, 0.4, 7, 4.0, 4, 0.6),
    MusicalPeriod.ROMANTIC: StyleFeatureVector(0.3, 0.5, 12, 8.0, 6, 0.9),
    MusicalPeriod.IMPRESSIONIST: StyleFeatureVector(0.35, 0.4, 10, 6.0, 3, 0.7),
    MusicalPeriod.MODERN: StyleFeatureVector(0.6, 0.7, 12, 3.0, 4, 0.8),
    MusicalPeriod.CONTEMPORARY: StyleFeatureVector(0.5, 0.8, 12, 4.0, 4, 0.9),
    MusicalPeriod.AVANT_GARDE: StyleFeatureVector(0.8, 0.9, 12, 2.0, 6, 1.0),
}


def style_evolution_trajectory(
    period_sequence: list[MusicalPeriod],
) -> list[StyleFeatureVector]:
    return [_PERIOD_DEFAULTS.get(p, StyleFeatureVector()) for p in period_sequence]


def _to_vec(sfv: StyleFeatureVector) -> np.ndarray:
    return np.array([
        sfv.dissonance_level, sfv.rhythmic_complexity,
        sfv.harmonic_vocabulary_size / 12.0, sfv.avg_phrase_length / 8.0,
        sfv.polyphony_degree / 6.0, sfv.dynamic_range,
    ])


def innovation_score(features: StyleFeatureVector, period: MusicalPeriod) -> float:
    norm = _PERIOD_DEFAULTS.get(period, StyleFeatureVector())
    diff = _to_vec(features) - _to_vec(norm)
    return float(np.linalg.norm(diff))


def influence_graph(
    composers_periods: dict[str, MusicalPeriod],
) -> dict[str, list[str]]:
    period_order = list(MusicalPeriod)
    graph: dict[str, list[str]] = {c: [] for c in composers_periods}
    composers = list(composers_periods.items())
    for i, (name_a, period_a) in enumerate(composers):
        idx_a = period_order.index(period_a)
        for name_b, period_b in composers[i + 1:]:
            idx_b = period_order.index(period_b)
            if abs(idx_a - idx_b) <= 1 and idx_a < idx_b:
                graph[name_a].append(name_b)
            elif abs(idx_a - idx_b) <= 1 and idx_b < idx_a:
                graph[name_b].append(name_a)
    return graph
