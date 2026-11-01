"""Evolution of musical forms across periods."""

from __future__ import annotations

from enum import Enum
from .periods import MusicalPeriod


class MusicalForm(Enum):
    PLAINCHANT = "plainchant"
    MOTET = "motet"
    MADRIGAL = "madrigal"
    FUGUE = "fugue"
    SONATA = "sonata"
    CONCERTO = "concerto"
    SYMPHONY = "symphony"
    RONDO = "rondo"
    THEME_AND_VARIATIONS = "theme_and_variations"
    TONE_POEM = "tone_poem"
    TWELVE_TONE_ROW = "twelve_tone_row"
    OPERA = "opera"


_FORM_PERIODS: dict[MusicalForm, list[MusicalPeriod]] = {
    MusicalForm.PLAINCHANT: [MusicalPeriod.MEDIEVAL],
    MusicalForm.MOTET: [MusicalPeriod.MEDIEVAL, MusicalPeriod.RENAISSANCE, MusicalPeriod.BAROQUE],
    MusicalForm.MADRIGAL: [MusicalPeriod.RENAISSANCE],
    MusicalForm.FUGUE: [MusicalPeriod.BAROQUE, MusicalPeriod.CLASSICAL],
    MusicalForm.SONATA: [MusicalPeriod.CLASSICAL, MusicalPeriod.ROMANTIC],
    MusicalForm.CONCERTO: [MusicalPeriod.BAROQUE, MusicalPeriod.CLASSICAL, MusicalPeriod.ROMANTIC, MusicalPeriod.MODERN],
    MusicalForm.SYMPHONY: [MusicalPeriod.CLASSICAL, MusicalPeriod.ROMANTIC, MusicalPeriod.MODERN],
    MusicalForm.RONDO: [MusicalPeriod.CLASSICAL, MusicalPeriod.ROMANTIC],
    MusicalForm.THEME_AND_VARIATIONS: [MusicalPeriod.BAROQUE, MusicalPeriod.CLASSICAL, MusicalPeriod.ROMANTIC],
    MusicalForm.TONE_POEM: [MusicalPeriod.ROMANTIC, MusicalPeriod.IMPRESSIONIST],
    MusicalForm.TWELVE_TONE_ROW: [MusicalPeriod.MODERN],
    MusicalForm.OPERA: [MusicalPeriod.BAROQUE, MusicalPeriod.CLASSICAL, MusicalPeriod.ROMANTIC, MusicalPeriod.MODERN],
}

_FORM_COMPLEXITY: dict[MusicalForm, float] = {
    MusicalForm.PLAINCHANT: 0.1,
    MusicalForm.MOTET: 0.4,
    MusicalForm.MADRIGAL: 0.35,
    MusicalForm.FUGUE: 0.9,
    MusicalForm.SONATA: 0.8,
    MusicalForm.CONCERTO: 0.75,
    MusicalForm.SYMPHONY: 0.85,
    MusicalForm.RONDO: 0.5,
    MusicalForm.THEME_AND_VARIATIONS: 0.6,
    MusicalForm.TONE_POEM: 0.7,
    MusicalForm.TWELVE_TONE_ROW: 0.95,
    MusicalForm.OPERA: 0.9,
}

_FORM_LINEAGE: dict[MusicalForm, list[MusicalForm]] = {
    MusicalForm.PLAINCHANT: [],
    MusicalForm.MOTET: [MusicalForm.PLAINCHANT],
    MusicalForm.MADRIGAL: [MusicalForm.MOTET],
    MusicalForm.FUGUE: [MusicalForm.MOTET],
    MusicalForm.SONATA: [MusicalForm.FUGUE],
    MusicalForm.CONCERTO: [MusicalForm.SONATA],
    MusicalForm.SYMPHONY: [MusicalForm.SONATA],
    MusicalForm.RONDO: [MusicalForm.SONATA],
    MusicalForm.THEME_AND_VARIATIONS: [MusicalForm.SONATA],
    MusicalForm.TONE_POEM: [MusicalForm.SYMPHONY],
    MusicalForm.TWELVE_TONE_ROW: [MusicalForm.FUGUE],
    MusicalForm.OPERA: [MusicalForm.MADRIGAL],
}


def form_period_distribution() -> dict[MusicalForm, list[MusicalPeriod]]:
    return dict(_FORM_PERIODS)


def form_complexity_score(form: MusicalForm) -> float:
    return _FORM_COMPLEXITY.get(form, 0.5)


def form_lineage(form: MusicalForm) -> list[MusicalForm]:
    chain = []
    current = form
    visited = set()
    while current and current not in visited:
        chain.append(current)
        visited.add(current)
        parents = _FORM_LINEAGE.get(current, [])
        current = parents[0] if parents else None
    return list(reversed(chain))
