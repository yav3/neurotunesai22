"""Evolution of musical forms across periods: form detection, section similarity, lineage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

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


class FormType(Enum):
    BINARY = "binary"
    TERNARY = "ternary"
    RONDO = "rondo"
    SONATA_ALLEGRO = "sonata_allegro"
    STROPHIC = "strophic"
    THROUGH_COMPOSED = "through_composed"
    THEME_VARIATIONS = "theme_variations"
    FUGAL = "fugal"


CANONICAL_LABEL_SEQUENCES: dict[FormType, list[list[str]]] = {
    FormType.BINARY: [["A", "B"], ["A", "A'", "B", "B'"]],
    FormType.TERNARY: [["A", "B", "A"], ["A", "B", "A'"]],
    FormType.RONDO: [["A", "B", "A", "C", "A"], ["A", "B", "A", "C", "A", "B", "A"]],
    FormType.SONATA_ALLEGRO: [["Exp", "Dev", "Recap"], ["Intro", "Exp", "Dev", "Recap", "Coda"]],
    FormType.STROPHIC: [["A", "A", "A"], ["A", "A", "A", "A"]],
    FormType.THROUGH_COMPOSED: [["A", "B", "C"], ["A", "B", "C", "D"]],
    FormType.THEME_VARIATIONS: [["T", "V1", "V2", "V3"], ["T", "V1", "V2", "V3", "V4"]],
    FormType.FUGAL: [["S", "A", "Ep", "S'", "Str"], ["S", "A", "CS", "Ep", "S'"]],
}


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
    chain: list[MusicalForm] = []
    current: MusicalForm | None = form
    visited: set[MusicalForm] = set()
    while current is not None and current not in visited:
        chain.append(current)
        visited.add(current)
        parents = _FORM_LINEAGE.get(current, [])
        current = parents[0] if parents else None
    return list(reversed(chain))


def section_similarity(
    features_a: np.ndarray,
    features_b: np.ndarray,
) -> float:
    if features_a.ndim == 1:
        features_a = features_a.reshape(1, -1)
    if features_b.ndim == 1:
        features_b = features_b.reshape(1, -1)
    mean_a = features_a.mean(axis=0)
    mean_b = features_b.mean(axis=0)
    norm_a = np.linalg.norm(mean_a)
    norm_b = np.linalg.norm(mean_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(mean_a, mean_b) / (norm_a * norm_b))


def detect_form(
    section_labels: Sequence[str],
    threshold: float = 0.6,
) -> FormType | None:
    labels = list(section_labels)
    n = len(labels)
    if n < 2:
        return None
    best_form: FormType | None = None
    best_score = threshold
    for form_type, canonical_seqs in CANONICAL_LABEL_SEQUENCES.items():
        for canon in canonical_seqs:
            if len(canon) != n:
                continue
            matches = sum(1 for a, b in zip(labels, canon) if a == b)
            score = matches / n
            if score > best_score:
                best_score = score
                best_form = form_type
    if best_form is not None:
        return best_form
    unique = set(labels)
    if len(unique) == 1:
        return FormType.STROPHIC
    if n >= 3 and labels[0] == labels[-1] and labels[0] != labels[1]:
        return FormType.TERNARY
    if n == 2 and labels[0] != labels[1]:
        return FormType.BINARY
    if len(unique) == n:
        return FormType.THROUGH_COMPOSED
    return None


def repetition_rate(section_labels: Sequence[str]) -> float:
    labels = list(section_labels)
    n = len(labels)
    if n <= 1:
        return 0.0
    first_occ: dict[str, int] = {}
    repeats = 0
    for i, lab in enumerate(labels):
        if lab in first_occ:
            repeats += 1
        else:
            first_occ[lab] = i
    return repeats / n


@dataclass(frozen=True)
class FormAnalysis:
    detected_form: FormType | None
    section_labels: list[str]
    repetition_rate: float
    unique_sections: int
    symmetry_score: float


def analyze_form(section_labels: Sequence[str]) -> FormAnalysis:
    labels = list(section_labels)
    n = len(labels)
    detected = detect_form(labels)
    rep = repetition_rate(labels)
    unique = len(set(labels))
    sym = 0.0
    if n >= 2:
        half = n // 2
        matches = sum(1 for i in range(half) if labels[i] == labels[n - 1 - i])
        sym = matches / half
    return FormAnalysis(
        detected_form=detected,
        section_labels=labels,
        repetition_rate=rep,
        unique_sections=unique,
        symmetry_score=sym,
    )


def period_from_year(year: int) -> MusicalPeriod:
    if year < 1400:
        return MusicalPeriod.MEDIEVAL
    if year < 1600:
        return MusicalPeriod.RENAISSANCE
    if year < 1750:
        return MusicalPeriod.BAROQUE
    if year < 1820:
        return MusicalPeriod.CLASSICAL
    if year < 1890:
        return MusicalPeriod.ROMANTIC
    if year < 1920:
        return MusicalPeriod.IMPRESSIONIST
    return MusicalPeriod.MODERN
