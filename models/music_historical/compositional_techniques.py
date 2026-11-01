"""Compositional techniques: counterpoint species, motivic development, orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class CounterpointSpecies(Enum):
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"
    FOURTH = "fourth"
    FIFTH = "fifth"
    FREE = "free"
    INVERTIBLE = "invertible"
    SERIAL = "serial"


class MotivicTechnique(Enum):
    SEQUENCE = "sequence"
    INVERSION = "inversion"
    RETROGRADE = "retrograde"
    RETROGRADE_INVERSION = "retrograde_inversion"
    AUGMENTATION = "augmentation"
    DIMINUTION = "diminution"
    FRAGMENTATION = "fragmentation"
    STRETTO = "stretto"
    LIQUIDATION = "liquidation"
    DEVELOPING_VARIATION = "developing_variation"


@dataclass
class CompositionalTechnique:
    name: str
    period: str
    complexity_score: float
    harmonic_density: float
    polyphonic_density: float
    description: str = ""


TECHNIQUE_CATALOG: dict[str, list[CompositionalTechnique]] = {
    "MEDIEVAL": [
        CompositionalTechnique("organum", "MEDIEVAL", 0.2, 0.1, 0.3),
        CompositionalTechnique("hocket", "MEDIEVAL", 0.3, 0.1, 0.4),
        CompositionalTechnique("isorhythm", "MEDIEVAL", 0.5, 0.2, 0.5),
    ],
    "RENAISSANCE": [
        CompositionalTechnique("imitative_counterpoint", "RENAISSANCE", 0.6, 0.4, 0.7),
        CompositionalTechnique("cantus_firmus", "RENAISSANCE", 0.4, 0.3, 0.5),
        CompositionalTechnique("word_painting", "RENAISSANCE", 0.5, 0.4, 0.6),
        CompositionalTechnique("parody_technique", "RENAISSANCE", 0.5, 0.4, 0.6),
    ],
    "BAROQUE": [
        CompositionalTechnique("fugue", "BAROQUE", 0.8, 0.6, 0.9),
        CompositionalTechnique("basso_continuo", "BAROQUE", 0.5, 0.5, 0.4),
        CompositionalTechnique("ritornello", "BAROQUE", 0.6, 0.5, 0.6),
        CompositionalTechnique("terraced_dynamics", "BAROQUE", 0.3, 0.4, 0.5),
        CompositionalTechnique("sequence", "BAROQUE", 0.5, 0.5, 0.5),
    ],
    "CLASSICAL": [
        CompositionalTechnique("sonata_form", "CLASSICAL", 0.7, 0.5, 0.5),
        CompositionalTechnique("theme_and_variations", "CLASSICAL", 0.6, 0.5, 0.4),
        CompositionalTechnique("alberti_bass", "CLASSICAL", 0.3, 0.4, 0.3),
        CompositionalTechnique("periodic_phrasing", "CLASSICAL", 0.4, 0.4, 0.3),
    ],
    "ROMANTIC": [
        CompositionalTechnique("leitmotif", "ROMANTIC", 0.7, 0.7, 0.6),
        CompositionalTechnique("thematic_transformation", "ROMANTIC", 0.8, 0.7, 0.6),
        CompositionalTechnique("cyclic_form", "ROMANTIC", 0.7, 0.6, 0.5),
        CompositionalTechnique("chromatic_saturation", "ROMANTIC", 0.8, 0.9, 0.6),
        CompositionalTechnique("orchestral_color", "ROMANTIC", 0.7, 0.6, 0.7),
    ],
    "IMPRESSIONIST": [
        CompositionalTechnique("planing", "IMPRESSIONIST", 0.5, 0.7, 0.4),
        CompositionalTechnique("non_functional_harmony", "IMPRESSIONIST", 0.6, 0.8, 0.4),
        CompositionalTechnique("whole_tone_scale", "IMPRESSIONIST", 0.4, 0.6, 0.3),
        CompositionalTechnique("pentatonic_overlay", "IMPRESSIONIST", 0.4, 0.5, 0.3),
    ],
    "EARLY_MODERN": [
        CompositionalTechnique("serialism", "EARLY_MODERN", 0.9, 0.8, 0.7),
        CompositionalTechnique("twelve_tone", "EARLY_MODERN", 0.9, 0.7, 0.7),
        CompositionalTechnique("polyrhythm", "EARLY_MODERN", 0.8, 0.5, 0.8),
        CompositionalTechnique("bitonality", "EARLY_MODERN", 0.7, 0.9, 0.5),
        CompositionalTechnique("neoclassical_form", "EARLY_MODERN", 0.6, 0.5, 0.5),
    ],
    "POSTMODERN": [
        CompositionalTechnique("minimalism", "POSTMODERN", 0.4, 0.3, 0.3),
        CompositionalTechnique("aleatory", "POSTMODERN", 0.7, 0.5, 0.5),
        CompositionalTechnique("spectral_composition", "POSTMODERN", 0.8, 0.9, 0.4),
        CompositionalTechnique("collage", "POSTMODERN", 0.6, 0.6, 0.5),
        CompositionalTechnique("process_music", "POSTMODERN", 0.5, 0.4, 0.4),
    ],
    "CONTEMPORARY": [
        CompositionalTechnique("algorithmic", "CONTEMPORARY", 0.8, 0.6, 0.6),
        CompositionalTechnique("granular_synthesis", "CONTEMPORARY", 0.7, 0.7, 0.3),
        CompositionalTechnique("microtonal", "CONTEMPORARY", 0.8, 0.9, 0.4),
        CompositionalTechnique("live_electronics", "CONTEMPORARY", 0.7, 0.5, 0.5),
        CompositionalTechnique("spatial_composition", "CONTEMPORARY", 0.7, 0.5, 0.6),
    ],
}


def technique_complexity_by_period(period: str) -> float:
    techniques = TECHNIQUE_CATALOG.get(period, [])
    if not techniques:
        return 0.0
    return float(np.mean([t.complexity_score for t in techniques]))


def harmonic_rhythm_density(
    chords_per_bar: float,
    tempo_bpm: float,
    time_signature_num: int = 4,
    time_signature_den: int = 4,
) -> float:
    if tempo_bpm <= 0 or time_signature_den <= 0:
        return 0.0
    beats_per_second = tempo_bpm / 60.0
    bar_duration = time_signature_num / beats_per_second
    if bar_duration <= 0:
        return 0.0
    return chords_per_bar / bar_duration


def motivic_development_score(
    n_transformations: int,
    n_unique_motives: int,
    total_measures: int,
) -> float:
    if total_measures <= 0 or n_unique_motives <= 0:
        return 0.0
    density = n_transformations / total_measures
    variety = min(n_transformations / n_unique_motives, 5.0) / 5.0
    return float(np.tanh(density * variety * 2.0))


def orchestration_complexity(
    n_instruments: int,
    n_independent_parts: int,
    doublings: int = 0,
    divisi_sections: int = 0,
) -> float:
    if n_instruments <= 0:
        return 0.0
    independence = n_independent_parts / n_instruments
    texture = 1.0 + 0.1 * divisi_sections - 0.05 * doublings
    raw = independence * texture * np.log1p(n_instruments)
    return float(np.clip(raw / 5.0, 0.0, 1.0))
