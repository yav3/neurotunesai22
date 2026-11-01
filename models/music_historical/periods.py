"""Musical period classification and signature analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MusicalPeriod(Enum):
    MEDIEVAL = "medieval"
    RENAISSANCE = "renaissance"
    BAROQUE = "baroque"
    CLASSICAL = "classical"
    ROMANTIC = "romantic"
    IMPRESSIONIST = "impressionist"
    MODERN = "modern"
    CONTEMPORARY = "contemporary"
    AVANT_GARDE = "avant_garde"


@dataclass(frozen=True)
class PeriodProfile:
    period: MusicalPeriod
    year_start: int
    year_end: int
    typical_tempo_range: tuple[int, int]
    typical_keys: list[str]
    harmonic_features: list[str]
    texture: str
    typical_forms: list[str]


PERIOD_PROFILES: dict[MusicalPeriod, PeriodProfile] = {
    MusicalPeriod.MEDIEVAL: PeriodProfile(
        MusicalPeriod.MEDIEVAL, 500, 1400, (60, 100),
        ["D", "G"], ["modal", "organum", "parallel_fifths"],
        "monophonic_to_early_polyphony", ["plainchant", "motet", "conductus"],
    ),
    MusicalPeriod.RENAISSANCE: PeriodProfile(
        MusicalPeriod.RENAISSANCE, 1400, 1600, (60, 120),
        ["C", "F", "G", "D"], ["triadic", "modal", "imitation"],
        "polyphonic", ["motet", "madrigal", "mass", "chanson"],
    ),
    MusicalPeriod.BAROQUE: PeriodProfile(
        MusicalPeriod.BAROQUE, 1600, 1750, (60, 140),
        ["D", "G", "C", "F", "Bb", "A"], ["basso_continuo", "sequences", "circle_of_fifths"],
        "homophonic_and_polyphonic", ["fugue", "concerto", "suite", "opera", "oratorio"],
    ),
    MusicalPeriod.CLASSICAL: PeriodProfile(
        MusicalPeriod.CLASSICAL, 1750, 1820, (60, 160),
        ["C", "G", "D", "F", "Bb", "Eb"], ["functional_harmony", "alberti_bass", "cadential_64"],
        "homophonic", ["sonata", "symphony", "concerto", "string_quartet", "rondo"],
    ),
    MusicalPeriod.ROMANTIC: PeriodProfile(
        MusicalPeriod.ROMANTIC, 1820, 1900, (40, 180),
        ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"],
        ["chromatic", "augmented_6th", "enharmonic_modulation", "leitmotif"],
        "homophonic_with_rich_texture", ["symphony", "tone_poem", "opera", "lied", "nocturne"],
    ),
    MusicalPeriod.IMPRESSIONIST: PeriodProfile(
        MusicalPeriod.IMPRESSIONIST, 1875, 1925, (50, 140),
        ["C", "Db", "D", "Gb", "Ab"], ["whole_tone", "parallel_chords", "modal_mixture", "pentatonic"],
        "layered_textural", ["prelude", "suite", "tone_poem", "ballet"],
    ),
    MusicalPeriod.MODERN: PeriodProfile(
        MusicalPeriod.MODERN, 1900, 1975, (40, 200),
        [], ["atonality", "serialism", "polytonality", "quartal_harmony"],
        "varied", ["symphony", "concerto", "ballet", "chamber_music", "twelve_tone_row"],
    ),
    MusicalPeriod.CONTEMPORARY: PeriodProfile(
        MusicalPeriod.CONTEMPORARY, 1975, 2030, (30, 240),
        [], ["spectral", "microtonality", "extended_techniques", "minimalism"],
        "diverse", ["concerto", "chamber_music", "electronic", "multimedia"],
    ),
    MusicalPeriod.AVANT_GARDE: PeriodProfile(
        MusicalPeriod.AVANT_GARDE, 1950, 2030, (0, 300),
        [], ["aleatory", "graphic_notation", "noise", "silence"],
        "experimental", ["happening", "sound_installation", "chance_music"],
    ),
}


def classify_period_by_year(year: int) -> MusicalPeriod:
    for p in [
        MusicalPeriod.MEDIEVAL, MusicalPeriod.RENAISSANCE, MusicalPeriod.BAROQUE,
        MusicalPeriod.CLASSICAL, MusicalPeriod.ROMANTIC, MusicalPeriod.IMPRESSIONIST,
        MusicalPeriod.MODERN, MusicalPeriod.CONTEMPORARY,
    ]:
        prof = PERIOD_PROFILES[p]
        if prof.year_start <= year < prof.year_end:
            return p
    if year >= 2030:
        return MusicalPeriod.CONTEMPORARY
    return MusicalPeriod.MEDIEVAL


def period_harmonic_signature(period: MusicalPeriod) -> dict[str, float]:
    prof = PERIOD_PROFILES[period]
    return {feat: 1.0 for feat in prof.harmonic_features}


def style_distance(a: PeriodProfile, b: PeriodProfile) -> float:
    sa = set(a.harmonic_features)
    sb = set(b.harmonic_features)
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - len(sa & sb) / len(union)
