"""Composer influence network and stylistic clustering."""

from __future__ import annotations

from dataclasses import dataclass, field
from .periods import MusicalPeriod


@dataclass(frozen=True)
class ComposerProfile:
    name: str
    birth_year: int
    death_year: int
    primary_period: MusicalPeriod
    nationalities: list[str]
    primary_forms: list[str]
    harmonic_innovations: list[str] = field(default_factory=list)


CANONICAL_COMPOSERS: list[ComposerProfile] = [
    ComposerProfile("Hildegard von Bingen", 1098, 1179, MusicalPeriod.MEDIEVAL,
                     ["German"], ["plainchant", "antiphon"], []),
    ComposerProfile("Guillaume de Machaut", 1300, 1377, MusicalPeriod.MEDIEVAL,
                     ["French"], ["motet", "ballade", "mass"], ["isorhythm"]),
    ComposerProfile("Josquin des Prez", 1450, 1521, MusicalPeriod.RENAISSANCE,
                     ["Franco-Flemish"], ["motet", "mass", "chanson"], ["pervasive_imitation"]),
    ComposerProfile("Giovanni Pierluigi da Palestrina", 1525, 1594, MusicalPeriod.RENAISSANCE,
                     ["Italian"], ["mass", "motet"], ["counterpoint_refinement"]),
    ComposerProfile("Claudio Monteverdi", 1567, 1643, MusicalPeriod.BAROQUE,
                     ["Italian"], ["opera", "madrigal"], ["seconda_pratica"]),
    ComposerProfile("Johann Sebastian Bach", 1685, 1750, MusicalPeriod.BAROQUE,
                     ["German"], ["fugue", "cantata", "concerto", "suite"],
                     ["chromatic_harmony", "counterpoint_mastery"]),
    ComposerProfile("George Frideric Handel", 1685, 1759, MusicalPeriod.BAROQUE,
                     ["German", "British"], ["opera", "oratorio", "concerto"], []),
    ComposerProfile("Wolfgang Amadeus Mozart", 1756, 1791, MusicalPeriod.CLASSICAL,
                     ["Austrian"], ["symphony", "concerto", "opera", "string_quartet"],
                     ["chromatic_voice_leading"]),
    ComposerProfile("Ludwig van Beethoven", 1770, 1827, MusicalPeriod.CLASSICAL,
                     ["German"], ["symphony", "sonata", "string_quartet", "concerto"],
                     ["motivic_development", "expanded_form"]),
    ComposerProfile("Frederic Chopin", 1810, 1849, MusicalPeriod.ROMANTIC,
                     ["Polish", "French"], ["nocturne", "etude", "ballade", "mazurka"],
                     ["chromatic_harmony", "rubato"]),
    ComposerProfile("Richard Wagner", 1813, 1883, MusicalPeriod.ROMANTIC,
                     ["German"], ["opera"], ["leitmotif", "tristan_chord", "endless_melody"]),
    ComposerProfile("Claude Debussy", 1862, 1918, MusicalPeriod.IMPRESSIONIST,
                     ["French"], ["prelude", "suite", "opera"],
                     ["whole_tone", "parallel_chords", "modal_mixture"]),
    ComposerProfile("Igor Stravinsky", 1882, 1971, MusicalPeriod.MODERN,
                     ["Russian", "French", "American"], ["ballet", "symphony", "concerto"],
                     ["polyrhythm", "neoclassicism", "serialism"]),
    ComposerProfile("Arnold Schoenberg", 1874, 1951, MusicalPeriod.MODERN,
                     ["Austrian", "American"], ["chamber_music", "opera"],
                     ["atonality", "twelve_tone_technique"]),
    ComposerProfile("John Cage", 1912, 1992, MusicalPeriod.AVANT_GARDE,
                     ["American"], ["chance_music", "prepared_piano"],
                     ["aleatory", "silence_as_music"]),
]


def composers_by_period(period: MusicalPeriod) -> list[ComposerProfile]:
    return [c for c in CANONICAL_COMPOSERS if c.primary_period == period]


def contemporaries(composer: ComposerProfile, overlap_years: int = 10) -> list[ComposerProfile]:
    result = []
    for c in CANONICAL_COMPOSERS:
        if c.name == composer.name:
            continue
        overlap_start = max(composer.birth_year, c.birth_year)
        overlap_end = min(composer.death_year, c.death_year)
        if overlap_end - overlap_start >= overlap_years:
            result.append(c)
    return result


def innovation_chain() -> list[tuple[str, str, str]]:
    chain = []
    period_order = list(MusicalPeriod)
    sorted_composers = sorted(CANONICAL_COMPOSERS, key=lambda c: c.birth_year)
    for i, c in enumerate(sorted_composers):
        for innovation in c.harmonic_innovations:
            for later in sorted_composers[i + 1:]:
                if innovation in later.harmonic_innovations:
                    chain.append((c.name, later.name, innovation))
    return chain


def stylistic_similarity(a: ComposerProfile, b: ComposerProfile) -> float:
    form_a = set(a.primary_forms)
    form_b = set(b.primary_forms)
    form_sim = len(form_a & form_b) / max(len(form_a | form_b), 1)

    innov_a = set(a.harmonic_innovations)
    innov_b = set(b.harmonic_innovations)
    innov_sim = len(innov_a & innov_b) / max(len(innov_a | innov_b), 1) if (innov_a or innov_b) else 0.0

    period_order = list(MusicalPeriod)
    try:
        period_dist = abs(period_order.index(a.primary_period) - period_order.index(b.primary_period))
    except ValueError:
        period_dist = len(period_order)
    period_sim = max(0.0, 1.0 - period_dist / (len(period_order) - 1))

    return 0.4 * form_sim + 0.3 * innov_sim + 0.3 * period_sim
