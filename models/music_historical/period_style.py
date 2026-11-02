"""Musical period classification and style characteristics."""
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
    POSTMODERN = "postmodern"
    CONTEMPORARY = "contemporary"


@dataclass
class PeriodCharacteristics:
    period: MusicalPeriod
    year_range: tuple[int, int]
    typical_forms: list[str]
    harmonic_language: str
    texture: str
    rhythm: str
    dynamics: str
    notable_innovations: list[str] = field(default_factory=list)


PERIOD_DATABASE: dict[MusicalPeriod, PeriodCharacteristics] = {
    MusicalPeriod.MEDIEVAL: PeriodCharacteristics(
        MusicalPeriod.MEDIEVAL, (500, 1400),
        ["plainchant", "organum", "motet", "madrigal"],
        "modal, parallel organum, limited polyphony",
        "monophonic to early polyphonic",
        "free rhythm (chant) to mensural notation",
        "terraced, limited range",
        ["notation systems", "polyphony", "isorhythm"],
    ),
    MusicalPeriod.RENAISSANCE: PeriodCharacteristics(
        MusicalPeriod.RENAISSANCE, (1400, 1600),
        ["mass", "motet", "chanson", "madrigal", "ricercar"],
        "modal, triadic harmony emerging, fauxbourdon",
        "imitative polyphony, homophony emerging",
        "regular tactus, proportional notation",
        "gradual, text-driven",
        ["equal temperament exploration", "music printing", "word painting"],
    ),
    MusicalPeriod.BAROQUE: PeriodCharacteristics(
        MusicalPeriod.BAROQUE, (1600, 1750),
        ["opera", "concerto grosso", "fugue", "suite", "oratorio", "sonata"],
        "functional tonality, basso continuo, major/minor system",
        "polyphonic (fugue) and homophonic, terraced dynamics",
        "regular metric patterns, dance rhythms",
        "terraced dynamics, echo effects",
        ["opera", "equal temperament", "orchestral standardization", "basso continuo"],
    ),
    MusicalPeriod.CLASSICAL: PeriodCharacteristics(
        MusicalPeriod.CLASSICAL, (1750, 1820),
        ["sonata", "symphony", "string quartet", "concerto", "opera buffa"],
        "diatonic tonality, Alberti bass, cadential formulas",
        "homophonic predominant, melody-accompaniment",
        "regular phrase structure, periodic rhythm",
        "gradual crescendo/decrescendo, Mannheim effects",
        ["sonata form", "public concerts", "piano as primary instrument"],
    ),
    MusicalPeriod.ROMANTIC: PeriodCharacteristics(
        MusicalPeriod.ROMANTIC, (1820, 1900),
        ["symphonic poem", "lied", "character piece", "grand opera", "concerto"],
        "chromatic harmony, distant modulations, augmented/diminished chords",
        "rich homophony, orchestral color",
        "rubato, complex meters, polyrhythm",
        "extreme dynamic range, sforzando, ppp to fff",
        ["programmatic music", "leitmotif", "expanded orchestra", "nationalism"],
    ),
    MusicalPeriod.IMPRESSIONIST: PeriodCharacteristics(
        MusicalPeriod.IMPRESSIONIST, (1875, 1925),
        ["tone poem", "prelude", "orchestral suite", "ballet"],
        "whole-tone, pentatonic, quartal, parallel chords, planing",
        "layered textures, orchestral color blending",
        "fluid, asymmetric phrasing",
        "subtle, veiled, atmospheric",
        ["non-functional harmony", "orchestral timbre as primary element", "exotic scales"],
    ),
    MusicalPeriod.MODERN: PeriodCharacteristics(
        MusicalPeriod.MODERN, (1900, 1975),
        ["serial composition", "aleatoric piece", "electronic work", "ballet"],
        "atonal, twelve-tone, quartal, polytonal, cluster",
        "pointillistic to dense clusters",
        "complex meters, asymmetric, additive",
        "extreme contrasts, Klangfarbenmelodie",
        ["serialism", "electronic music", "aleatory", "microtonality", "musique concrete"],
    ),
    MusicalPeriod.POSTMODERN: PeriodCharacteristics(
        MusicalPeriod.POSTMODERN, (1960, 2000),
        ["minimalist piece", "collage", "quotation music", "new complexity"],
        "minimalist diatonic, poly-stylistic, neo-romantic",
        "repetitive patterns, layered loops",
        "steady pulse (minimalism) to hyper-complex",
        "wide range, process-based",
        ["minimalism", "spectralism", "new complexity", "quotation/collage"],
    ),
    MusicalPeriod.CONTEMPORARY: PeriodCharacteristics(
        MusicalPeriod.CONTEMPORARY, (2000, 2030),
        ["hybrid electronic-acoustic", "installation", "AI-assisted"],
        "eclectic, spectral, microtonal, AI-generated",
        "hybrid acoustic-electronic, spatial",
        "grid-based (DAW), complex polyrhythm, free",
        "full range, spatially distributed",
        ["AI composition", "spatial audio", "live coding", "network music"],
    ),
}


def classify_period(features: dict) -> list[tuple[MusicalPeriod, float]]:
    """Heuristic classification from harmonic/textural features."""
    scores = []
    tonality = features.get("tonality_strength", 0.5)
    chromaticism = features.get("chromaticism", 0.3)
    polyphony = features.get("polyphonic_density", 0.5)
    regularity = features.get("rhythmic_regularity", 0.5)

    for period, chars in PERIOD_DATABASE.items():
        score = 0.5
        if period in (MusicalPeriod.BAROQUE, MusicalPeriod.CLASSICAL):
            score += 0.3 * tonality - 0.2 * chromaticism + 0.2 * regularity
        elif period == MusicalPeriod.ROMANTIC:
            score += 0.2 * chromaticism + 0.1 * tonality
        elif period == MusicalPeriod.MODERN:
            score += 0.3 * (1 - tonality) + 0.2 * chromaticism
        elif period in (MusicalPeriod.MEDIEVAL, MusicalPeriod.RENAISSANCE):
            score += 0.3 * polyphony - 0.2 * chromaticism
        scores.append((period, max(0.0, min(1.0, score))))

    scores.sort(key=lambda x: -x[1])
    return scores


def period_distance(p1: MusicalPeriod, p2: MusicalPeriod) -> int:
    ordered = list(MusicalPeriod)
    return abs(ordered.index(p1) - ordered.index(p2))


def evolution_timeline(start: MusicalPeriod, end: MusicalPeriod) -> list[PeriodCharacteristics]:
    ordered = list(MusicalPeriod)
    i1, i2 = ordered.index(start), ordered.index(end)
    lo, hi = min(i1, i2), max(i1, i2)
    return [PERIOD_DATABASE[ordered[i]] for i in range(lo, hi + 1)]
