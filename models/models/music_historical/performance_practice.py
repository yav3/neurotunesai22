"""Historical performance practice conventions by period."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from .periods import MusicalPeriod


class TuningSystem(Enum):
    PYTHAGOREAN = "pythagorean"
    MEANTONE_QUARTER = "meantone_quarter_comma"
    WERCKMEISTER_III = "werckmeister_iii"
    KIRNBERGER_III = "kirnberger_iii"
    EQUAL_TEMPERAMENT = "equal_temperament"
    JUST_INTONATION = "just_intonation"


class ArticulationStyle(Enum):
    LEGATO = "legato"
    DETACHE = "detache"
    STACCATO = "staccato"
    PORTATO = "portato"
    NOTES_INEGALES = "notes_inegales"
    MESSA_DI_VOCE = "messa_di_voce"


@dataclass(frozen=True)
class PerformancePractice:
    period: MusicalPeriod
    standard_pitch_hz: float
    tuning: TuningSystem
    vibrato_use: str
    ornamentation_density: float
    continuo_present: bool
    conductor_present: bool
    default_articulations: list[ArticulationStyle]


PERIOD_PRACTICES: dict[MusicalPeriod, PerformancePractice] = {
    MusicalPeriod.MEDIEVAL: PerformancePractice(
        MusicalPeriod.MEDIEVAL, 466.0, TuningSystem.PYTHAGOREAN,
        "none", 0.1, False, False,
        [ArticulationStyle.LEGATO],
    ),
    MusicalPeriod.RENAISSANCE: PerformancePractice(
        MusicalPeriod.RENAISSANCE, 466.0, TuningSystem.MEANTONE_QUARTER,
        "ornamental_only", 0.3, False, False,
        [ArticulationStyle.LEGATO, ArticulationStyle.PORTATO],
    ),
    MusicalPeriod.BAROQUE: PerformancePractice(
        MusicalPeriod.BAROQUE, 415.0, TuningSystem.WERCKMEISTER_III,
        "ornamental", 0.7, True, False,
        [ArticulationStyle.DETACHE, ArticulationStyle.NOTES_INEGALES, ArticulationStyle.MESSA_DI_VOCE],
    ),
    MusicalPeriod.CLASSICAL: PerformancePractice(
        MusicalPeriod.CLASSICAL, 430.0, TuningSystem.KIRNBERGER_III,
        "sparing", 0.5, False, True,
        [ArticulationStyle.LEGATO, ArticulationStyle.STACCATO, ArticulationStyle.PORTATO],
    ),
    MusicalPeriod.ROMANTIC: PerformancePractice(
        MusicalPeriod.ROMANTIC, 440.0, TuningSystem.EQUAL_TEMPERAMENT,
        "continuous", 0.4, False, True,
        [ArticulationStyle.LEGATO, ArticulationStyle.PORTATO, ArticulationStyle.MESSA_DI_VOCE],
    ),
    MusicalPeriod.IMPRESSIONIST: PerformancePractice(
        MusicalPeriod.IMPRESSIONIST, 440.0, TuningSystem.EQUAL_TEMPERAMENT,
        "subtle_continuous", 0.3, False, True,
        [ArticulationStyle.LEGATO, ArticulationStyle.PORTATO],
    ),
    MusicalPeriod.MODERN: PerformancePractice(
        MusicalPeriod.MODERN, 440.0, TuningSystem.EQUAL_TEMPERAMENT,
        "varied", 0.5, False, True,
        [ArticulationStyle.LEGATO, ArticulationStyle.STACCATO, ArticulationStyle.DETACHE],
    ),
    MusicalPeriod.CONTEMPORARY: PerformancePractice(
        MusicalPeriod.CONTEMPORARY, 442.0, TuningSystem.EQUAL_TEMPERAMENT,
        "as_specified", 0.6, False, True,
        [ArticulationStyle.LEGATO, ArticulationStyle.STACCATO, ArticulationStyle.PORTATO],
    ),
    MusicalPeriod.AVANT_GARDE: PerformancePractice(
        MusicalPeriod.AVANT_GARDE, 440.0, TuningSystem.EQUAL_TEMPERAMENT,
        "extended_techniques", 0.8, False, True,
        [],
    ),
}


_PITCH_STANDARD_TIMELINE = [
    (1500, 466.0), (1600, 415.0), (1700, 415.0),
    (1750, 422.0), (1780, 430.0), (1834, 435.0),
    (1858, 435.4), (1939, 440.0), (1975, 442.0),
]


def reference_pitch_for_year(year: int) -> float:
    if year < _PITCH_STANDARD_TIMELINE[0][0]:
        return _PITCH_STANDARD_TIMELINE[0][1]
    for i in range(len(_PITCH_STANDARD_TIMELINE) - 1):
        y0, p0 = _PITCH_STANDARD_TIMELINE[i]
        y1, p1 = _PITCH_STANDARD_TIMELINE[i + 1]
        if y0 <= year < y1:
            frac = (year - y0) / (y1 - y0)
            return p0 + frac * (p1 - p0)
    return _PITCH_STANDARD_TIMELINE[-1][1]


def tuning_deviation_cents(tuning: TuningSystem, interval_semitones: int) -> float:
    if tuning == TuningSystem.EQUAL_TEMPERAMENT:
        return 0.0
    if tuning == TuningSystem.PYTHAGOREAN:
        pyth_cents = {0: 0, 1: 90.22, 2: 203.91, 3: 294.13, 4: 407.82,
                      5: 498.04, 6: 611.73, 7: 701.96, 8: 792.18,
                      9: 905.87, 10: 996.09, 11: 1109.78, 12: 1200.0}
        et = interval_semitones * 100.0
        return pyth_cents.get(interval_semitones % 12, et) - (interval_semitones % 12) * 100.0
    if tuning == TuningSystem.MEANTONE_QUARTER:
        mt_cents = {0: 0, 1: 76.0, 2: 193.2, 3: 310.3, 4: 386.3,
                    5: 503.4, 6: 579.5, 7: 696.6, 8: 772.6,
                    9: 889.7, 10: 1006.8, 11: 1082.9, 12: 1200.0}
        return mt_cents.get(interval_semitones % 12, 0.0) - (interval_semitones % 12) * 100.0
    return 0.0


def ornamentation_density(period: MusicalPeriod) -> float:
    pp = PERIOD_PRACTICES.get(period)
    return pp.ornamentation_density if pp else 0.3
