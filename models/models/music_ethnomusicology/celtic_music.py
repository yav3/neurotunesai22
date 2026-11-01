"""Celtic music theory: modes, ornamentation, tune types, session structure."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np


class CelticMode(Enum):
    IONIAN = "ionian"
    DORIAN = "dorian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"


CELTIC_MODE_INTERVALS: dict[CelticMode, tuple[int, ...]] = {
    CelticMode.IONIAN:     (0, 2, 4, 5, 7, 9, 11),
    CelticMode.DORIAN:     (0, 2, 3, 5, 7, 9, 10),
    CelticMode.MIXOLYDIAN: (0, 2, 4, 5, 7, 9, 10),
    CelticMode.AEOLIAN:    (0, 2, 3, 5, 7, 8, 10),
}


class TuneType(Enum):
    REEL = "reel"
    JIG = "jig"
    SLIP_JIG = "slip_jig"
    HORNPIPE = "hornpipe"
    POLKA = "polka"
    SLOW_AIR = "slow_air"
    MARCH = "march"
    WALTZ = "waltz"


TUNE_METER: dict[TuneType, tuple[int, int]] = {
    TuneType.REEL: (4, 4),
    TuneType.JIG: (6, 8),
    TuneType.SLIP_JIG: (9, 8),
    TuneType.HORNPIPE: (4, 4),
    TuneType.POLKA: (2, 4),
    TuneType.SLOW_AIR: (4, 4),
    TuneType.MARCH: (4, 4),
    TuneType.WALTZ: (3, 4),
}

TUNE_TEMPO_RANGE: dict[TuneType, tuple[int, int]] = {
    TuneType.REEL: (100, 140),
    TuneType.JIG: (100, 132),
    TuneType.SLIP_JIG: (100, 132),
    TuneType.HORNPIPE: (72, 100),
    TuneType.POLKA: (120, 160),
    TuneType.SLOW_AIR: (40, 72),
    TuneType.MARCH: (80, 120),
    TuneType.WALTZ: (90, 130),
}


class OrnamentType(Enum):
    CUT = "cut"
    TAP = "tap"
    ROLL = "roll"
    CRAN = "cran"
    TRIPLET = "triplet"
    SLIDE = "slide"


@dataclass(frozen=True)
class Ornament:
    ornament_type: OrnamentType
    main_note: int
    auxiliary_notes: tuple[int, ...]
    duration_fraction: float = 0.15


def classify_tune_type(time_sig: tuple[int, int], tempo_bpm: float,
                       swing_ratio: float = 1.0) -> TuneType:
    num, denom = time_sig
    if num == 6 and denom == 8:
        return TuneType.JIG
    if num == 9 and denom == 8:
        return TuneType.SLIP_JIG
    if num == 3:
        return TuneType.WALTZ
    if num == 2 and denom == 4:
        return TuneType.POLKA
    if swing_ratio > 1.3 and 72 <= tempo_bpm <= 100:
        return TuneType.HORNPIPE
    if tempo_bpm < 72:
        return TuneType.SLOW_AIR
    return TuneType.REEL


def classify_celtic_mode(pitches: list[int], tonic: int = 0) -> CelticMode:
    pcs = sorted(set((p - tonic) % 12 for p in pitches))
    best_mode = CelticMode.IONIAN
    best_score = -1
    for mode, intervals in CELTIC_MODE_INTERVALS.items():
        score = len(set(pcs) & set(intervals))
        if score > best_score:
            best_score = score
            best_mode = mode
    return best_mode


def generate_ornament(note: int, ornament_type: OrnamentType,
                      scale_degrees: tuple[int, ...] = ()) -> Ornament:
    if ornament_type == OrnamentType.CUT:
        aux = (note + 2,) if not scale_degrees else (scale_degrees[min(scale_degrees.index(note) + 1, len(scale_degrees) - 1)],) if note in scale_degrees else (note + 1,)
        return Ornament(OrnamentType.CUT, note, aux, 0.1)
    if ornament_type == OrnamentType.TAP:
        aux = (note - 2,) if not scale_degrees else (scale_degrees[max(scale_degrees.index(note) - 1, 0)],) if note in scale_degrees else (note - 1,)
        return Ornament(OrnamentType.TAP, note, aux, 0.1)
    if ornament_type == OrnamentType.ROLL:
        return Ornament(OrnamentType.ROLL, note, (note + 1, note - 1), 0.25)
    if ornament_type == OrnamentType.CRAN:
        return Ornament(OrnamentType.CRAN, note, (note + 2, note + 1, note + 2), 0.3)
    if ornament_type == OrnamentType.TRIPLET:
        return Ornament(OrnamentType.TRIPLET, note, (note, note), 0.33)
    return Ornament(OrnamentType.SLIDE, note, (note - 1,), 0.15)


def session_set_order(tune_types: list[TuneType]) -> list[TuneType]:
    priority = {TuneType.SLOW_AIR: 0, TuneType.MARCH: 1, TuneType.WALTZ: 2,
                TuneType.JIG: 3, TuneType.SLIP_JIG: 4, TuneType.POLKA: 5,
                TuneType.HORNPIPE: 6, TuneType.REEL: 7}
    return sorted(tune_types, key=lambda t: priority.get(t, 99))


def tune_key_histogram(pitches: list[int]) -> np.ndarray:
    hist = np.zeros(12)
    for p in pitches:
        hist[p % 12] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist
