"""Harmonic analysis: Roman-numeral analysis, pivot-chord modulation, cadence detection.

Computational harmonic analysis: chord-to-Roman-numeral mapping, secondary
dominant detection, pivot-chord modulation identification, cadence
classification, harmonic rhythm extraction, and prolongational analysis.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CadenceType(str, Enum):
    PERFECT_AUTHENTIC = "PAC"
    IMPERFECT_AUTHENTIC = "IAC"
    HALF = "HC"
    DECEPTIVE = "DC"
    PLAGAL = "plagal"
    PHRYGIAN_HALF = "phrygian_half"


class ChordFunction(str, Enum):
    TONIC = "T"
    PREDOMINANT = "PD"
    DOMINANT = "D"
    SECONDARY_DOMINANT = "V/x"
    CHROMATIC = "Chr"
    PASSING = "Pass"


DIATONIC_MAJOR = {
    0: ("I", ChordFunction.TONIC),
    2: ("ii", ChordFunction.PREDOMINANT),
    4: ("iii", ChordFunction.TONIC),
    5: ("IV", ChordFunction.PREDOMINANT),
    7: ("V", ChordFunction.DOMINANT),
    9: ("vi", ChordFunction.TONIC),
    11: ("viio", ChordFunction.DOMINANT),
}

DIATONIC_MINOR = {
    0: ("i", ChordFunction.TONIC),
    2: ("iio", ChordFunction.PREDOMINANT),
    3: ("III", ChordFunction.TONIC),
    5: ("iv", ChordFunction.PREDOMINANT),
    7: ("V", ChordFunction.DOMINANT),
    8: ("VI", ChordFunction.TONIC),
    10: ("VII", ChordFunction.DOMINANT),
}


@dataclass
class RomanNumeral:
    numeral: str
    function: ChordFunction
    key: int
    quality: str
    is_secondary: bool = False
    secondary_target: str = ""


@dataclass
class HarmonicSegment:
    start_beat: float
    end_beat: float
    chord_root: int
    quality: str
    roman: RomanNumeral
    inversion: int = 0


def root_to_roman(root_pc: int, key_pc: int,
                  is_minor: bool = False) -> RomanNumeral:
    interval = (root_pc - key_pc) % 12
    table = DIATONIC_MINOR if is_minor else DIATONIC_MAJOR
    if interval in table:
        numeral, func = table[interval]
        quality = "minor" if numeral[0].islower() else "major"
        return RomanNumeral(numeral, func, key_pc, quality)
    return RomanNumeral(f"#{interval}", ChordFunction.CHROMATIC, key_pc, "chromatic")


def detect_secondary_dominant(root_pc: int, next_root_pc: int,
                              key_pc: int) -> Optional[str]:
    interval_to_next = (next_root_pc - root_pc) % 12
    if interval_to_next == 7 or interval_to_next == 5:
        target_interval = (next_root_pc - key_pc) % 12
        table = DIATONIC_MAJOR
        if target_interval in table:
            target_numeral = table[target_interval][0]
            return f"V/{target_numeral}"
    return None


def classify_cadence(penultimate_root: int, final_root: int,
                     key_pc: int, final_in_root_pos: bool = True,
                     is_minor: bool = False) -> Optional[CadenceType]:
    pen_iv = (penultimate_root - key_pc) % 12
    fin_iv = (final_root - key_pc) % 12
    if pen_iv == 7 and fin_iv == 0:
        return CadenceType.PERFECT_AUTHENTIC if final_in_root_pos else CadenceType.IMPERFECT_AUTHENTIC
    if pen_iv == 7 and fin_iv != 0:
        if fin_iv == 9 or (is_minor and fin_iv == 8):
            return CadenceType.DECEPTIVE
    if fin_iv == 7:
        if is_minor and pen_iv == 8:
            return CadenceType.PHRYGIAN_HALF
        return CadenceType.HALF
    if pen_iv == 5 and fin_iv == 0:
        return CadenceType.PLAGAL
    return None


def harmonic_rhythm(chord_onsets_beats: np.ndarray) -> np.ndarray:
    if len(chord_onsets_beats) < 2:
        return np.array([4.0])
    return np.diff(chord_onsets_beats)


def harmonic_rhythm_regularity(durations_beats: np.ndarray) -> float:
    if len(durations_beats) < 2:
        return 1.0
    mu = durations_beats.mean()
    if mu < 1e-6:
        return 0.0
    return 1.0 - float(durations_beats.std() / mu)


def analyze_progression(roots_pc: np.ndarray, key_pc: int,
                        is_minor: bool = False) -> List[RomanNumeral]:
    result: List[RomanNumeral] = []
    for i, root in enumerate(roots_pc):
        rn = root_to_roman(int(root) % 12, key_pc, is_minor)
        if i < len(roots_pc) - 1:
            sec = detect_secondary_dominant(int(root) % 12,
                                           int(roots_pc[i + 1]) % 12, key_pc)
            if sec is not None:
                rn = RomanNumeral(sec, ChordFunction.SECONDARY_DOMINANT,
                                  key_pc, "major", True, sec.split("/")[1])
        result.append(rn)
    return result


def find_cadences(roots_pc: np.ndarray, key_pc: int,
                  is_minor: bool = False) -> List[Tuple[int, CadenceType]]:
    cadences: List[Tuple[int, CadenceType]] = []
    for i in range(1, len(roots_pc)):
        cad = classify_cadence(int(roots_pc[i - 1]) % 12,
                               int(roots_pc[i]) % 12,
                               key_pc, True, is_minor)
        if cad is not None:
            cadences.append((i, cad))
    return cadences


def pivot_chord_modulation(roots_pc: np.ndarray, key1: int, key2: int,
                           is_minor: bool = False) -> List[Tuple[int, str, str]]:
    pivots: List[Tuple[int, str, str]] = []
    for i, root in enumerate(roots_pc):
        rn1 = root_to_roman(int(root) % 12, key1, is_minor)
        rn2 = root_to_roman(int(root) % 12, key2, is_minor)
        if (rn1.function != ChordFunction.CHROMATIC and
                rn2.function != ChordFunction.CHROMATIC):
            pivots.append((i, rn1.numeral, rn2.numeral))
    return pivots


def prolongational_reduction(roots_pc: np.ndarray,
                             durations_beats: np.ndarray,
                             key_pc: int) -> np.ndarray:
    importance = np.zeros(len(roots_pc))
    for i, root in enumerate(roots_pc):
        iv = (int(root) - key_pc) % 12
        if iv == 0:
            importance[i] = 3.0
        elif iv == 7:
            importance[i] = 2.0
        elif iv in (5, 2, 9):
            importance[i] = 1.0
        else:
            importance[i] = 0.5
    if len(durations_beats) == len(importance):
        dur_weight = durations_beats / max(durations_beats.max(), 1e-6)
        importance *= (1.0 + dur_weight)
    return importance


def functional_distance(fn1: ChordFunction, fn2: ChordFunction) -> float:
    order = [ChordFunction.TONIC, ChordFunction.PREDOMINANT,
             ChordFunction.DOMINANT, ChordFunction.TONIC]
    idx = {ChordFunction.TONIC: 0, ChordFunction.PREDOMINANT: 1,
           ChordFunction.DOMINANT: 2, ChordFunction.SECONDARY_DOMINANT: 2.5,
           ChordFunction.CHROMATIC: 1.5, ChordFunction.PASSING: 0.5}
    return abs(idx.get(fn1, 0) - idx.get(fn2, 0))
