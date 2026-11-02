"""Functional harmony analysis: Roman numeral analysis, tonicization, cadence detection.

Analyse chord sequences in terms of tonal function (T/S/D), detect
tonicizations and modulations, identify cadence types, and compute
harmonic tension curves. References Aldwell & Schachter (2011) and
Kostka & Payne (2018). Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple


class HarmonicFunction(Enum):
    TONIC = "T"
    SUBDOMINANT = "S"
    DOMINANT = "D"
    PREDOMINANT = "PD"
    CHROMATIC = "CHR"


class CadenceType(Enum):
    PERFECT_AUTHENTIC = "PAC"
    IMPERFECT_AUTHENTIC = "IAC"
    HALF = "HC"
    DECEPTIVE = "DC"
    PLAGAL = "PL"
    PHRYGIAN = "PHR"
    NONE = "NONE"


@dataclass
class RomanNumeral:
    degree: int
    quality: str
    inversion: int = 0
    secondary: Optional[int] = None
    applied: bool = False

    def __str__(self) -> str:
        numerals = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}
        base = numerals.get(self.degree, str(self.degree))
        if self.quality in ("minor", "diminished"):
            base = base.lower()
        suffix = ""
        if self.quality == "diminished":
            suffix = "°"
        elif self.quality == "augmented":
            suffix = "+"
        elif self.quality == "dominant7":
            suffix = "7"
        elif self.quality == "half_diminished":
            suffix = "ø7"
        inv_label = ["", "6", "64", "7"][min(self.inversion, 3)]
        sec = ""
        if self.applied and self.secondary is not None:
            sec_num = numerals.get(self.secondary, str(self.secondary))
            sec = f"/{sec_num}"
        return f"{base}{suffix}{inv_label}{sec}"


FUNCTION_MAP: Dict[int, HarmonicFunction] = {
    1: HarmonicFunction.TONIC,
    2: HarmonicFunction.PREDOMINANT,
    3: HarmonicFunction.TONIC,
    4: HarmonicFunction.SUBDOMINANT,
    5: HarmonicFunction.DOMINANT,
    6: HarmonicFunction.TONIC,
    7: HarmonicFunction.DOMINANT,
}


@dataclass
class FunctionalAnalysis:
    roman_numerals: List[RomanNumeral]
    functions: List[HarmonicFunction]
    cadences: List[Tuple[int, CadenceType]]
    tension_curve: np.ndarray
    tonicizations: List[Tuple[int, int, int]]


def pitch_class_set_to_chord_quality(pcs: Set[int], root: int) -> str:
    """Determine chord quality from pitch class set relative to root."""
    intervals = sorted((pc - root) % 12 for pc in pcs)
    if 4 in intervals and 7 in intervals:
        if 10 in intervals:
            return "dominant7"
        if 11 in intervals:
            return "major7"
        return "major"
    if 3 in intervals and 7 in intervals:
        if 10 in intervals:
            return "minor7"
        return "minor"
    if 3 in intervals and 6 in intervals:
        if 10 in intervals:
            return "half_diminished"
        if 9 in intervals:
            return "diminished7"
        return "diminished"
    if 4 in intervals and 8 in intervals:
        return "augmented"
    return "unknown"


def chord_to_roman_numeral(root_pc: int, quality: str,
                           key_root: int, key_mode: str = "major",
                           bass_pc: Optional[int] = None) -> RomanNumeral:
    """Convert a chord root + quality to a Roman numeral in the given key."""
    if key_mode == "major":
        scale_pcs = [(key_root + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11]]
    else:
        scale_pcs = [(key_root + s) % 12 for s in [0, 2, 3, 5, 7, 8, 10]]

    degree = None
    for i, pc in enumerate(scale_pcs):
        if pc == root_pc:
            degree = i + 1
            break

    if degree is None:
        degree_est = round(((root_pc - key_root) % 12) / 2) + 1
        degree = max(1, min(7, degree_est))

    inversion = 0
    if bass_pc is not None and bass_pc != root_pc:
        chord_pcs = _chord_pcs(root_pc, quality)
        for inv, pc in enumerate(chord_pcs[1:], 1):
            if pc == bass_pc:
                inversion = inv
                break

    return RomanNumeral(degree=degree, quality=quality, inversion=inversion)


def _chord_pcs(root: int, quality: str) -> List[int]:
    if quality in ("major", "dominant7", "major7"):
        base = [0, 4, 7]
    elif quality in ("minor", "minor7"):
        base = [0, 3, 7]
    elif quality in ("diminished", "diminished7", "half_diminished"):
        base = [0, 3, 6]
    elif quality == "augmented":
        base = [0, 4, 8]
    else:
        base = [0, 4, 7]
    return [(root + i) % 12 for i in base]


def detect_cadence(rn_prev: RomanNumeral, rn_curr: RomanNumeral,
                   soprano_on_tonic: bool = False) -> CadenceType:
    """Detect cadence type from two consecutive Roman numerals."""
    if rn_prev.degree == 5 and rn_curr.degree == 1:
        if rn_curr.inversion == 0 and soprano_on_tonic:
            return CadenceType.PERFECT_AUTHENTIC
        return CadenceType.IMPERFECT_AUTHENTIC
    if rn_curr.degree == 5:
        return CadenceType.HALF
    if rn_prev.degree == 5 and rn_curr.degree == 6:
        return CadenceType.DECEPTIVE
    if rn_prev.degree == 4 and rn_curr.degree == 1:
        return CadenceType.PLAGAL
    if rn_prev.degree == 4 and rn_curr.degree == 5:
        if rn_prev.quality == "minor":
            return CadenceType.PHRYGIAN
    return CadenceType.NONE


def harmonic_tension(rn: RomanNumeral) -> float:
    """Assign a tension value to a Roman numeral chord.

    Scale: 0.0 (stable tonic) to 1.0 (maximum tension).
    """
    base_tension = {
        1: 0.0,
        2: 0.4,
        3: 0.2,
        4: 0.5,
        5: 0.7,
        6: 0.3,
        7: 0.8,
    }
    t = base_tension.get(rn.degree, 0.5)
    if rn.quality == "dominant7":
        t = min(1.0, t + 0.15)
    if rn.quality in ("diminished", "diminished7"):
        t = min(1.0, t + 0.1)
    if rn.quality == "augmented":
        t = min(1.0, t + 0.12)
    if rn.inversion > 0:
        t = min(1.0, t + 0.05 * rn.inversion)
    if rn.applied:
        t = min(1.0, t + 0.1)
    return t


def detect_tonicization(roman_numerals: Sequence[RomanNumeral],
                        window: int = 3) -> List[Tuple[int, int, int]]:
    """Detect tonicizations: V→I or V7→I patterns targeting non-tonic degrees.

    Returns list of (start_index, end_index, tonicized_degree).
    """
    tonicizations = []
    for i in range(len(roman_numerals) - 1):
        rn_a = roman_numerals[i]
        rn_b = roman_numerals[i + 1]
        if (rn_a.degree == 5 and rn_a.quality in ("major", "dominant7") and
                rn_b.degree != 1 and rn_b.quality in ("major", "minor")):
            tonicizations.append((i, i + 1, rn_b.degree))
    return tonicizations


def tension_curve(roman_numerals: Sequence[RomanNumeral]) -> np.ndarray:
    """Compute tension trajectory from Roman numeral sequence."""
    return np.array([harmonic_tension(rn) for rn in roman_numerals])


def functional_rhythm(functions: Sequence[HarmonicFunction]) -> np.ndarray:
    """Encode functional rhythm as a numeric sequence for analysis.

    T=0, S/PD=0.5, D=1.0, CHR=0.75.
    """
    mapping = {
        HarmonicFunction.TONIC: 0.0,
        HarmonicFunction.SUBDOMINANT: 0.5,
        HarmonicFunction.PREDOMINANT: 0.5,
        HarmonicFunction.DOMINANT: 1.0,
        HarmonicFunction.CHROMATIC: 0.75,
    }
    return np.array([mapping.get(f, 0.5) for f in functions])


def analyze_progression(
    roots: Sequence[int],
    qualities: Sequence[str],
    key_root: int = 0,
    key_mode: str = "major",
) -> FunctionalAnalysis:
    """Full functional analysis of a chord progression."""
    rns = [chord_to_roman_numeral(r, q, key_root, key_mode)
           for r, q in zip(roots, qualities)]
    funcs = [FUNCTION_MAP.get(rn.degree, HarmonicFunction.CHROMATIC) for rn in rns]
    cadences = []
    for i in range(1, len(rns)):
        cad = detect_cadence(rns[i - 1], rns[i])
        if cad != CadenceType.NONE:
            cadences.append((i, cad))
    tc = tension_curve(rns)
    tonicizations = detect_tonicization(rns)

    return FunctionalAnalysis(
        roman_numerals=rns,
        functions=funcs,
        cadences=cadences,
        tension_curve=tc,
        tonicizations=tonicizations,
    )


def harmonic_surprise(tension_values: np.ndarray) -> np.ndarray:
    """Information-theoretic surprise: how unexpected each tension value is."""
    if len(tension_values) < 2:
        return np.zeros_like(tension_values)
    surprise = np.zeros_like(tension_values)
    for i in range(1, len(tension_values)):
        context_mean = np.mean(tension_values[:i])
        surprise[i] = abs(tension_values[i] - context_mean)
    return surprise
