"""Cadence detection and generation.

Classifies cadential patterns (PAC, IAC, half, deceptive, plagal, Phrygian)
and rates cadential strength.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class CadenceType(Enum):
    PERFECT_AUTHENTIC = "PAC"
    IMPERFECT_AUTHENTIC = "IAC"
    HALF = "HC"
    DECEPTIVE = "DC"
    PLAGAL = "PC"
    PHRYGIAN = "PHR"


@dataclass
class Cadence:
    type: CadenceType
    chords: Tuple[str, str]
    position_beats: float
    key: str


ROMAN_TO_DEGREE = {
    "I": 0, "i": 0,
    "II": 2, "ii": 2,
    "III": 4, "iii": 4,
    "IV": 5, "iv": 5,
    "V": 7,
    "VI": 9, "vi": 9,
    "VII": 11, "vii": 11,
    "bII": 1,
    "bVI": 8,
    "bVII": 10,
}


def _degree(roman: str) -> int:
    base = roman.replace("7", "").replace("dim", "").replace("aug", "").replace("+", "").replace("o", "").replace("64", "").replace("6", "")
    return ROMAN_TO_DEGREE.get(base, -1)


def _is_major(roman: str) -> bool:
    base = roman[0] if roman else ""
    return base.isupper()


def detect_cadence(chord_a: str, chord_b: str, key: str = "C") -> Optional[CadenceType]:
    deg_a = _degree(chord_a)
    deg_b = _degree(chord_b)
    if deg_a == -1 or deg_b == -1:
        return None

    if deg_a == 7 and deg_b == 0:
        if _is_major(chord_b):
            return CadenceType.PERFECT_AUTHENTIC
        return CadenceType.IMPERFECT_AUTHENTIC

    if deg_a == 7 and deg_b == 9:
        return CadenceType.DECEPTIVE

    if deg_a == 5 and deg_b == 0:
        return CadenceType.PLAGAL

    if deg_a == 1 and deg_b == 7 and not _is_major(chord_a):
        return CadenceType.PHRYGIAN

    if deg_b == 7:
        return CadenceType.HALF

    return None


def generate_cadence(cadence_type: CadenceType, key: str = "C") -> Tuple[str, str]:
    mapping = {
        CadenceType.PERFECT_AUTHENTIC: ("V", "I"),
        CadenceType.IMPERFECT_AUTHENTIC: ("V", "i"),
        CadenceType.HALF: ("IV", "V"),
        CadenceType.DECEPTIVE: ("V", "vi"),
        CadenceType.PLAGAL: ("IV", "I"),
        CadenceType.PHRYGIAN: ("bII", "V"),
    }
    return mapping.get(cadence_type, ("V", "I"))


def cadential_strength(cadence: Cadence) -> float:
    """Rate cadential closure strength in [0,1]."""
    base_strength = {
        CadenceType.PERFECT_AUTHENTIC: 1.0,
        CadenceType.IMPERFECT_AUTHENTIC: 0.7,
        CadenceType.HALF: 0.4,
        CadenceType.DECEPTIVE: 0.3,
        CadenceType.PLAGAL: 0.6,
        CadenceType.PHRYGIAN: 0.5,
    }
    return base_strength.get(cadence.type, 0.0)


def find_cadences(
    chord_sequence: List[str], key: str = "C"
) -> List[Cadence]:
    """Scan a chord progression for all cadence points."""
    results: list[Cadence] = []
    for i in range(len(chord_sequence) - 1):
        ct = detect_cadence(chord_sequence[i], chord_sequence[i + 1], key)
        if ct is not None:
            results.append(Cadence(
                type=ct,
                chords=(chord_sequence[i], chord_sequence[i + 1]),
                position_beats=float(i + 1),
                key=key,
            ))
    return results


def suggest_cadence(
    context_chords: List[str],
    key: str = "C",
    strength: float = 0.8,
) -> Tuple[str, str]:
    """Suggest an appropriate closing cadence given context and desired strength."""
    if strength >= 0.9:
        return generate_cadence(CadenceType.PERFECT_AUTHENTIC, key)
    elif strength >= 0.6:
        return generate_cadence(CadenceType.PLAGAL, key)
    elif strength >= 0.4:
        return generate_cadence(CadenceType.HALF, key)
    else:
        return generate_cadence(CadenceType.DECEPTIVE, key)


def is_cadential_64(chord_a: str, chord_b: str) -> bool:
    """Detect a cadential 6/4 pattern (I64 → V)."""
    return "64" in chord_a and _degree(chord_a) == 0 and _degree(chord_b) == 7


def cadence_rate(chord_sequence: List[str], key: str = "C") -> float:
    """Fraction of chord transitions that form cadences."""
    if len(chord_sequence) < 2:
        return 0.0
    cadences = find_cadences(chord_sequence, key)
    return len(cadences) / (len(chord_sequence) - 1)
