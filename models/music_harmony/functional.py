"""Functional harmony: tonic/subdominant/dominant classification, cadence detection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from models.music_harmony.chords import Chord, ChordQuality


class HarmonicFunction(Enum):
    TONIC = "tonic"
    PREDOMINANT = "predominant"
    DOMINANT = "dominant"
    CHROMATIC = "chromatic"


FUNCTION_MAP_MAJOR: dict[int, HarmonicFunction] = {
    0: HarmonicFunction.TONIC,         # I
    2: HarmonicFunction.PREDOMINANT,    # ii
    4: HarmonicFunction.TONIC,          # iii
    5: HarmonicFunction.PREDOMINANT,    # IV
    7: HarmonicFunction.DOMINANT,       # V
    9: HarmonicFunction.TONIC,          # vi
    11: HarmonicFunction.DOMINANT,      # vii°
}

FUNCTION_MAP_MINOR: dict[int, HarmonicFunction] = {
    0: HarmonicFunction.TONIC,          # i
    2: HarmonicFunction.PREDOMINANT,    # ii°
    3: HarmonicFunction.TONIC,          # III
    5: HarmonicFunction.PREDOMINANT,    # iv
    7: HarmonicFunction.DOMINANT,       # v / V
    8: HarmonicFunction.PREDOMINANT,    # VI
    10: HarmonicFunction.DOMINANT,      # VII
}


def classify_function(
    chord: Chord, key_root: int = 0, mode: str = "major",
) -> HarmonicFunction:
    degree = (chord.root - key_root) % 12
    table = FUNCTION_MAP_MAJOR if mode == "major" else FUNCTION_MAP_MINOR
    return table.get(degree, HarmonicFunction.CHROMATIC)


CADENCE_TYPES: dict[str, str] = {
    "V-I": "authentic",
    "V7-I": "authentic",
    "vii°-I": "authentic",
    "IV-I": "plagal",
    "iv-I": "plagal",
    "iv-i": "plagal",
    "V-vi": "deceptive",
    "V-VI": "deceptive",
    "V7-vi": "deceptive",
    "I-V": "half",
    "ii-V": "half",
    "IV-V": "half",
    "vi-V": "half",
}


def cadence_type(
    penultimate: Chord, final: Chord,
    key_root: int = 0, mode: str = "major",
) -> str | None:
    from models.music_harmony.progressions import SCALE_DEGREES_MAJOR, SCALE_DEGREES_MINOR
    table = SCALE_DEGREES_MAJOR if mode == "major" else SCALE_DEGREES_MINOR
    deg_p = (penultimate.root - key_root) % 12
    deg_f = (final.root - key_root) % 12
    num_p = table.get(deg_p, (None, f"?{deg_p}"))[1]
    num_f = table.get(deg_f, (None, f"?{deg_f}"))[1]
    if penultimate.quality == ChordQuality.DOMINANT7:
        key_7 = f"{num_p}7-{num_f}"
        if key_7 in CADENCE_TYPES:
            return CADENCE_TYPES[key_7]
    key_plain = f"{num_p}-{num_f}"
    return CADENCE_TYPES.get(key_plain)


def tonicization_target(
    chord: Chord, next_chord: Chord,
) -> int | None:
    if chord.quality not in (ChordQuality.DOMINANT7, ChordQuality.MAJOR):
        return None
    expected_resolution = (chord.root + 5) % 12
    if next_chord.root == expected_resolution:
        return next_chord.root
    return None
