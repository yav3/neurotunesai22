"""Chord progression analysis: Roman numerals, tension curves, complexity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from models.music_harmony.chords import Chord, ChordQuality, CHORD_TYPES


SCALE_DEGREES_MAJOR = {
    0: (ChordQuality.MAJOR, "I"),
    2: (ChordQuality.MINOR, "ii"),
    4: (ChordQuality.MINOR, "iii"),
    5: (ChordQuality.MAJOR, "IV"),
    7: (ChordQuality.MAJOR, "V"),
    9: (ChordQuality.MINOR, "vi"),
    11: (ChordQuality.DIMINISHED, "vii°"),
}

SCALE_DEGREES_MINOR = {
    0: (ChordQuality.MINOR, "i"),
    2: (ChordQuality.DIMINISHED, "ii°"),
    3: (ChordQuality.MAJOR, "III"),
    5: (ChordQuality.MINOR, "iv"),
    7: (ChordQuality.MINOR, "v"),
    8: (ChordQuality.MAJOR, "VI"),
    10: (ChordQuality.MAJOR, "VII"),
}

TENSION_WEIGHTS: dict[str, float] = {
    "I": 0.0, "i": 0.0,
    "ii": 0.4, "ii°": 0.5,
    "iii": 0.2, "III": 0.2,
    "IV": 0.3, "iv": 0.35,
    "V": 0.7, "v": 0.5,
    "vi": 0.15, "VI": 0.15,
    "vii°": 0.85, "VII": 0.6,
    "V7": 0.9, "viidim7": 0.95,
}


@dataclass(frozen=True)
class RomanNumeral:
    degree: int
    numeral: str
    quality: ChordQuality
    is_secondary: bool = False
    secondary_target: str | None = None


@dataclass
class ChordProgression:
    chords: list[Chord]
    key_root: int = 0
    mode: str = "major"

    @property
    def length(self) -> int:
        return len(self.chords)

    def transposed(self, semitones: int) -> "ChordProgression":
        new_chords = [
            Chord(root=(c.root + semitones) % 12, quality=c.quality,
                  bass=(c.bass + semitones) % 12 if c.bass is not None else None,
                  extensions=c.extensions)
            for c in self.chords
        ]
        return ChordProgression(new_chords, (self.key_root + semitones) % 12, self.mode)


COMMON_PROGRESSIONS: dict[str, list[str]] = {
    "I-IV-V-I": ["I", "IV", "V", "I"],
    "I-V-vi-IV": ["I", "V", "vi", "IV"],
    "ii-V-I": ["ii", "V", "I"],
    "I-vi-IV-V": ["I", "vi", "IV", "V"],
    "I-IV-vi-V": ["I", "IV", "vi", "V"],
    "vi-IV-I-V": ["vi", "IV", "I", "V"],
    "i-iv-v-i": ["i", "iv", "v", "i"],
    "i-VI-III-VII": ["i", "VI", "III", "VII"],
    "i-iv-VII-III": ["i", "iv", "VII", "III"],
    "12-bar-blues": ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"],
}


def analyze_roman_numerals(
    chords: Sequence[Chord], key_root: int = 0, mode: str = "major",
) -> list[RomanNumeral]:
    table = SCALE_DEGREES_MAJOR if mode == "major" else SCALE_DEGREES_MINOR
    result: list[RomanNumeral] = []
    for chord in chords:
        degree = (chord.root - key_root) % 12
        if degree in table:
            expected_q, numeral = table[degree]
            if chord.quality == ChordQuality.DOMINANT7 and numeral in ("V", "v"):
                numeral = "V7"
            result.append(RomanNumeral(degree, numeral, chord.quality))
        else:
            result.append(RomanNumeral(degree, f"?{degree}", chord.quality))
    return result


def progression_complexity(numerals: Sequence[RomanNumeral]) -> float:
    if len(numerals) <= 1:
        return 0.0
    unique_degrees = len({rn.degree for rn in numerals})
    degree_ratio = unique_degrees / 7.0
    non_diatonic = sum(1 for rn in numerals if rn.numeral.startswith("?"))
    chromatic_ratio = non_diatonic / len(numerals)
    return min(1.0, 0.5 * degree_ratio + 0.5 * chromatic_ratio)


def progression_tension_curve(numerals: Sequence[RomanNumeral]) -> list[float]:
    return [TENSION_WEIGHTS.get(rn.numeral, 0.5) for rn in numerals]
