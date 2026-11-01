from __future__ import annotations

from dataclasses import dataclass

from models.music_theory.scales import CHROMATIC_NOTES, SCALE_INTERVALS, ENHARMONIC_MAP
from models.music_theory.chords import Chord, ChordLibrary


NUMERAL_DEGREE: dict[str, int] = {
    "I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6,
    "i": 0, "ii": 1, "iii": 2, "iv": 3, "v": 4, "vi": 5, "vii": 6,
}

MAJOR_SCALE_QUALITIES: tuple[str, ...] = (
    "major", "minor", "minor", "major", "major", "minor", "dim",
)


def _note_index(note: str) -> int:
    if note in ENHARMONIC_MAP and note not in CHROMATIC_NOTES:
        note = ENHARMONIC_MAP[note]
    return CHROMATIC_NOTES.index(note)


@dataclass(frozen=True)
class RomanNumeral:
    numeral: str
    chord_quality: str
    scale_degree: int


@dataclass(frozen=True)
class Progression:
    key: str
    roman_numerals: tuple[RomanNumeral, ...]
    chords: tuple[Chord, ...]


COMMON_PROGRESSIONS: dict[str, list[str]] = {
    "I-V-vi-IV":        ["I", "V", "vi", "IV"],
    "I-IV-V":           ["I", "IV", "V"],
    "I-IV-V-I":         ["I", "IV", "V", "I"],
    "ii-V-I":           ["ii", "V", "I"],
    "I-vi-IV-V":        ["I", "vi", "IV", "V"],
    "vi-IV-I-V":        ["vi", "IV", "I", "V"],
    "I-V-vi-iii-IV":    ["I", "V", "vi", "iii", "IV"],
    "I-IV-vi-V":        ["I", "IV", "vi", "V"],
    "I-iii-IV-V":       ["I", "iii", "IV", "V"],
    "I-V-IV":           ["I", "V", "IV"],
    "I-bVII-IV":        ["I", "bVII", "IV"],
    "i-bVI-bIII-bVII":  ["i", "bVI", "bIII", "bVII"],
    "i-iv-v":           ["i", "iv", "v"],
    "i-bVII-bVI-V":     ["i", "bVII", "bVI", "V"],
    "12-bar-blues":     ["I", "I", "I", "I", "IV", "IV", "I", "I",
                         "V", "IV", "I", "V"],
    "pachelbel":        ["I", "V", "vi", "iii", "IV", "I", "IV", "V"],
    "andalusian":       ["i", "bVII", "bVI", "V"],
    "rhythm-changes-A": ["I", "vi", "ii", "V"],
}

CAMELOT_MAP: dict[str, str] = {
    "C major": "8B",  "A minor": "8A",
    "G major": "9B",  "E minor": "9A",
    "D major": "10B", "B minor": "10A",
    "A major": "11B", "F# minor": "11A",
    "E major": "12B", "C# minor": "12A",
    "B major": "1B",  "G# minor": "1A",
    "F# major": "2B", "D# minor": "2A",
    "C# major": "3B", "A# minor": "3A",
    "Ab major": "4B", "F minor": "4A",
    "Eb major": "5B", "C minor": "5A",
    "Bb major": "6B", "G minor": "6A",
    "F major": "7B",  "D minor": "7A",
}


def _parse_numeral(numeral_str: str) -> tuple[int, str, bool]:
    s = numeral_str
    flat = s.startswith("b")
    if flat:
        s = s[1:]
    has_7 = s.endswith("7")
    if has_7:
        s = s[:-1]
    upper = s[0].isupper() if s else True
    base = s.upper()
    if base not in NUMERAL_DEGREE:
        raise ValueError(f"Unknown numeral: {numeral_str}")
    degree = NUMERAL_DEGREE[base]
    if upper:
        quality = "dom7" if has_7 else "major"
    else:
        quality = "min7" if has_7 else "minor"
    return degree, quality, flat


class ProgressionAnalyzer:

    @staticmethod
    def parse_roman(numeral_str: str, key: str) -> Chord:
        degree, quality, flat = _parse_numeral(numeral_str)
        key_idx = _note_index(key)
        major_intervals = SCALE_INTERVALS["major"]
        if degree < len(major_intervals):
            semitones = major_intervals[degree]
        else:
            semitones = degree * 2
        if flat:
            semitones -= 1
        root = CHROMATIC_NOTES[(key_idx + semitones) % 12]
        return ChordLibrary.build_chord(root, quality)

    @staticmethod
    def build_progression(key: str, numeral_list: list[str]) -> Progression:
        rn_list: list[RomanNumeral] = []
        chords: list[Chord] = []
        for n in numeral_list:
            degree, quality, flat = _parse_numeral(n)
            rn_list.append(RomanNumeral(numeral=n, chord_quality=quality, scale_degree=degree))
            chords.append(ProgressionAnalyzer.parse_roman(n, key))
        return Progression(
            key=key,
            roman_numerals=tuple(rn_list),
            chords=tuple(chords),
        )

    @staticmethod
    def analyze_function(chord: Chord, key: str) -> str:
        key_idx = _note_index(key)
        chord_idx = _note_index(chord.root)
        degree = (chord_idx - key_idx) % 12
        tonic_degrees = {0, 4, 9}
        subdominant_degrees = {2, 5}
        dominant_degrees = {7, 11}
        if degree in tonic_degrees:
            return "tonic"
        if degree in subdominant_degrees:
            return "subdominant"
        if degree in dominant_degrees:
            return "dominant"
        return "chromatic"

    @staticmethod
    def detect_cadence(chord1: Chord, chord2: Chord, key: str) -> str:
        key_idx = _note_index(key)
        d1 = (_note_index(chord1.root) - key_idx) % 12
        d2 = (_note_index(chord2.root) - key_idx) % 12
        if d1 == 7 and d2 == 0:
            return "authentic"
        if d2 == 7:
            return "half"
        if d1 == 5 and d2 == 0:
            return "plagal"
        if d1 == 7 and d2 != 0:
            return "deceptive"
        return "other"

    @staticmethod
    def functional_analysis(progression: Progression) -> list[str]:
        return [
            ProgressionAnalyzer.analyze_function(c, progression.key)
            for c in progression.chords
        ]


class CamelotWheel:

    @staticmethod
    def get_code(key: str, quality: str = "major") -> str:
        lookup = f"{key} {quality}"
        return CAMELOT_MAP.get(lookup, "")

    @staticmethod
    def is_compatible(key1: str, key2: str) -> bool:
        c1 = CAMELOT_MAP.get(key1, "")
        c2 = CAMELOT_MAP.get(key2, "")
        if not c1 or not c2:
            return False
        n1, l1 = int(c1[:-1]), c1[-1]
        n2, l2 = int(c2[:-1]), c2[-1]
        if c1 == c2:
            return True
        if l1 == l2 and abs(n1 - n2) <= 1:
            return True
        if l1 == l2 and {n1, n2} == {1, 12}:
            return True
        if n1 == n2 and l1 != l2:
            return True
        return False

    @staticmethod
    def compatible_keys(key: str) -> list[str]:
        return [k for k in CAMELOT_MAP if CamelotWheel.is_compatible(key, k) and k != key]
