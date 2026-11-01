"""Harmonic progression analysis: Roman numeral parsing, functional harmony, tension arcs."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class HarmonicFunction(str, Enum):
    TONIC = "T"
    PREDOMINANT = "PD"
    DOMINANT = "D"
    APPLIED_DOMINANT = "V/"
    PASSING = "PASS"
    NEIGHBOR = "N"


ROMAN_TO_SCALE_DEGREE: Dict[str, int] = {
    "I": 0, "bII": 1, "II": 2, "bIII": 3, "III": 4,
    "IV": 5, "#IV": 6, "V": 7, "bVI": 8, "VI": 9,
    "bVII": 10, "VII": 11,
    "i": 0, "bii": 1, "ii": 2, "biii": 3, "iii": 4,
    "iv": 5, "#iv": 6, "v": 7, "bvi": 8, "vi": 9,
    "bvii": 10, "vii": 11,
}

FUNCTION_MAP: Dict[str, HarmonicFunction] = {
    "I": HarmonicFunction.TONIC, "i": HarmonicFunction.TONIC,
    "III": HarmonicFunction.TONIC, "iii": HarmonicFunction.TONIC,
    "VI": HarmonicFunction.TONIC, "vi": HarmonicFunction.TONIC,
    "II": HarmonicFunction.PREDOMINANT, "ii": HarmonicFunction.PREDOMINANT,
    "IV": HarmonicFunction.PREDOMINANT, "iv": HarmonicFunction.PREDOMINANT,
    "bVI": HarmonicFunction.PREDOMINANT,
    "V": HarmonicFunction.DOMINANT, "v": HarmonicFunction.DOMINANT,
    "VII": HarmonicFunction.DOMINANT, "vii": HarmonicFunction.DOMINANT,
    "bVII": HarmonicFunction.DOMINANT,
}


@dataclass
class Chord:
    root: int
    quality: str
    roman: str
    inversion: int = 0
    extensions: List[int] = field(default_factory=list)
    function: HarmonicFunction = HarmonicFunction.TONIC


@dataclass
class Progression:
    chords: List[Chord]
    key_root: int = 0
    mode: str = "major"

    @property
    def length(self) -> int:
        return len(self.chords)

    def functions(self) -> List[HarmonicFunction]:
        return [c.function for c in self.chords]

    def roots(self) -> List[int]:
        return [c.root for c in self.chords]


def parse_roman(symbol: str) -> Chord:
    base = symbol.strip()
    inversion = 0
    quality = "major" if base[0].isupper() else "minor"
    extensions: List[int] = []
    if "7" in base:
        extensions.append(7)
    if "9" in base:
        extensions.extend([7, 9])
    if "dim" in base.lower():
        quality = "diminished"
    elif "aug" in base.lower():
        quality = "augmented"
    if "/6" in base or base.endswith("6"):
        inversion = 1
    elif "64" in base:
        inversion = 2
    roman_key = base.split("7")[0].split("9")[0].split("dim")[0].split("aug")[0]
    roman_key = roman_key.replace("/6", "").replace("64", "").strip()
    root = ROMAN_TO_SCALE_DEGREE.get(roman_key, 0)
    func = FUNCTION_MAP.get(roman_key, HarmonicFunction.PASSING)
    return Chord(root=root, quality=quality, roman=symbol,
                 inversion=inversion, extensions=extensions, function=func)


def progression_from_symbols(symbols: List[str], key_root: int = 0,
                              mode: str = "major") -> Progression:
    chords = [parse_roman(s) for s in symbols]
    return Progression(chords=chords, key_root=key_root, mode=mode)


def functional_analysis(prog: Progression) -> List[Tuple[str, HarmonicFunction]]:
    return [(c.roman, c.function) for c in prog.chords]


TENSION_VALUES: Dict[HarmonicFunction, float] = {
    HarmonicFunction.TONIC: 0.0,
    HarmonicFunction.PREDOMINANT: 0.4,
    HarmonicFunction.DOMINANT: 0.8,
    HarmonicFunction.APPLIED_DOMINANT: 0.7,
    HarmonicFunction.PASSING: 0.3,
    HarmonicFunction.NEIGHBOR: 0.2,
}

INVERSION_TENSION_BOOST: Dict[int, float] = {
    0: 0.0, 1: 0.05, 2: 0.15, 3: 0.1,
}


def chord_tension(chord: Chord) -> float:
    base = TENSION_VALUES.get(chord.function, 0.3)
    inv_boost = INVERSION_TENSION_BOOST.get(chord.inversion, 0.0)
    ext_boost = 0.05 * len(chord.extensions)
    return min(1.0, base + inv_boost + ext_boost)


def tension_arc(prog: Progression) -> np.ndarray:
    if prog.length == 0:
        return np.array([])
    return np.array([chord_tension(c) for c in prog.chords])


def harmonic_rhythm_density(prog: Progression, durations: Optional[List[float]] = None) -> float:
    if prog.length <= 1:
        return 0.0
    if durations is None:
        return float(prog.length)
    total_dur = sum(durations)
    return prog.length / total_dur if total_dur > 0 else 0.0


def detect_cadence_type(last_two: List[Chord]) -> str:
    if len(last_two) < 2:
        return "none"
    prev, curr = last_two[-2], last_two[-1]
    if prev.function == HarmonicFunction.DOMINANT and curr.function == HarmonicFunction.TONIC:
        return "authentic"
    if prev.function == HarmonicFunction.PREDOMINANT and curr.function == HarmonicFunction.TONIC:
        return "plagal"
    if curr.function == HarmonicFunction.DOMINANT:
        return "half"
    if prev.function == HarmonicFunction.DOMINANT and curr.function != HarmonicFunction.TONIC:
        return "deceptive"
    return "none"


def root_motion_profile(prog: Progression) -> Dict[int, int]:
    profile: Dict[int, int] = {}
    for i in range(len(prog.chords) - 1):
        motion = (prog.chords[i + 1].root - prog.chords[i].root) % 12
        profile[motion] = profile.get(motion, 0) + 1
    return profile


def fifth_motion_ratio(prog: Progression) -> float:
    profile = root_motion_profile(prog)
    total = sum(profile.values())
    if total == 0:
        return 0.0
    fifths = profile.get(7, 0) + profile.get(5, 0)
    return fifths / total


def harmonic_variety(prog: Progression) -> float:
    if prog.length == 0:
        return 0.0
    unique_roots = len(set(c.root for c in prog.chords))
    return unique_roots / min(prog.length, 12)


@dataclass
class ModulationEvent:
    position: int
    from_key: int
    to_key: int
    pivot_chord: Optional[Chord] = None
    type: str = "pivot"


def detect_modulations(prog: Progression, window: int = 4) -> List[ModulationEvent]:
    if prog.length < window * 2:
        return []
    modulations = []
    for i in range(window, prog.length - window):
        roots_before = [prog.chords[j].root for j in range(i - window, i)]
        roots_after = [prog.chords[j].root for j in range(i, min(i + window, prog.length))]
        center_before = int(np.round(np.mean(roots_before))) % 12
        center_after = int(np.round(np.mean(roots_after))) % 12
        if center_before != center_after:
            motion = (center_after - center_before) % 12
            if motion in (5, 7, 2, 3, 4, 8, 9):
                modulations.append(ModulationEvent(
                    position=i, from_key=center_before, to_key=center_after,
                    pivot_chord=prog.chords[i], type="pivot",
                ))
    return modulations


def progression_complexity(prog: Progression) -> float:
    if prog.length <= 1:
        return 0.0
    variety = harmonic_variety(prog)
    fifth_ratio = fifth_motion_ratio(prog)
    tension = tension_arc(prog)
    tension_var = float(np.std(tension)) if len(tension) > 1 else 0.0
    return 0.4 * variety + 0.3 * (1.0 - fifth_ratio) + 0.3 * tension_var
