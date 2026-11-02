"""Harmonic analysis: chord identification, Roman numeral analysis, functional harmony."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_TEMPLATES: Dict[str, List[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim7": [0, 3, 6, 9],
    "hdim7": [0, 3, 6, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "add9": [0, 2, 4, 7],
    "6": [0, 4, 7, 9],
    "min6": [0, 3, 7, 9],
}


@dataclass
class ChordLabel:
    root: int
    quality: str
    bass: Optional[int] = None

    @property
    def name(self) -> str:
        root_name = PITCH_CLASSES[self.root % 12]
        q = self.quality if self.quality != "maj" else ""
        if self.bass is not None and self.bass != self.root:
            bass_name = PITCH_CLASSES[self.bass % 12]
            return f"{root_name}{q}/{bass_name}"
        return f"{root_name}{q}"

    @property
    def pitch_classes(self) -> Set[int]:
        template = CHORD_TEMPLATES.get(self.quality, [0, 4, 7])
        return {(self.root + iv) % 12 for iv in template}


def chroma_from_pitches(pitches: np.ndarray) -> np.ndarray:
    chroma = np.zeros(12)
    for p in pitches:
        chroma[int(p) % 12] += 1.0
    total = chroma.sum()
    if total > 0:
        chroma /= total
    return chroma


def template_matching_chord(chroma: np.ndarray,
                            qualities: Optional[List[str]] = None) -> ChordLabel:
    if qualities is None:
        qualities = list(CHORD_TEMPLATES.keys())
    best_score = -1.0
    best_root = 0
    best_quality = "maj"
    for root in range(12):
        for quality in qualities:
            template = CHORD_TEMPLATES[quality]
            t_chroma = np.zeros(12)
            for iv in template:
                t_chroma[(root + iv) % 12] = 1.0
            t_chroma /= t_chroma.sum()
            score = float(np.dot(chroma, t_chroma))
            if score > best_score:
                best_score = score
                best_root = root
                best_quality = quality
    bass = int(np.argmax(chroma))
    return ChordLabel(root=best_root, quality=best_quality,
                      bass=bass if bass != best_root else None)


def identify_chord(pitches: np.ndarray,
                   qualities: Optional[List[str]] = None) -> ChordLabel:
    chroma = chroma_from_pitches(pitches)
    return template_matching_chord(chroma, qualities)


ROMAN_NUMERALS = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]

MAJOR_SCALE_DEGREES = {0: "I", 2: "II", 4: "III", 5: "IV", 7: "V", 9: "VI", 11: "VII"}
MINOR_SCALE_DEGREES = {0: "i", 2: "ii", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}


@dataclass
class RomanNumeral:
    degree: str
    quality: str
    key: int
    mode: str = "major"

    @property
    def label(self) -> str:
        if self.quality in ("min", "min7", "dim", "hdim7"):
            return self.degree.lower()
        return self.degree


def to_roman_numeral(chord: ChordLabel, key: int, mode: str = "major") -> RomanNumeral:
    interval = (chord.root - key) % 12
    if mode == "major":
        degree = MAJOR_SCALE_DEGREES.get(interval, ROMAN_NUMERALS[interval])
    else:
        degree = MINOR_SCALE_DEGREES.get(interval, ROMAN_NUMERALS[interval])
    return RomanNumeral(degree=degree, quality=chord.quality, key=key, mode=mode)


@dataclass
class HarmonicFunction:
    name: str
    tension: float

FUNCTIONS: Dict[str, HarmonicFunction] = {
    "tonic": HarmonicFunction("tonic", 0.0),
    "subdominant": HarmonicFunction("subdominant", 0.3),
    "dominant": HarmonicFunction("dominant", 0.7),
    "predominant": HarmonicFunction("predominant", 0.4),
    "chromatic": HarmonicFunction("chromatic", 0.5),
}

DEGREE_FUNCTION: Dict[str, str] = {
    "I": "tonic", "i": "tonic",
    "II": "predominant", "ii": "predominant",
    "III": "tonic", "iii": "tonic",
    "IV": "subdominant", "iv": "subdominant",
    "V": "dominant", "v": "dominant",
    "VI": "tonic", "vi": "tonic",
    "VII": "dominant", "vii": "dominant",
}


def classify_function(rn: RomanNumeral) -> HarmonicFunction:
    key = rn.degree.upper().strip("B#")
    func_name = DEGREE_FUNCTION.get(rn.degree, DEGREE_FUNCTION.get(key, "chromatic"))
    return FUNCTIONS[func_name]


@dataclass
class HarmonicProgressionStep:
    chord: ChordLabel
    roman: RomanNumeral
    function: HarmonicFunction
    time: float


def analyze_progression(chords: List[np.ndarray], times: np.ndarray,
                        key: int, mode: str = "major") -> List[HarmonicProgressionStep]:
    steps: List[HarmonicProgressionStep] = []
    for i, pitches in enumerate(chords):
        chord = identify_chord(pitches)
        rn = to_roman_numeral(chord, key, mode)
        func = classify_function(rn)
        t = float(times[i]) if i < len(times) else float(i)
        steps.append(HarmonicProgressionStep(chord=chord, roman=rn,
                                             function=func, time=t))
    return steps


def functional_tension_curve(steps: List[HarmonicProgressionStep]) -> np.ndarray:
    return np.array([s.function.tension for s in steps])


def detect_cadence(steps: List[HarmonicProgressionStep]) -> List[Tuple[int, str]]:
    cadences: List[Tuple[int, str]] = []
    for i in range(1, len(steps)):
        prev_fn = steps[i - 1].function.name
        curr_fn = steps[i].function.name
        if prev_fn == "dominant" and curr_fn == "tonic":
            cadences.append((i, "authentic"))
        elif prev_fn == "subdominant" and curr_fn == "tonic":
            cadences.append((i, "plagal"))
        elif prev_fn == "dominant" and curr_fn != "tonic":
            cadences.append((i, "deceptive"))
        elif prev_fn == "tonic" and curr_fn == "dominant":
            cadences.append((i, "half"))
    return cadences


def harmonic_rhythm(steps: List[HarmonicProgressionStep]) -> np.ndarray:
    if len(steps) < 2:
        return np.array([])
    durations = []
    for i in range(1, len(steps)):
        durations.append(steps[i].time - steps[i - 1].time)
    return np.array(durations)


def chord_complexity(chord: ChordLabel) -> float:
    pcs = chord.pitch_classes
    n = len(pcs)
    if n <= 2:
        return 0.1
    intervals = set()
    pcs_list = sorted(pcs)
    for i in range(len(pcs_list)):
        for j in range(i + 1, len(pcs_list)):
            intervals.add((pcs_list[j] - pcs_list[i]) % 12)
    dissonant = {1, 2, 6, 10, 11}
    diss_count = len(intervals & dissonant)
    return diss_count / max(len(intervals), 1)


def progression_complexity(steps: List[HarmonicProgressionStep]) -> float:
    if not steps:
        return 0.0
    roots = [s.chord.root for s in steps]
    unique_roots = len(set(roots))
    root_entropy = unique_roots / 12.0
    mean_chord_complex = np.mean([chord_complexity(s.chord) for s in steps])
    cadences = detect_cadence(steps)
    cadence_density = len(cadences) / max(len(steps), 1)
    return float(0.4 * root_entropy + 0.4 * mean_chord_complex + 0.2 * cadence_density)


def modulation_points(steps: List[HarmonicProgressionStep],
                      window: int = 4) -> List[Tuple[int, int]]:
    if len(steps) < window * 2:
        return []
    modulations: List[Tuple[int, int]] = []
    for i in range(window, len(steps) - window):
        left_roots = [s.chord.root for s in steps[i - window:i]]
        right_roots = [s.chord.root for s in steps[i:i + window]]
        left_center = int(np.round(np.mean(left_roots))) % 12
        right_center = int(np.round(np.mean(right_roots))) % 12
        if left_center != right_center:
            modulations.append((i, right_center))
    return modulations
