"""Chord symbol parsing and manipulation: lead-sheet notation parsing,
pitch-class extraction, interval content, transposition, functional analysis.

Handles standard jazz/pop chord symbols like Cmaj7, Dm7b5, G7#9, Bb/D, Asus4.
"""
from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_NOTE_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}


class ChordQuality(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    DOMINANT = "dominant"
    HALF_DIMINISHED = "half_diminished"
    MAJOR_SEVENTH = "major_seventh"
    MINOR_SEVENTH = "minor_seventh"
    DIMINISHED_SEVENTH = "diminished_seventh"
    AUGMENTED_SEVENTH = "augmented_seventh"
    SUS2 = "sus2"
    SUS4 = "sus4"
    POWER = "power"


QUALITY_INTERVALS: Dict[ChordQuality, List[int]] = {
    ChordQuality.MAJOR: [0, 4, 7],
    ChordQuality.MINOR: [0, 3, 7],
    ChordQuality.DIMINISHED: [0, 3, 6],
    ChordQuality.AUGMENTED: [0, 4, 8],
    ChordQuality.DOMINANT: [0, 4, 7, 10],
    ChordQuality.HALF_DIMINISHED: [0, 3, 6, 10],
    ChordQuality.MAJOR_SEVENTH: [0, 4, 7, 11],
    ChordQuality.MINOR_SEVENTH: [0, 3, 7, 10],
    ChordQuality.DIMINISHED_SEVENTH: [0, 3, 6, 9],
    ChordQuality.AUGMENTED_SEVENTH: [0, 4, 8, 10],
    ChordQuality.SUS2: [0, 2, 7],
    ChordQuality.SUS4: [0, 5, 7],
    ChordQuality.POWER: [0, 7],
}


@dataclass
class ChordSymbol:
    root: int
    quality: ChordQuality
    bass: Optional[int] = None
    extensions: List[int] = field(default_factory=list)
    alterations: Dict[int, int] = field(default_factory=dict)


def _parse_root(text: str) -> Tuple[int, str]:
    """Parse root note, return (pitch_class, remainder)."""
    if len(text) >= 2 and text[1] in ("#", "b"):
        note = text[:2]
        rest = text[2:]
    else:
        note = text[:1]
        rest = text[1:]
    if note not in _NOTE_MAP:
        raise ValueError(f"Unknown note: {note}")
    return _NOTE_MAP[note], rest


def parse_chord_symbol(text: str) -> ChordSymbol:
    """Parse a chord symbol string into a ChordSymbol."""
    text = text.strip()
    if not text:
        raise ValueError("Empty chord symbol")

    bass = None
    if "/" in text:
        parts = text.split("/", 1)
        text = parts[0]
        if parts[1] in _NOTE_MAP:
            bass = _NOTE_MAP[parts[1]]
        elif len(parts[1]) >= 2 and parts[1][:2] in _NOTE_MAP:
            bass = _NOTE_MAP[parts[1][:2]]

    root, rest = _parse_root(text)
    quality = ChordQuality.MAJOR
    extensions = []
    alterations: Dict[int, int] = {}

    if rest.startswith("maj7") or rest.startswith("M7") or rest.startswith("Δ7"):
        quality = ChordQuality.MAJOR_SEVENTH
        rest = rest[4:] if rest.startswith("maj7") else rest[2:]
    elif rest.startswith("maj") or rest.startswith("M"):
        quality = ChordQuality.MAJOR
        rest = rest[3:] if rest.startswith("maj") else rest[1:]
    elif rest.startswith("m7b5") or rest.startswith("ø"):
        quality = ChordQuality.HALF_DIMINISHED
        rest = rest[4:] if rest.startswith("m7b5") else rest[1:]
    elif rest.startswith("m7") or rest.startswith("min7") or rest.startswith("-7"):
        quality = ChordQuality.MINOR_SEVENTH
        rest = rest[2:] if rest.startswith("m7") or rest.startswith("-7") else rest[4:]
    elif rest.startswith("min") or rest.startswith("m") or rest.startswith("-"):
        quality = ChordQuality.MINOR
        rest = rest[3:] if rest.startswith("min") else rest[1:]
    elif rest.startswith("dim7") or rest.startswith("°7"):
        quality = ChordQuality.DIMINISHED_SEVENTH
        rest = rest[4:] if rest.startswith("dim7") else rest[2:]
    elif rest.startswith("dim") or rest.startswith("°"):
        quality = ChordQuality.DIMINISHED
        rest = rest[3:] if rest.startswith("dim") else rest[1:]
    elif rest.startswith("aug7") or rest.startswith("+7"):
        quality = ChordQuality.AUGMENTED_SEVENTH
        rest = rest[4:] if rest.startswith("aug7") else rest[2:]
    elif rest.startswith("aug") or rest.startswith("+"):
        quality = ChordQuality.AUGMENTED
        rest = rest[3:] if rest.startswith("aug") else rest[1:]
    elif rest.startswith("sus2"):
        quality = ChordQuality.SUS2
        rest = rest[4:]
    elif rest.startswith("sus4") or rest.startswith("sus"):
        quality = ChordQuality.SUS4
        rest = rest[4:] if rest.startswith("sus4") else rest[3:]
    elif rest.startswith("5"):
        quality = ChordQuality.POWER
        rest = rest[1:]
    elif rest.startswith("7"):
        quality = ChordQuality.DOMINANT
        rest = rest[1:]

    alt_pattern = re.compile(r"([#b])(\d+)")
    for m in alt_pattern.finditer(rest):
        degree = int(m.group(2))
        shift = 1 if m.group(1) == "#" else -1
        alterations[degree] = shift

    ext_pattern = re.compile(r"(?:add)?(\d+)")
    for m in ext_pattern.finditer(rest):
        deg = int(m.group(1))
        if deg > 7 and deg not in alterations:
            interval = _degree_to_semitones(deg)
            if interval is not None:
                extensions.append(interval)

    return ChordSymbol(root=root, quality=quality, bass=bass,
                       extensions=extensions, alterations=alterations)


def _degree_to_semitones(degree: int) -> Optional[int]:
    """Map scale degree (9, 11, 13) to semitones above root."""
    mapping = {9: 14, 11: 17, 13: 21, 2: 2, 4: 5, 6: 9}
    if degree in mapping:
        return mapping[degree] % 12
    return None


def chord_to_text(chord: ChordSymbol, use_flats: bool = False) -> str:
    """Convert ChordSymbol to string notation."""
    names = FLAT_NAMES if use_flats else NOTE_NAMES
    text = names[chord.root % 12]

    quality_str = {
        ChordQuality.MAJOR: "", ChordQuality.MINOR: "m",
        ChordQuality.DIMINISHED: "dim", ChordQuality.AUGMENTED: "aug",
        ChordQuality.DOMINANT: "7", ChordQuality.HALF_DIMINISHED: "m7b5",
        ChordQuality.MAJOR_SEVENTH: "maj7", ChordQuality.MINOR_SEVENTH: "m7",
        ChordQuality.DIMINISHED_SEVENTH: "dim7",
        ChordQuality.AUGMENTED_SEVENTH: "aug7",
        ChordQuality.SUS2: "sus2", ChordQuality.SUS4: "sus4",
        ChordQuality.POWER: "5",
    }
    text += quality_str.get(chord.quality, "")

    for deg, shift in sorted(chord.alterations.items()):
        text += ("#" if shift > 0 else "b") + str(deg)

    if chord.bass is not None:
        text += "/" + names[chord.bass % 12]
    return text


def chord_pitches(chord: ChordSymbol) -> List[int]:
    """Pitch classes present in the chord."""
    intervals = list(QUALITY_INTERVALS[chord.quality])
    for ext in chord.extensions:
        if ext not in intervals:
            intervals.append(ext)
    for deg, shift in chord.alterations.items():
        base = _degree_to_semitones(deg)
        if base is not None:
            altered = (base + shift) % 12
            if base in intervals:
                intervals.remove(base)
            if altered not in intervals:
                intervals.append(altered)
    pcs = [(chord.root + i) % 12 for i in intervals]
    if chord.bass is not None and chord.bass not in pcs:
        pcs.insert(0, chord.bass)
    return sorted(set(pcs))


def chord_to_midi(chord: ChordSymbol, octave: int = 4) -> List[int]:
    """Absolute MIDI notes for the chord in a given octave."""
    base = 12 * (octave + 1)
    intervals = list(QUALITY_INTERVALS[chord.quality])
    for ext in chord.extensions:
        if ext not in intervals:
            intervals.append(ext)
    notes = sorted(set(base + chord.root + i for i in intervals))
    if chord.bass is not None:
        bass_midi = base + chord.bass
        if bass_midi >= notes[0]:
            bass_midi -= 12
        notes.insert(0, bass_midi)
    return notes


def transpose_chord(chord: ChordSymbol, semitones: int) -> ChordSymbol:
    """Transpose a chord by the given number of semitones."""
    new_root = (chord.root + semitones) % 12
    new_bass = (chord.bass + semitones) % 12 if chord.bass is not None else None
    return ChordSymbol(root=new_root, quality=chord.quality,
                       bass=new_bass, extensions=list(chord.extensions),
                       alterations=dict(chord.alterations))


def enharmonic_root(root: int, prefer_flat: bool = True) -> int:
    """Return the same pitch class (enharmonic identity)."""
    return root % 12


def chord_complexity(chord: ChordSymbol) -> float:
    """Heuristic complexity score: more extensions/alterations = higher."""
    base = {
        ChordQuality.POWER: 0.1, ChordQuality.MAJOR: 0.2,
        ChordQuality.MINOR: 0.2, ChordQuality.SUS2: 0.3,
        ChordQuality.SUS4: 0.3, ChordQuality.DOMINANT: 0.5,
        ChordQuality.DIMINISHED: 0.4, ChordQuality.AUGMENTED: 0.4,
        ChordQuality.MAJOR_SEVENTH: 0.6, ChordQuality.MINOR_SEVENTH: 0.5,
        ChordQuality.HALF_DIMINISHED: 0.7, ChordQuality.DIMINISHED_SEVENTH: 0.7,
        ChordQuality.AUGMENTED_SEVENTH: 0.7,
    }.get(chord.quality, 0.3)
    return base + 0.15 * len(chord.extensions) + 0.2 * len(chord.alterations)


def common_tones(a: ChordSymbol, b: ChordSymbol) -> List[int]:
    """Shared pitch classes between two chords."""
    pcs_a = set(chord_pitches(a))
    pcs_b = set(chord_pitches(b))
    return sorted(pcs_a & pcs_b)


def interval_content(chord: ChordSymbol) -> np.ndarray:
    """6-element interval-class vector (IC1 through IC6)."""
    pcs = chord_pitches(chord)
    ic_vec = np.zeros(6)
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            interval = abs(pcs[j] - pcs[i]) % 12
            ic = min(interval, 12 - interval)
            if 1 <= ic <= 6:
                ic_vec[ic - 1] += 1
    return ic_vec


_SCALE_DEGREES = {
    0: "tonic", 1: "neapolitan", 2: "supertonic", 3: "mediant",
    4: "subdominant", 5: "subdominant", 6: "submediant",
    7: "dominant", 8: "submediant", 9: "submediant",
    10: "subtonic", 11: "leading_tone",
}

_FUNCTION_MAP = {
    "tonic": "tonic", "supertonic": "predominant", "mediant": "tonic",
    "subdominant": "subdominant", "dominant": "dominant",
    "submediant": "tonic", "subtonic": "dominant",
    "leading_tone": "dominant", "neapolitan": "predominant",
}


def classify_function(chord: ChordSymbol, key_root: int) -> str:
    """Classify harmonic function: tonic, dominant, subdominant, predominant."""
    degree = (chord.root - key_root) % 12
    scale_deg = _SCALE_DEGREES.get(degree, "chromatic")
    return _FUNCTION_MAP.get(scale_deg, "chromatic")


def pitch_class_name(pc: int, use_flats: bool = False) -> str:
    """Convert pitch class (0-11) to note name."""
    names = FLAT_NAMES if use_flats else NOTE_NAMES
    return names[pc % 12]


def name_to_pitch_class(name: str) -> int:
    """Convert note name to pitch class."""
    if name not in _NOTE_MAP:
        raise ValueError(f"Unknown note: {name}")
    return _NOTE_MAP[name]
