"""Score reading and analysis for music education.

Staff position mapping, key/time signature identification,
reading difficulty analysis, and exercise generation.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np


class ClefType(enum.Enum):
    TREBLE = "treble"
    BASS = "bass"
    ALTO = "alto"
    TENOR = "tenor"


@dataclass
class KeySignatureInfo:
    sharps: int = 0
    flats: int = 0
    key_name: str = "C major"
    mode: str = "major"


@dataclass
class TimeSignatureInfo:
    numerator: int = 4
    denominator: int = 4
    compound: bool = False


@dataclass
class NoteOnStaff:
    line_or_space: int
    clef: ClefType = ClefType.TREBLE
    accidental: Optional[str] = None


_TREBLE_BASE = 64
_BASS_BASE = 43
_ALTO_BASE = 60
_TENOR_BASE = 57

_CLEF_BASES = {
    ClefType.TREBLE: _TREBLE_BASE,
    ClefType.BASS: _BASS_BASE,
    ClefType.ALTO: _ALTO_BASE,
    ClefType.TENOR: _TENOR_BASE,
}

_DIATONIC_STEPS = [0, 2, 4, 5, 7, 9, 11]


def staff_position_to_midi(position: int, clef: ClefType = ClefType.TREBLE) -> int:
    base = _CLEF_BASES[clef]
    octaves, step = divmod(position, 7)
    if step < 0:
        octaves -= 1
        step += 7
    return base + octaves * 12 + _DIATONIC_STEPS[step]


def midi_to_staff_position(midi: int, clef: ClefType = ClefType.TREBLE) -> int:
    base = _CLEF_BASES[clef]
    diff = midi - base
    octaves = diff // 12
    remainder = diff % 12
    best_step = 0
    best_dist = 99
    for i, s in enumerate(_DIATONIC_STEPS):
        d = abs(s - remainder)
        if d < best_dist:
            best_dist = d
            best_step = i
    return octaves * 7 + best_step


_SHARP_ORDER = ["F#", "C#", "G#", "D#", "A#", "E#", "B#"]
_FLAT_ORDER = ["Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb"]
_SHARP_KEYS = ["C", "G", "D", "A", "E", "B", "F#", "C#"]
_FLAT_KEYS = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]


def identify_key_signature(sharps: int = 0, flats: int = 0, minor: bool = False) -> KeySignatureInfo:
    if sharps > 0:
        base = _SHARP_KEYS[min(sharps, 7)]
        mode = "minor" if minor else "major"
        name = f"{base} {mode}"
        return KeySignatureInfo(sharps=sharps, flats=0, key_name=name, mode=mode)
    if flats > 0:
        base = _FLAT_KEYS[min(flats, 7)]
        mode = "minor" if minor else "major"
        name = f"{base} {mode}"
        return KeySignatureInfo(sharps=0, flats=flats, key_name=name, mode=mode)
    mode = "minor" if minor else "major"
    key = "A minor" if minor else "C major"
    return KeySignatureInfo(key_name=key, mode=mode)


def analyze_reading_difficulty(
    midi_notes: list[int],
    clef: ClefType = ClefType.TREBLE,
    accidentals: int = 0,
    rhythm_complexity: float = 0.5,
) -> float:
    if not midi_notes:
        return 0.0
    positions = [midi_to_staff_position(m, clef) for m in midi_notes]
    ledger_count = sum(1 for p in positions if p < -2 or p > 12)
    ledger_penalty = ledger_count / max(len(positions), 1)
    note_range = max(midi_notes) - min(midi_notes)
    range_penalty = min(note_range / 36.0, 1.0)
    acc_penalty = min(accidentals / max(len(midi_notes), 1), 1.0)
    score = 0.3 * ledger_penalty + 0.25 * range_penalty + 0.25 * acc_penalty + 0.2 * rhythm_complexity
    return float(np.clip(score, 0.0, 1.0))


def generate_reading_exercise(
    clef: ClefType = ClefType.TREBLE,
    n_notes: int = 8,
    difficulty: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[int]:
    if rng is None:
        rng = np.random.default_rng()
    base = _CLEF_BASES[clef]
    spread = int(6 + difficulty * 18)
    low = base - spread // 2
    high = base + spread // 2
    notes = rng.integers(low, high + 1, size=n_notes).tolist()
    return notes


_CONFUSION_PAIRS: dict[ClefType, list[tuple[str, str]]] = {
    ClefType.TREBLE: [("E4", "F4"), ("B4", "C5"), ("D5", "E5"), ("A4", "B4")],
    ClefType.BASS: [("G2", "A2"), ("B2", "C3"), ("D3", "E3"), ("F2", "G2")],
    ClefType.ALTO: [("B3", "C4"), ("D4", "E4"), ("F3", "G3")],
    ClefType.TENOR: [("A3", "B3"), ("C4", "D4"), ("E3", "F3")],
}


def common_reading_errors(clef: ClefType = ClefType.TREBLE) -> list[tuple[str, str]]:
    return _CONFUSION_PAIRS.get(clef, [])
