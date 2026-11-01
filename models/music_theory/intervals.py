"""Interval calculation, pitch-class utilities, and consonance scoring."""
from __future__ import annotations

from dataclasses import dataclass


PITCH_CLASSES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

ENHARMONIC_MAP: dict[str, str] = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#",
    "Bb": "A#", "Cb": "B", "E#": "F", "B#": "C",
}

CONSONANCE_SCORES: dict[int, float] = {
    0:  1.00,   # unison
    1:  0.10,   # minor 2nd
    2:  0.20,   # major 2nd
    3:  0.80,   # minor 3rd
    4:  0.85,   # major 3rd
    5:  0.90,   # perfect 4th
    6:  0.15,   # tritone
    7:  0.95,   # perfect 5th
    8:  0.75,   # minor 6th
    9:  0.80,   # major 6th
    10: 0.30,   # minor 7th
    11: 0.25,   # major 7th
    12: 1.00,   # octave
}

INTERVAL_NAMES: dict[int, str] = {
    0: "unison", 1: "minor 2nd", 2: "major 2nd", 3: "minor 3rd",
    4: "major 3rd", 5: "perfect 4th", 6: "tritone", 7: "perfect 5th",
    8: "minor 6th", 9: "major 6th", 10: "minor 7th", 11: "major 7th",
    12: "perfect octave",
}


@dataclass(frozen=True)
class Interval:
    semitones: int
    quality: str
    number: str
    consonance: float


INTERVAL_TABLE: dict[int, Interval] = {
    s: Interval(s, q.split()[0] if " " in q else "perfect", q, CONSONANCE_SCORES[s])
    for s, q in INTERVAL_NAMES.items()
}


def normalize_note(note: str) -> str:
    return ENHARMONIC_MAP.get(note, note) if note not in PITCH_CLASSES else note


def note_to_index(note: str) -> int:
    n = normalize_note(note)
    return PITCH_CLASSES.index(n)


def interval_between(note1: str, note2: str) -> int:
    return (note_to_index(note2) - note_to_index(note1)) % 12


def interval_name(semitones: int) -> str:
    return INTERVAL_NAMES.get(semitones % 12, f"{semitones} semitones")


def consonance_score(semitones: int) -> float:
    return CONSONANCE_SCORES.get(semitones % 12, 0.5)


def melodic_tension(notes: list[str]) -> float:
    if len(notes) < 2:
        return 0.0
    total = 0.0
    for i in range(len(notes) - 1):
        s = interval_between(notes[i], notes[i + 1])
        total += 1.0 - consonance_score(s)
    return total / (len(notes) - 1)


class IntervalCalculator:
    @staticmethod
    def note_to_midi(note: str, octave: int = 4) -> int:
        return note_to_index(note) + 12 * (octave + 1)

    @staticmethod
    def midi_to_note(midi: int) -> tuple[str, int]:
        return PITCH_CLASSES[midi % 12], (midi // 12) - 1

    @staticmethod
    def interval_between(note1: str, note2: str) -> Interval:
        s = interval_between(note1, note2)
        return INTERVAL_TABLE[s]

    @staticmethod
    def transpose(note: str, semitones: int) -> str:
        return PITCH_CLASSES[(note_to_index(note) + semitones) % 12]

    @staticmethod
    def invert(interval: Interval) -> Interval:
        return INTERVAL_TABLE[(12 - interval.semitones) % 12]

    @staticmethod
    def is_consonant(interval: Interval, threshold: float = 0.60) -> bool:
        return interval.consonance >= threshold

    @staticmethod
    def consonance_ranking() -> list[Interval]:
        return sorted(INTERVAL_TABLE.values(), key=lambda i: -i.consonance)
