"""Alternative visual notation systems for accessibility."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_BRAILLE_PITCHES = {
    0: "⠙",   # C
    1: "⠑",   # C#
    2: "⠋",   # D
    3: "⠛",   # D#
    4: "⠓",   # E
    5: "⠊",   # F
    6: "⠚",   # F#
    7: "⠒",   # G
    8: "⠎",   # G#
    9: "⠞",   # A
    10: "⠖",  # A#
    11: "⠶",  # B
}

_BRAILLE_DURATIONS = {
    4.0: "⠸",   # whole
    2.0: "⠘",   # half
    1.0: "⠐",   # quarter
    0.5: "⠰",   # eighth
    0.25: "⠠",  # sixteenth
}

_CHROMATIC_COLORS = {
    0: (255, 0, 0),      # C - red
    1: (255, 69, 0),     # C# - orange-red
    2: (255, 165, 0),    # D - orange
    3: (255, 215, 0),    # D# - gold
    4: (255, 255, 0),    # E - yellow
    5: (0, 128, 0),      # F - green
    6: (0, 206, 209),    # F# - turquoise
    7: (0, 0, 255),      # G - blue
    8: (75, 0, 130),     # G# - indigo
    9: (148, 0, 211),    # A - violet
    10: (199, 21, 133),  # A# - pink
    11: (139, 69, 19),   # B - brown
}


@dataclass(frozen=True)
class BrailleMusic:
    pitch: int
    octave: int
    duration: float
    articulation: str = ""


def pitch_to_braille(midi_pitch: int) -> str:
    pc = midi_pitch % 12
    return _BRAILLE_PITCHES.get(pc, "⠀")


def duration_to_braille(beats: float) -> str:
    closest = min(_BRAILLE_DURATIONS.keys(), key=lambda d: abs(d - beats))
    return _BRAILLE_DURATIONS[closest]


def sequence_to_braille(pitches: list[int], durations: list[float]) -> str:
    parts = []
    for p, d in zip(pitches, durations):
        parts.append(duration_to_braille(d) + pitch_to_braille(p))
    return " ".join(parts)


@dataclass(frozen=True)
class ColorNotation:
    midi_pitch: int
    rgb: tuple[int, int, int]
    note_name: str


def pitch_to_color(midi_pitch: int) -> tuple[int, int, int]:
    pc = midi_pitch % 12
    return _CHROMATIC_COLORS[pc]


class ShapeNotation(Enum):
    CIRCLE = "whole"
    SQUARE = "half"
    TRIANGLE = "quarter"
    DIAMOND = "eighth"
    STAR = "sixteenth"


_DURATION_SHAPES = {
    4.0: ShapeNotation.CIRCLE,
    2.0: ShapeNotation.SQUARE,
    1.0: ShapeNotation.TRIANGLE,
    0.5: ShapeNotation.DIAMOND,
    0.25: ShapeNotation.STAR,
}


@dataclass
class SimplifiedStaff:
    line_count: int = 3
    note_size_multiplier: float = 2.0


def simplify_notation(
    pitches: list[int],
    durations: list[float],
    staff: SimplifiedStaff | None = None,
) -> list[dict]:
    if staff is None:
        staff = SimplifiedStaff()
    result = []
    for p, d in zip(pitches, durations):
        pc = p % 12
        octave = p // 12 - 1
        closest_dur = min(_DURATION_SHAPES.keys(), key=lambda k: abs(k - d))
        result.append({
            "note_name": _NOTE_NAMES[pc],
            "octave": octave,
            "color": _CHROMATIC_COLORS[pc],
            "shape": _DURATION_SHAPES[closest_dur].name,
            "size": staff.note_size_multiplier,
            "staff_position": p % (staff.line_count * 2),
        })
    return result
