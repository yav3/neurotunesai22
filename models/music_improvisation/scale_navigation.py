"""Scale and mode navigation for improvisation."""

from __future__ import annotations

from enum import Enum


class ScaleMode(Enum):
    IONIAN = "ionian"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"
    MELODIC_MINOR = "melodic_minor"
    HARMONIC_MINOR = "harmonic_minor"
    WHOLE_TONE = "whole_tone"
    DIMINISHED_HW = "diminished_half_whole"
    DIMINISHED_WH = "diminished_whole_half"
    PENTATONIC_MAJOR = "pentatonic_major"
    PENTATONIC_MINOR = "pentatonic_minor"
    BLUES = "blues"
    BEBOP_DOMINANT = "bebop_dominant"
    ALTERED = "altered"
    LYDIAN_DOMINANT = "lydian_dominant"


_INTERVALS: dict[ScaleMode, list[int]] = {
    ScaleMode.IONIAN:           [0, 2, 4, 5, 7, 9, 11],
    ScaleMode.DORIAN:           [0, 2, 3, 5, 7, 9, 10],
    ScaleMode.PHRYGIAN:         [0, 1, 3, 5, 7, 8, 10],
    ScaleMode.LYDIAN:           [0, 2, 4, 6, 7, 9, 11],
    ScaleMode.MIXOLYDIAN:       [0, 2, 4, 5, 7, 9, 10],
    ScaleMode.AEOLIAN:          [0, 2, 3, 5, 7, 8, 10],
    ScaleMode.LOCRIAN:          [0, 1, 3, 5, 6, 8, 10],
    ScaleMode.MELODIC_MINOR:    [0, 2, 3, 5, 7, 9, 11],
    ScaleMode.HARMONIC_MINOR:   [0, 2, 3, 5, 7, 8, 11],
    ScaleMode.WHOLE_TONE:       [0, 2, 4, 6, 8, 10],
    ScaleMode.DIMINISHED_HW:    [0, 1, 3, 4, 6, 7, 9, 10],
    ScaleMode.DIMINISHED_WH:    [0, 2, 3, 5, 6, 8, 9, 11],
    ScaleMode.PENTATONIC_MAJOR: [0, 2, 4, 7, 9],
    ScaleMode.PENTATONIC_MINOR: [0, 3, 5, 7, 10],
    ScaleMode.BLUES:            [0, 3, 5, 6, 7, 10],
    ScaleMode.BEBOP_DOMINANT:   [0, 2, 4, 5, 7, 9, 10, 11],
    ScaleMode.ALTERED:          [0, 1, 3, 4, 6, 8, 10],
    ScaleMode.LYDIAN_DOMINANT:  [0, 2, 4, 6, 7, 9, 10],
}


def scale_degrees(root: int, mode: ScaleMode) -> list[int]:
    return [(root + iv) % 12 for iv in _INTERVALS[mode]]


def chord_tones_in_scale(chord_pitches: list[int], root: int, mode: ScaleMode) -> list[bool]:
    sc = set(scale_degrees(root, mode))
    return [p % 12 in sc for p in chord_pitches]


_TENSION_MAP: dict[str, dict[ScaleMode, list[int]]] = {
    "maj7": {
        ScaleMode.IONIAN: [9, 11, 6],
        ScaleMode.LYDIAN: [9, 6, 11],
    },
    "dom7": {
        ScaleMode.MIXOLYDIAN: [9, 13],
        ScaleMode.LYDIAN_DOMINANT: [6, 9, 13],
        ScaleMode.ALTERED: [1, 3, 6, 8],
    },
    "min7": {
        ScaleMode.DORIAN: [9, 11],
        ScaleMode.AEOLIAN: [9, 11],
        ScaleMode.PHRYGIAN: [1, 8],
    },
    "min7b5": {
        ScaleMode.LOCRIAN: [1, 8, 10],
    },
}


def available_tensions(chord_quality: str, mode: ScaleMode) -> list[int]:
    qual = _TENSION_MAP.get(chord_quality, {})
    return qual.get(mode, [])


def approach_note_options(target: int) -> dict[str, list[int]]:
    return {
        "chromatic_below": [(target - 1) % 12],
        "chromatic_above": [(target + 1) % 12],
        "diatonic_below": [(target - 2) % 12],
        "diatonic_above": [(target + 2) % 12],
        "double_chromatic": [(target - 1) % 12, (target + 1) % 12],
        "enclosure": [(target + 1) % 12, (target - 1) % 12],
    }


def target_note_gravity(note: int, chord_root: int, chord_pitches: list[int]) -> float:
    pc = note % 12
    cr = chord_root % 12
    if pc == cr:
        return 1.0
    if pc in {p % 12 for p in chord_pitches}:
        return 0.7
    min_dist = min(min(abs(pc - (p % 12)), 12 - abs(pc - (p % 12))) for p in chord_pitches)
    return max(0.0, 1.0 - min_dist * 0.2)
