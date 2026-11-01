"""Scale construction, detection, and key relationships."""
from __future__ import annotations

from dataclasses import dataclass

from models.music_theory.intervals import PITCH_CLASSES, ENHARMONIC_MAP, note_to_index


SCALE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major":             (0, 2, 4, 5, 7, 9, 11),
    "natural_minor":     (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor":    (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor":     (0, 2, 3, 5, 7, 9, 11),
    "dorian":            (0, 2, 3, 5, 7, 9, 10),
    "phrygian":          (0, 1, 3, 5, 7, 8, 10),
    "lydian":            (0, 2, 4, 6, 7, 9, 11),
    "mixolydian":        (0, 2, 4, 5, 7, 9, 10),
    "aeolian":           (0, 2, 3, 5, 7, 8, 10),
    "locrian":           (0, 1, 3, 5, 6, 8, 10),
    "pentatonic_major":  (0, 2, 4, 7, 9),
    "pentatonic_minor":  (0, 3, 5, 7, 10),
    "blues":             (0, 3, 5, 6, 7, 10),
    "whole_tone":        (0, 2, 4, 6, 8, 10),
    "diminished":        (0, 1, 3, 4, 6, 7, 9, 10),
    "chromatic":         tuple(range(12)),
    "bebop_dominant":    (0, 2, 4, 5, 7, 9, 10, 11),
    "bebop_major":       (0, 2, 4, 5, 7, 8, 9, 11),
    "hungarian_minor":   (0, 2, 3, 6, 7, 8, 11),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),
    "double_harmonic":   (0, 1, 4, 5, 7, 8, 11),
    "enigmatic":         (0, 1, 4, 6, 8, 10, 11),
    "neapolitan_minor":  (0, 1, 3, 5, 7, 8, 11),
    "neapolitan_major":  (0, 1, 3, 5, 7, 9, 11),
    "hirajoshi":         (0, 4, 6, 7, 11),
}

ALL_SCALE_TYPES: list[str] = list(SCALE_INTERVALS.keys())

CHROMATIC_NOTES = PITCH_CLASSES  # alias for compatibility

KEY_SIGNATURES: dict[str, int] = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6,
}

CHURCH_MODES: tuple[str, ...] = (
    "major", "dorian", "phrygian", "lydian",
    "mixolydian", "aeolian", "locrian",
)


@dataclass(frozen=True)
class Scale:
    root: str
    scale_type: str
    intervals: tuple[int, ...]
    notes: tuple[str, ...]


def get_scale(root: str, scale_type: str = "major") -> Scale:
    if scale_type not in SCALE_INTERVALS:
        raise ValueError(f"Unknown scale type: {scale_type}")
    intervals = SCALE_INTERVALS[scale_type]
    root_idx = note_to_index(root)
    notes = tuple(PITCH_CLASSES[(root_idx + i) % 12] for i in intervals)
    return Scale(root=root, scale_type=scale_type, intervals=intervals, notes=notes)


def scale_degree_to_note(root: str, scale_type: str, degree: int) -> str:
    scale = get_scale(root, scale_type)
    idx = (degree - 1) % len(scale.notes)
    return scale.notes[idx]


def get_relative_minor(major_key: str) -> str:
    idx = note_to_index(major_key)
    return PITCH_CLASSES[(idx + 9) % 12]


def get_relative_major(minor_key: str) -> str:
    idx = note_to_index(minor_key)
    return PITCH_CLASSES[(idx + 3) % 12]


def detect_scale_from_notes(notes: list[str]) -> list[tuple[str, str, float]]:
    if not notes:
        return []
    input_pcs: set[int] = {note_to_index(n) for n in notes}
    results: list[tuple[str, str, float]] = []
    for root_idx in range(12):
        root_name = PITCH_CLASSES[root_idx]
        for st, intervals in SCALE_INTERVALS.items():
            scale_pcs = {(root_idx + i) % 12 for i in intervals}
            matched = len(input_pcs & scale_pcs)
            if matched == 0:
                continue
            coverage = matched / max(len(input_pcs), len(scale_pcs))
            results.append((root_name, st, round(coverage, 4)))
    results.sort(key=lambda x: -x[2])
    return results


class ScaleLibrary:
    get_scale = staticmethod(get_scale)
    all_scales = staticmethod(lambda: [
        get_scale(r, st) for r in PITCH_CLASSES for st in SCALE_INTERVALS
    ])
    get_modes = staticmethod(lambda root: [get_scale(root, m) for m in CHURCH_MODES])
    relative_minor = staticmethod(get_relative_minor)
    relative_major = staticmethod(get_relative_major)
    enharmonic_equivalent = staticmethod(lambda note: ENHARMONIC_MAP.get(note, note))
    key_signature = staticmethod(lambda key: KEY_SIGNATURES.get(
        ENHARMONIC_MAP.get(key, key), KEY_SIGNATURES.get(key, 0)))
