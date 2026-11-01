"""Music theory module — scales, intervals, chords, progressions, Camelot wheel, rhythm.

Pure music theory primitives for the music vertical. No ML, no medical code.
"""
from __future__ import annotations

from models.music_theory.intervals import (
    CONSONANCE_SCORES,
    ENHARMONIC_MAP,
    INTERVAL_NAMES,
    INTERVAL_TABLE,
    Interval,
    IntervalCalculator,
    PITCH_CLASSES,
    consonance_score,
    interval_between,
    interval_name,
    melodic_tension,
    normalize_note,
    note_to_index,
)
from models.music_theory.scales import (
    ALL_SCALE_TYPES,
    SCALE_INTERVALS,
    Scale,
    ScaleLibrary,
    detect_scale_from_notes,
    get_relative_major,
    get_relative_minor,
    get_scale,
    scale_degree_to_note,
)
from models.music_theory.chords import (
    CHORD_FORMULAS,
    Chord,
    ChordLibrary,
)
from models.music_theory.progressions import (
    COMMON_PROGRESSIONS,
    Progression,
    ProgressionAnalyzer,
    RomanNumeral,
)
from models.music_theory.camelot import (
    CamelotKey,
    camelot_compatibility,
    camelot_to_key,
    harmonic_mixing_score,
    key_to_camelot,
    suggest_transitions,
)
from models.music_theory.rhythm import (
    COMMON_METERS,
    GROOVE_TEMPLATES,
    RhythmAnalyzer,
    RhythmPattern,
    TimeSignature,
)

__all__ = [
    "PITCH_CLASSES", "ENHARMONIC_MAP", "INTERVAL_NAMES", "CONSONANCE_SCORES",
    "Interval", "INTERVAL_TABLE", "IntervalCalculator",
    "normalize_note", "note_to_index", "interval_between", "interval_name",
    "consonance_score", "melodic_tension",
    "Scale", "ScaleLibrary", "SCALE_INTERVALS", "ALL_SCALE_TYPES", "get_scale",
    "get_relative_minor", "get_relative_major", "scale_degree_to_note",
    "detect_scale_from_notes",
    "Chord", "ChordLibrary", "CHORD_FORMULAS",
    "Progression", "COMMON_PROGRESSIONS", "ProgressionAnalyzer", "RomanNumeral",
    "CamelotKey", "key_to_camelot", "camelot_to_key",
    "camelot_compatibility", "harmonic_mixing_score", "suggest_transitions",
    "TimeSignature", "RhythmPattern", "COMMON_METERS", "GROOVE_TEMPLATES",
    "RhythmAnalyzer",
]
