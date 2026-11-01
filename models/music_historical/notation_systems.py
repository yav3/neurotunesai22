"""Historical notation systems: neume, mensural, tablature, shape-note.

Provides parsers and converters for historical Western notation systems,
mapping between them and modern pitch/duration representations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class NotationEra(Enum):
    NEUMATIC = "neumatic"
    SQUARE = "square"
    MENSURAL_WHITE = "mensural_white"
    MENSURAL_BLACK = "mensural_black"
    MODERN = "modern"


class MensuralDuration(Enum):
    MAXIMA = 8.0
    LONGA = 4.0
    BREVIS = 2.0
    SEMIBREVIS = 1.0
    MINIMA = 0.5
    SEMIMINIMA = 0.25
    FUSA = 0.125
    SEMIFUSA = 0.0625


class Mensuration(Enum):
    TEMPUS_PERFECTUM_PROLATIO_MAJOR = "O."
    TEMPUS_PERFECTUM_PROLATIO_MINOR = "O"
    TEMPUS_IMPERFECTUM_PROLATIO_MAJOR = "C."
    TEMPUS_IMPERFECTUM_PROLATIO_MINOR = "C"


@dataclass
class MensuralNote:
    duration: MensuralDuration
    pitch_name: str
    octave: int
    colored: bool = False
    dotted: bool = False

    @property
    def beats(self) -> float:
        base = self.duration.value
        if self.dotted:
            base *= 1.5
        if self.colored:
            base *= 2.0 / 3.0
        return base


@dataclass
class NeumeGroup:
    name: str
    pitches: Tuple[int, ...]
    ascending: bool = True

    @property
    def interval_sequence(self) -> Tuple[int, ...]:
        return tuple(self.pitches[i + 1] - self.pitches[i]
                     for i in range(len(self.pitches) - 1))


NEUME_CATALOG: Dict[str, NeumeGroup] = {
    "punctum": NeumeGroup("punctum", (0,)),
    "virga": NeumeGroup("virga", (0,)),
    "podatus": NeumeGroup("podatus", (0, 2), ascending=True),
    "clivis": NeumeGroup("clivis", (2, 0), ascending=False),
    "scandicus": NeumeGroup("scandicus", (0, 2, 4), ascending=True),
    "climacus": NeumeGroup("climacus", (4, 2, 0), ascending=False),
    "torculus": NeumeGroup("torculus", (0, 2, 0)),
    "porrectus": NeumeGroup("porrectus", (2, 0, 2)),
}


@dataclass
class ShapeNote:
    shape: str
    solmization: str
    scale_degree: int

    @property
    def pitch_class(self) -> int:
        _map = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
        return _map.get(self.scale_degree, 0)


SHAPE_NOTE_FOUR: Dict[str, ShapeNote] = {
    "fa": ShapeNote("triangle", "fa", 1),
    "sol": ShapeNote("circle", "sol", 2),
    "la": ShapeNote("square", "la", 3),
    "mi": ShapeNote("diamond", "mi", 7),
}

SHAPE_NOTE_SEVEN: Dict[str, ShapeNote] = {
    "do": ShapeNote("triangle", "do", 1),
    "re": ShapeNote("crescent", "re", 2),
    "mi": ShapeNote("diamond", "mi", 3),
    "fa": ShapeNote("triangle_right", "fa", 4),
    "sol": ShapeNote("circle", "sol", 5),
    "la": ShapeNote("square", "la", 6),
    "si": ShapeNote("diamond_half", "si", 7),
}


def mensuration_ratio(m: Mensuration) -> Tuple[int, int]:
    """Return (tempus, prolatio) division ratios for a mensuration sign."""
    if m == Mensuration.TEMPUS_PERFECTUM_PROLATIO_MAJOR:
        return (3, 3)
    elif m == Mensuration.TEMPUS_PERFECTUM_PROLATIO_MINOR:
        return (3, 2)
    elif m == Mensuration.TEMPUS_IMPERFECTUM_PROLATIO_MAJOR:
        return (2, 3)
    else:
        return (2, 2)


def mensural_to_modern_duration(note: MensuralNote,
                                mensuration: Mensuration) -> float:
    """Convert mensural note to modern duration in quarter-note units."""
    tempus, prolatio = mensuration_ratio(mensuration)
    base = note.beats
    if note.duration in (MensuralDuration.BREVIS, MensuralDuration.LONGA,
                         MensuralDuration.MAXIMA):
        base *= tempus / 2.0
    if note.duration == MensuralDuration.SEMIBREVIS:
        base *= prolatio / 2.0
    return base


def neume_to_intervals(name: str) -> Tuple[int, ...]:
    """Look up a neume's interval sequence."""
    ng = NEUME_CATALOG.get(name)
    if ng is None:
        return ()
    return ng.interval_sequence


def intervals_to_neume(intervals: Tuple[int, ...]) -> Optional[str]:
    """Find the neume name matching an interval sequence."""
    for name, ng in NEUME_CATALOG.items():
        if ng.interval_sequence == intervals:
            return name
    return None


@dataclass
class TabNote:
    string: int
    fret: int
    duration: Optional[float] = None

    def to_midi(self, tuning: Tuple[int, ...] = (40, 45, 50, 55, 59, 64)) -> int:
        if self.string < len(tuning):
            return tuning[self.string] + self.fret
        return 60 + self.fret


def parse_tab_line(line: str,
                   tuning: Tuple[int, ...] = (40, 45, 50, 55, 59, 64)
                   ) -> List[TabNote]:
    """Parse a simple ASCII guitar tab line (e.g. 'e|--0--2--3--')."""
    notes = []
    string_idx = 0
    for ch in line.split('|')[-1] if '|' in line else line:
        if ch.isdigit():
            notes.append(TabNote(string=string_idx, fret=int(ch)))
    return notes


def tab_to_pitches(tab_notes: List[TabNote],
                   tuning: Tuple[int, ...] = (40, 45, 50, 55, 59, 64)
                   ) -> List[int]:
    """Convert tab notes to MIDI pitch numbers."""
    return [n.to_midi(tuning) for n in tab_notes]


def notation_complexity(era: NotationEra) -> float:
    """Rough measure of notational precision (0=least, 1=most)."""
    _scores = {
        NotationEra.NEUMATIC: 0.2,
        NotationEra.SQUARE: 0.4,
        NotationEra.MENSURAL_BLACK: 0.6,
        NotationEra.MENSURAL_WHITE: 0.7,
        NotationEra.MODERN: 1.0,
    }
    return _scores.get(era, 0.5)


def era_pitch_precision(era: NotationEra) -> float:
    """Approximate pitch precision in semitones for each notation era."""
    _precision = {
        NotationEra.NEUMATIC: 3.0,
        NotationEra.SQUARE: 1.0,
        NotationEra.MENSURAL_BLACK: 1.0,
        NotationEra.MENSURAL_WHITE: 1.0,
        NotationEra.MODERN: 0.5,
    }
    return _precision.get(era, 1.0)
