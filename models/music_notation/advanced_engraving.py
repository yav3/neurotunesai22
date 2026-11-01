"""Advanced music notation engraving: layout optimization, collision avoidance,
beaming rules, stem direction, and optical spacing — pure numpy.

Implements Gould (2011) "Behind Bars" and Elaine Gould engraving conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


class ClefType(Enum):
    TREBLE = auto()
    BASS = auto()
    ALTO = auto()
    TENOR = auto()


class NoteValue(Enum):
    WHOLE = 1
    HALF = 2
    QUARTER = 4
    EIGHTH = 8
    SIXTEENTH = 16
    THIRTY_SECOND = 32


@dataclass
class NoteHead:
    pitch: int
    duration: NoteValue
    x: float = 0.0
    y: float = 0.0
    accidental: Optional[str] = None
    tied: bool = False
    dot: bool = False


@dataclass
class BeamGroup:
    notes: List[NoteHead] = field(default_factory=list)
    beam_level: int = 1

    @property
    def stem_up(self) -> bool:
        avg_pitch = np.mean([n.pitch for n in self.notes]) if self.notes else 71
        return avg_pitch < 71


@dataclass
class StaffLayout:
    staff_space: float = 7.0
    line_thickness: float = 0.5
    stem_length: float = 3.5
    beam_thickness: float = 0.5
    beam_spacing: float = 0.25
    note_spacing_base: float = 1.6
    clef: ClefType = ClefType.TREBLE


def pitch_to_staff_y(pitch: int, clef: ClefType = ClefType.TREBLE) -> float:
    note_in_octave = pitch % 12
    octave = pitch // 12
    diatonic_map = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
    nearest = min(diatonic_map.keys(), key=lambda k: abs(k - note_in_octave))
    diatonic = diatonic_map[nearest] + 7 * octave
    if clef == ClefType.TREBLE:
        ref = 5 * 7 + 5
    elif clef == ClefType.BASS:
        ref = 3 * 7 + 5
    elif clef == ClefType.ALTO:
        ref = 4 * 7 + 2
    else:
        ref = 4 * 7 + 5
    return (diatonic - ref) * 0.5


def stem_direction(pitches: List[int], clef: ClefType = ClefType.TREBLE) -> bool:
    ys = [pitch_to_staff_y(p, clef) for p in pitches]
    avg = np.mean(ys)
    return avg < 0.0


def optimal_stem_length(note: NoteHead, layout: StaffLayout) -> float:
    y = pitch_to_staff_y(note.pitch, layout.clef)
    base = layout.stem_length * layout.staff_space
    if abs(y) > 3.0:
        base += (abs(y) - 3.0) * 0.5 * layout.staff_space
    return base


def beam_angle(group: BeamGroup, layout: StaffLayout) -> float:
    if len(group.notes) < 2:
        return 0.0
    ys = np.array([pitch_to_staff_y(n.pitch, layout.clef) for n in group.notes])
    xs = np.array([n.x for n in group.notes])
    if xs[-1] == xs[0]:
        return 0.0
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    max_slope = 0.5
    return float(np.clip(slope, -max_slope, max_slope))


def note_spacing(durations: List[NoteValue], base: float = 1.6) -> np.ndarray:
    """Optical spacing proportional to sqrt(duration ratio) per Gould."""
    values = np.array([d.value for d in durations], dtype=float)
    ref = NoteValue.QUARTER.value
    ratios = ref / values
    spaces = base * np.sqrt(ratios)
    return np.cumsum(np.concatenate([[0], spaces[:-1]]))


def collision_detect(notes: List[NoteHead], min_gap: float = 0.8) -> List[Tuple[int, int]]:
    collisions = []
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            dx = abs(notes[i].x - notes[j].x)
            dy = abs(notes[i].y - notes[j].y)
            if dx < min_gap and dy < min_gap:
                collisions.append((i, j))
    return collisions


def resolve_collisions(notes: List[NoteHead], min_gap: float = 0.8) -> List[NoteHead]:
    result = [NoteHead(n.pitch, n.duration, n.x, n.y, n.accidental, n.tied, n.dot) for n in notes]
    for _ in range(10):
        colls = collision_detect(result, min_gap)
        if not colls:
            break
        for i, j in colls:
            result[j].x += min_gap * 0.6
    return result


def accidental_layout(notes: List[NoteHead]) -> List[float]:
    acc_notes = [(i, n) for i, n in enumerate(notes) if n.accidental]
    offsets = [0.0] * len(notes)
    acc_notes.sort(key=lambda pair: -pair[1].y)
    columns: List[List[float]] = []
    for idx, note in acc_notes:
        placed = False
        for col_idx, col_ys in enumerate(columns):
            if all(abs(note.y - cy) >= 1.0 for cy in col_ys):
                col_ys.append(note.y)
                offsets[idx] = -(col_idx + 1) * 1.2
                placed = True
                break
        if not placed:
            columns.append([note.y])
            offsets[idx] = -(len(columns)) * 1.2
    return offsets


def ledger_lines(pitch: int, clef: ClefType = ClefType.TREBLE) -> List[float]:
    y = pitch_to_staff_y(pitch, clef)
    lines = []
    if y > 2.0:
        pos = 3.0
        while pos <= y + 0.25:
            lines.append(pos)
            pos += 1.0
    elif y < -2.0:
        pos = -3.0
        while pos >= y - 0.25:
            lines.append(pos)
            pos -= 1.0
    return lines


def measure_width(notes: List[NoteHead], layout: StaffLayout) -> float:
    if not notes:
        return layout.note_spacing_base * 4
    durations = [n.duration for n in notes]
    xs = note_spacing(durations, layout.note_spacing_base)
    return float(xs[-1] + layout.note_spacing_base) * layout.staff_space


def justify_measure(notes: List[NoteHead], target_width: float, layout: StaffLayout) -> List[NoteHead]:
    if len(notes) < 2:
        return notes
    current_width = measure_width(notes, layout)
    if current_width < 1e-6:
        return notes
    scale = target_width / current_width
    result = []
    for n in notes:
        new = NoteHead(n.pitch, n.duration, n.x * scale, n.y, n.accidental, n.tied, n.dot)
        result.append(new)
    return result


def engrave_system(measures: List[List[NoteHead]], system_width: float,
                   layout: Optional[StaffLayout] = None) -> List[List[NoteHead]]:
    if layout is None:
        layout = StaffLayout()
    total = sum(measure_width(m, layout) for m in measures)
    if total < 1e-6:
        return measures
    result = []
    x_offset = 0.0
    for measure in measures:
        w = measure_width(measure, layout)
        target = (w / total) * system_width
        justified = justify_measure(measure, target, layout)
        shifted = []
        for n in justified:
            sn = NoteHead(n.pitch, n.duration, n.x + x_offset, n.y, n.accidental, n.tied, n.dot)
            shifted.append(sn)
        result.append(shifted)
        x_offset += target
    return result
