"""Score layout engine: staff spacing, beam grouping, note spacing, collision detection."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional

import numpy as np


class ClefType(str, Enum):
    TREBLE = "treble"
    BASS = "bass"
    ALTO = "alto"
    TENOR = "tenor"


class StemDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    AUTO = "auto"


@dataclass
class LayoutNote:
    midi: int
    x: float
    duration: float
    staff: int = 0
    voice: int = 0
    stem_dir: StemDirection = StemDirection.AUTO
    accidental: int = 0


@dataclass
class StaffSpec:
    clef: ClefType
    y_offset: float
    line_spacing: float = 10.0
    n_lines: int = 5


@dataclass
class BeamGroup:
    notes: List[LayoutNote]
    stem_direction: StemDirection


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def overlaps(self, other: BoundingBox) -> bool:
        return not (self.x + self.width < other.x or
                    other.x + other.width < self.x or
                    self.y + self.height < other.y or
                    other.y + other.height < self.y)


@dataclass
class PageBreak:
    measure_index: int
    system_index: int


def midi_to_staff_y(midi: int, staff: StaffSpec) -> float:
    if staff.clef == ClefType.TREBLE:
        ref_midi, ref_line = 71, 0  # B4 = top line
    elif staff.clef == ClefType.BASS:
        ref_midi, ref_line = 50, 0  # D3 = top line
    elif staff.clef == ClefType.ALTO:
        ref_midi, ref_line = 60, 2  # C4 = middle line
    else:
        ref_midi, ref_line = 57, 2  # A3 = middle line
    white_keys = [0, 2, 4, 5, 7, 9, 11]
    ref_oct, ref_pc = divmod(ref_midi, 12)
    note_oct, note_pc = divmod(midi, 12)
    ref_steps = ref_oct * 7 + white_keys.index(ref_pc) if ref_pc in white_keys else ref_oct * 7
    closest_pc = min(white_keys, key=lambda w: abs(w - note_pc))
    note_steps = note_oct * 7 + white_keys.index(closest_pc)
    step_diff = note_steps - ref_steps
    half_space = staff.line_spacing / 2.0
    return staff.y_offset + ref_line * staff.line_spacing - step_diff * half_space


def auto_stem_direction(notes: List[LayoutNote], staff: StaffSpec) -> StemDirection:
    if not notes:
        return StemDirection.UP
    avg_y = np.mean([midi_to_staff_y(n.midi, staff) for n in notes])
    center = staff.y_offset + 2 * staff.line_spacing
    return StemDirection.DOWN if avg_y < center else StemDirection.UP


def note_spacing(duration: float, base_width: float = 30.0,
                 exponent: float = 0.6) -> float:
    return base_width * (duration ** exponent)


def measure_width(durations: List[float], base_width: float = 30.0,
                  padding: float = 20.0) -> float:
    return sum(note_spacing(d, base_width) for d in durations) + padding


def system_line_break(measure_widths: List[float],
                      page_width: float,
                      left_margin: float = 60.0,
                      right_margin: float = 30.0) -> List[List[int]]:
    available = page_width - left_margin - right_margin
    systems: List[List[int]] = []
    current: List[int] = []
    current_width = 0.0
    for i, w in enumerate(measure_widths):
        if current and current_width + w > available:
            systems.append(current)
            current = [i]
            current_width = w
        else:
            current.append(i)
            current_width += w
    if current:
        systems.append(current)
    return systems


def page_breaks(systems: List[List[int]], systems_per_page: int = 6) -> List[PageBreak]:
    breaks: List[PageBreak] = []
    for i in range(0, len(systems), systems_per_page):
        if i > 0:
            breaks.append(PageBreak(
                measure_index=systems[i][0],
                system_index=i,
            ))
    return breaks


def beam_groups_from_beats(notes: List[LayoutNote],
                           beat_duration: float = 1.0) -> List[BeamGroup]:
    if not notes:
        return []
    groups: List[BeamGroup] = []
    current: List[LayoutNote] = [notes[0]]
    current_beat = math.floor(notes[0].x / beat_duration)
    for n in notes[1:]:
        beat = math.floor(n.x / beat_duration)
        if beat != current_beat or n.duration >= beat_duration:
            if len(current) >= 2 and all(c.duration < beat_duration for c in current):
                groups.append(BeamGroup(notes=current, stem_direction=StemDirection.AUTO))
            current = [n]
            current_beat = beat
        else:
            current.append(n)
    if len(current) >= 2 and all(c.duration < beat_duration for c in current):
        groups.append(BeamGroup(notes=current, stem_direction=StemDirection.AUTO))
    return groups


def note_bounding_box(note: LayoutNote, staff: StaffSpec,
                      note_head_w: float = 12.0,
                      note_head_h: float = 8.0) -> BoundingBox:
    y = midi_to_staff_y(note.midi, staff)
    return BoundingBox(
        x=note.x - note_head_w / 2,
        y=y - note_head_h / 2,
        width=note_head_w,
        height=note_head_h,
    )


def detect_collisions(notes: List[LayoutNote],
                      staff: StaffSpec) -> List[Tuple[int, int]]:
    boxes = [(i, note_bounding_box(n, staff)) for i, n in enumerate(notes)]
    collisions: List[Tuple[int, int]] = []
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if boxes[a][1].overlaps(boxes[b][1]):
                collisions.append((a, b))
    return collisions


def resolve_collisions(notes: List[LayoutNote], staff: StaffSpec,
                       shift: float = 14.0) -> List[LayoutNote]:
    result = [LayoutNote(**n.__dict__) for n in notes]
    collisions = detect_collisions(result, staff)
    shifted: set = set()
    for a, b in collisions:
        if b not in shifted:
            result[b].x += shift
            shifted.add(b)
    return result


def staff_system_height(staves: List[StaffSpec],
                        inter_staff_gap: float = 40.0) -> float:
    if not staves:
        return 0.0
    total = 0.0
    for i, s in enumerate(staves):
        total += s.line_spacing * (s.n_lines - 1)
        if i < len(staves) - 1:
            total += inter_staff_gap
    return total


def ledger_line_count(midi: int, staff: StaffSpec) -> int:
    y = midi_to_staff_y(midi, staff)
    top = staff.y_offset
    bottom = staff.y_offset + (staff.n_lines - 1) * staff.line_spacing
    half = staff.line_spacing / 2.0
    if y < top:
        return max(0, int(math.ceil((top - y) / staff.line_spacing)))
    elif y > bottom:
        return max(0, int(math.ceil((y - bottom) / staff.line_spacing)))
    return 0


def optimal_clef(midi_values: List[int]) -> ClefType:
    if not midi_values:
        return ClefType.TREBLE
    avg = np.mean(midi_values)
    if avg < 48:
        return ClefType.BASS
    elif avg < 55:
        return ClefType.TENOR
    elif avg < 62:
        return ClefType.ALTO
    return ClefType.TREBLE
