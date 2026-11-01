"""Staff layout engine: compute vertical positions for notation rendering."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Clef(Enum):
    TREBLE = "treble"
    BASS = "bass"
    ALTO = "alto"
    TENOR = "tenor"


CLEF_MIDDLE_LINE_MIDI = {
    Clef.TREBLE: 71,
    Clef.BASS: 50,
    Clef.ALTO: 60,
    Clef.TENOR: 57,
}

STEP_DIATONIC = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}

STAFF_LINE_SPACING = 10.0
LEDGER_LINE_EXTENSION = 6.0


def _diatonic_position(step: str, octave: int) -> int:
    return octave * 7 + STEP_DIATONIC.get(step.upper(), 0)


def _midi_to_diatonic(midi: int) -> tuple[str, int, int]:
    octave = (midi // 12) - 1
    pc = midi % 12
    natural_map = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}
    sharp_map = {1: ("C", 1), 3: ("D", 1), 6: ("F", 1), 8: ("G", 1), 10: ("A", 1)}
    if pc in natural_map:
        return natural_map[pc], octave, 0
    step, acc = sharp_map[pc]
    return step, octave, acc


@dataclass
class StaffLine:
    y_position: float = 0.0
    clef: Clef = Clef.TREBLE
    key_sig: str = "C"
    time_sig: tuple[int, int] = (4, 4)
    line_positions: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.line_positions:
            self.line_positions = [
                self.y_position + i * STAFF_LINE_SPACING for i in range(5)
            ]

    def middle_line_y(self) -> float:
        return self.line_positions[2]

    def top_line_y(self) -> float:
        return self.line_positions[4]

    def bottom_line_y(self) -> float:
        return self.line_positions[0]


@dataclass
class NoteHead:
    pitch: str = "C"
    octave: int = 4
    accidental: int = 0
    x_position: float = 0.0
    y_position: float = 0.0
    duration_type: str = "quarter"
    stem_direction: str = "up"
    ledger_lines: list[float] = field(default_factory=list)
    midi: int = 60

    @property
    def filled(self) -> bool:
        return self.duration_type not in ("whole", "half")


@dataclass
class Beam:
    note_indices: list[int] = field(default_factory=list)
    beam_count: int = 1

    def length(self) -> int:
        return len(self.note_indices)


class StaffLayout:
    def __init__(self, clef: Clef = Clef.TREBLE, spacing_x: float = 30.0):
        self.clef = clef
        self.spacing_x = spacing_x
        self._ref_diatonic = self._clef_reference_diatonic()

    def _clef_reference_diatonic(self) -> int:
        mid_midi = CLEF_MIDDLE_LINE_MIDI[self.clef]
        step, octave, _ = _midi_to_diatonic(mid_midi)
        return _diatonic_position(step, octave)

    def compute_y(self, step: str, octave: int, staff: StaffLine) -> float:
        note_dia = _diatonic_position(step, octave)
        offset = note_dia - self._ref_diatonic
        half_space = STAFF_LINE_SPACING / 2.0
        return staff.middle_line_y() + offset * half_space

    def needs_ledger_lines(self, y: float, staff: StaffLine) -> list[float]:
        ledgers = []
        half = STAFF_LINE_SPACING / 2.0
        if y > staff.top_line_y() + half * 0.5:
            pos = staff.top_line_y() + STAFF_LINE_SPACING
            while pos <= y + half * 0.5:
                ledgers.append(pos)
                pos += STAFF_LINE_SPACING
        elif y < staff.bottom_line_y() - half * 0.5:
            pos = staff.bottom_line_y() - STAFF_LINE_SPACING
            while pos >= y - half * 0.5:
                ledgers.append(pos)
                pos -= STAFF_LINE_SPACING
        return ledgers

    def stem_direction(self, y: float, staff: StaffLine) -> str:
        return "down" if y >= staff.middle_line_y() else "up"

    def layout_notes(
        self,
        notes: list[tuple[str, int, int, str]],
        staff: StaffLine,
        start_x: float = 50.0,
    ) -> list[NoteHead]:
        heads = []
        x = start_x
        for step, octave, accidental, dur_type in notes:
            y = self.compute_y(step, octave, staff)
            ledgers = self.needs_ledger_lines(y, staff)
            stem = self.stem_direction(y, staff)
            from .abc_parser import SEMITONE_MAP
            midi = 12 * (octave + 1) + SEMITONE_MAP.get(step.upper(), 0) + accidental
            head = NoteHead(
                pitch=step,
                octave=octave,
                accidental=accidental,
                x_position=x,
                y_position=y,
                duration_type=dur_type,
                stem_direction=stem,
                ledger_lines=ledgers,
                midi=midi,
            )
            heads.append(head)
            x += self.spacing_x
        return heads

    @staticmethod
    def group_beams(
        heads: list[NoteHead], beam_grouping: int = 2
    ) -> list[Beam]:
        beamable = {"eighth", "16th", "32nd", "64th"}
        beam_counts = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4}
        beams: list[Beam] = []
        current: list[int] = []
        current_bc = 1

        for i, h in enumerate(heads):
            if h.duration_type in beamable:
                current.append(i)
                current_bc = max(current_bc, beam_counts.get(h.duration_type, 1))
                if len(current) >= beam_grouping:
                    beams.append(Beam(note_indices=current[:], beam_count=current_bc))
                    current = []
                    current_bc = 1
            else:
                if len(current) >= 2:
                    beams.append(Beam(note_indices=current[:], beam_count=current_bc))
                current = []
                current_bc = 1

        if len(current) >= 2:
            beams.append(Beam(note_indices=current[:], beam_count=current_bc))

        return beams
