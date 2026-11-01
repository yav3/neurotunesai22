"""LilyPond notation: generation, parsing, and conversion utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

PITCH_TO_LILY = {0: "c", 1: "cis", 2: "d", 3: "dis", 4: "e", 5: "f",
                 6: "fis", 7: "g", 8: "gis", 9: "a", 10: "ais", 11: "b"}

LILY_TO_PITCH: Dict[str, int] = {v: k for k, v in PITCH_TO_LILY.items()}

DURATION_MAP = {4.0: "1", 2.0: "2", 1.0: "4", 0.5: "8", 0.25: "16", 0.125: "32"}


class Articulation(str, Enum):
    STACCATO = "-."
    ACCENT = "->"
    TENUTO = "--"
    MARCATO = "-^"
    FERMATA = "\\fermata"


@dataclass
class LilyNote:
    pitch_class: int
    octave: int = 4
    duration_beats: float = 1.0
    is_rest: bool = False
    articulations: List[Articulation] = field(default_factory=list)
    dynamic: Optional[str] = None

    def to_lily(self) -> str:
        if self.is_rest:
            dur = DURATION_MAP.get(self.duration_beats, "4")
            return f"r{dur}"
        name = PITCH_TO_LILY.get(self.pitch_class % 12, "c")
        rel_oct = self.octave - 3
        if rel_oct > 0:
            name += "'" * rel_oct
        elif rel_oct < 0:
            name += "," * (-rel_oct)
        dur = DURATION_MAP.get(self.duration_beats, "4")
        s = f"{name}{dur}"
        for art in self.articulations:
            s += art.value
        if self.dynamic:
            s += f"\\{self.dynamic}"
        return s


@dataclass
class LilyMeasure:
    notes: List[LilyNote] = field(default_factory=list)

    def to_lily(self) -> str:
        return " ".join(n.to_lily() for n in self.notes)


@dataclass
class LilyScore:
    title: Optional[str] = None
    composer: Optional[str] = None
    tempo_bpm: int = 120
    time_sig: Tuple[int, int] = (4, 4)
    key_tonic: str = "c"
    key_mode: str = "\\major"
    measures: List[LilyMeasure] = field(default_factory=list)

    def to_lily(self) -> str:
        lines: List[str] = ['\\version "2.24.0"']
        if self.title or self.composer:
            lines.append("\\header {")
            if self.title:
                lines.append(f'  title = "{self.title}"')
            if self.composer:
                lines.append(f'  composer = "{self.composer}"')
            lines.append("}")
        lines.append("\\relative c' {")
        lines.append(f"  \\key {self.key_tonic} {self.key_mode}")
        lines.append(f"  \\time {self.time_sig[0]}/{self.time_sig[1]}")
        lines.append(f"  \\tempo 4 = {self.tempo_bpm}")
        for m in self.measures:
            lines.append(f"  {m.to_lily()} |")
        lines.append("}")
        return "\n".join(lines)


def midi_to_lily_note(midi_pitch: int, duration_beats: float = 1.0) -> LilyNote:
    return LilyNote(pitch_class=midi_pitch % 12, octave=midi_pitch // 12 - 1,
                    duration_beats=duration_beats)


def notes_to_score(notes: List[Tuple[int, float]], beats_per_measure: int = 4,
                   title: Optional[str] = None) -> LilyScore:
    lily_notes = [midi_to_lily_note(p, d) for p, d in notes]
    measures: List[LilyMeasure] = []
    current: List[LilyNote] = []
    beat_sum = 0.0
    for ln in lily_notes:
        current.append(ln)
        beat_sum += ln.duration_beats
        if beat_sum >= beats_per_measure:
            measures.append(LilyMeasure(notes=current))
            current = []
            beat_sum = 0.0
    if current:
        measures.append(LilyMeasure(notes=current))
    return LilyScore(title=title, measures=measures)


def parse_lily_pitch(token: str) -> Optional[Tuple[int, int]]:
    base = ""
    rest = token
    for name, pc in sorted(LILY_TO_PITCH.items(), key=lambda x: -len(x[0])):
        if rest.startswith(name):
            base = name
            rest = rest[len(name):]
            break
    if not base:
        return None
    pc = LILY_TO_PITCH[base]
    octave = 3
    for ch in rest:
        if ch == "'":
            octave += 1
        elif ch == ",":
            octave -= 1
        else:
            break
    return (pc, octave)


def transpose_lily(score: LilyScore, semitones: int) -> LilyScore:
    new_measures: List[LilyMeasure] = []
    for m in score.measures:
        new_notes: List[LilyNote] = []
        for n in m.notes:
            if n.is_rest:
                new_notes.append(n)
            else:
                new_midi = (n.octave + 1) * 12 + n.pitch_class + semitones
                new_notes.append(LilyNote(pitch_class=new_midi % 12,
                                          octave=new_midi // 12 - 1,
                                          duration_beats=n.duration_beats,
                                          articulations=list(n.articulations),
                                          dynamic=n.dynamic))
        new_measures.append(LilyMeasure(notes=new_notes))
    tonic_pc = LILY_TO_PITCH.get(score.key_tonic, 0)
    new_tonic = (tonic_pc + semitones) % 12
    return LilyScore(title=score.title, composer=score.composer,
                     tempo_bpm=score.tempo_bpm, time_sig=score.time_sig,
                     key_tonic=PITCH_TO_LILY.get(new_tonic, "c"),
                     key_mode=score.key_mode, measures=new_measures)
