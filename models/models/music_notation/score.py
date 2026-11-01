"""Score representation: notes, rests, measures, transposition."""
from __future__ import annotations
from dataclasses import dataclass, field

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass
class Note:
    pitch_class: str
    octave: int = 4
    duration: float = 1.0
    velocity: int = 80

    @property
    def midi(self) -> int:
        idx = PITCH_CLASSES.index(self.pitch_class) if self.pitch_class in PITCH_CLASSES else 0
        return idx + 12 * (self.octave + 1)


@dataclass
class Rest:
    duration: float = 1.0


@dataclass
class Measure:
    events: list[Note | Rest] = field(default_factory=list)
    time_sig: tuple[int, int] = (4, 4)
    key_sig: str = "C"

    @property
    def total_duration(self) -> float:
        return sum(e.duration for e in self.events)


@dataclass
class Score:
    parts: list[list[Measure]] = field(default_factory=list)
    title: str = ""
    tempo: int = 120

    def to_pitch_sequence(self, part_idx: int = 0) -> list[int]:
        if part_idx >= len(self.parts):
            return []
        pitches = []
        for measure in self.parts[part_idx]:
            for event in measure.events:
                if isinstance(event, Note):
                    pitches.append(event.midi)
        return pitches

    def to_interval_sequence(self, part_idx: int = 0) -> list[int]:
        pitches = self.to_pitch_sequence(part_idx)
        return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]

    def total_duration(self, part_idx: int = 0) -> float:
        if part_idx >= len(self.parts):
            return 0.0
        return sum(m.total_duration for m in self.parts[part_idx])


def transpose_score(score: Score, semitones: int) -> Score:
    new_parts = []
    for part in score.parts:
        new_measures = []
        for measure in part:
            new_events = []
            for event in measure.events:
                if isinstance(event, Note):
                    midi = event.midi + semitones
                    new_pc = PITCH_CLASSES[midi % 12]
                    new_oct = (midi // 12) - 1
                    new_events.append(Note(new_pc, new_oct, event.duration, event.velocity))
                else:
                    new_events.append(Rest(event.duration))
            new_measures.append(Measure(new_events, measure.time_sig, measure.key_sig))
        new_parts.append(new_measures)
    return Score(new_parts, score.title, score.tempo)
