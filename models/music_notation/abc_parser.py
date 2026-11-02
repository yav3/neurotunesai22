"""ABC notation parser and writer."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


NOTE_NAMES = "CDEFGAB"
SEMITONE_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

KEY_ACCIDENTALS: dict[str, dict[str, int]] = {
    "C": {}, "G": {"F": 1}, "D": {"F": 1, "C": 1},
    "A": {"F": 1, "C": 1, "G": 1}, "E": {"F": 1, "C": 1, "G": 1, "D": 1},
    "B": {"F": 1, "C": 1, "G": 1, "D": 1, "A": 1},
    "F#": {"F": 1, "C": 1, "G": 1, "D": 1, "A": 1, "E": 1},
    "F": {"B": -1}, "Bb": {"B": -1, "E": -1},
    "Eb": {"B": -1, "E": -1, "A": -1},
    "Ab": {"B": -1, "E": -1, "A": -1, "D": -1},
    "Db": {"B": -1, "E": -1, "A": -1, "D": -1, "G": -1},
    "Gb": {"B": -1, "E": -1, "A": -1, "D": -1, "G": -1, "C": -1},
}


@dataclass
class ABCHeader:
    title: str = ""
    composer: str = ""
    meter: str = "4/4"
    key: str = "C"
    tempo: int = 120
    reference_number: int = 1
    unit_note_length: str = "1/8"

    def meter_tuple(self) -> tuple[int, int]:
        parts = self.meter.split("/")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return 4, 4

    def unit_fraction(self) -> float:
        parts = self.unit_note_length.split("/")
        if len(parts) == 2:
            return int(parts[0]) / int(parts[1])
        return 0.125


@dataclass
class ABCNote:
    pitch: str = "C"
    duration: float = 1.0
    accidental: int = 0
    octave: int = 4
    is_rest: bool = False
    is_chord: bool = False
    chord_notes: list[ABCNote] = field(default_factory=list)

    def midi_number(self) -> int:
        if self.is_rest:
            return -1
        base = SEMITONE_MAP.get(self.pitch.upper(), 0)
        return 12 * (self.octave + 1) + base + self.accidental

    def frequency(self, a4: float = 440.0) -> float:
        if self.is_rest:
            return 0.0
        midi = self.midi_number()
        return a4 * (2.0 ** ((midi - 69) / 12.0))


@dataclass
class ABCTune:
    header: ABCHeader = field(default_factory=ABCHeader)
    notes: list[ABCNote] = field(default_factory=list)
    bars: list[list[ABCNote]] = field(default_factory=list)

    def total_duration(self) -> float:
        return sum(n.duration for n in self.notes)

    def pitch_range(self) -> tuple[int, int]:
        midis = [n.midi_number() for n in self.notes if not n.is_rest]
        if not midis:
            return 0, 0
        return min(midis), max(midis)

    def note_count(self) -> int:
        return sum(1 for n in self.notes if not n.is_rest)

    def rest_count(self) -> int:
        return sum(1 for n in self.notes if n.is_rest)


class ABCParser:
    _NOTE_RE = re.compile(
        r"(\[?)([_=^]{0,2})([A-Ga-g])([,\']*)([\d]*/?[\d]*)(\]?)"
    )
    _REST_RE = re.compile(r"[zZ]([\d]*/?[\d]*)")
    _HEADER_RE = re.compile(r"^([A-Za-z]):\s*(.+)$", re.MULTILINE)

    def parse(self, abc_text: str) -> ABCTune:
        tune = ABCTune()
        tune.header = self._parse_headers(abc_text)
        body = self._extract_body(abc_text)
        tune.notes = self._parse_body(body, tune.header)
        tune.bars = self._split_bars(tune.notes, body)
        return tune

    def _parse_headers(self, text: str) -> ABCHeader:
        hdr = ABCHeader()
        for m in self._HEADER_RE.finditer(text):
            tag, value = m.group(1), m.group(2).strip()
            if tag == "T":
                hdr.title = value
            elif tag == "C":
                hdr.composer = value
            elif tag == "M":
                hdr.meter = value
            elif tag == "K":
                hdr.key = value
            elif tag == "Q":
                try:
                    hdr.tempo = int(re.search(r"\d+", value).group())
                except (AttributeError, ValueError):
                    pass
            elif tag == "X":
                try:
                    hdr.reference_number = int(value)
                except ValueError:
                    pass
            elif tag == "L":
                hdr.unit_note_length = value
        return hdr

    def _extract_body(self, text: str) -> str:
        lines = text.strip().split("\n")
        body_lines = []
        past_key = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("K:"):
                past_key = True
                continue
            if past_key and not (len(stripped) >= 2 and stripped[1] == ":" and stripped[0].isalpha()):
                body_lines.append(stripped)
            elif not past_key:
                continue
        return " ".join(body_lines)

    def _parse_duration(self, dur_str: str, unit: float) -> float:
        if not dur_str:
            return unit
        if "/" in dur_str:
            parts = dur_str.split("/")
            num = int(parts[0]) if parts[0] else 1
            den = int(parts[1]) if len(parts) > 1 and parts[1] else 2
            return unit * num / den
        return unit * int(dur_str)

    def _parse_body(self, body: str, header: ABCHeader) -> list[ABCNote]:
        notes: list[ABCNote] = []
        unit = header.unit_fraction()
        pos = 0
        while pos < len(body):
            rest_m = self._REST_RE.match(body, pos)
            if rest_m:
                dur = self._parse_duration(rest_m.group(1), unit)
                notes.append(ABCNote(pitch="z", duration=dur, is_rest=True))
                pos = rest_m.end()
                continue

            note_m = self._NOTE_RE.match(body, pos)
            if note_m:
                bracket_open = note_m.group(1)
                acc_str = note_m.group(2)
                pitch_char = note_m.group(3)
                octave_str = note_m.group(4)
                dur_str = note_m.group(5)
                bracket_close = note_m.group(6)

                accidental = 0
                for ch in acc_str:
                    if ch == "^":
                        accidental += 1
                    elif ch == "_":
                        accidental -= 1

                is_lower = pitch_char.islower()
                pitch_upper = pitch_char.upper()
                octave = 5 if is_lower else 4
                for ch in octave_str:
                    if ch == "'":
                        octave += 1
                    elif ch == ",":
                        octave -= 1

                dur = self._parse_duration(dur_str, unit)
                note = ABCNote(
                    pitch=pitch_upper,
                    duration=dur,
                    accidental=accidental,
                    octave=octave,
                    is_rest=False,
                )
                notes.append(note)
                pos = note_m.end()
                continue

            pos += 1

        return notes

    def _split_bars(self, notes: list[ABCNote], body: str) -> list[list[ABCNote]]:
        if not notes:
            return []
        bars: list[list[ABCNote]] = [[]]
        bar_positions = [i for i, ch in enumerate(body) if ch == "|"]
        note_idx = 0
        char_idx = 0
        bar_idx = 0
        for note in notes:
            bars[-1].append(note)
            while bar_idx < len(bar_positions) and char_idx > bar_positions[bar_idx]:
                bar_idx += 1
            char_idx += 1
        if not bars[-1]:
            bars.pop()
        return bars if bars else [notes[:]]


class ABCWriter:
    def write(self, tune: ABCTune) -> str:
        lines = []
        h = tune.header
        lines.append(f"X: {h.reference_number}")
        if h.title:
            lines.append(f"T: {h.title}")
        if h.composer:
            lines.append(f"C: {h.composer}")
        lines.append(f"M: {h.meter}")
        lines.append(f"L: {h.unit_note_length}")
        lines.append(f"Q: 1/4={h.tempo}")
        lines.append(f"K: {h.key}")

        body_parts = []
        unit = h.unit_fraction()
        for note in tune.notes:
            body_parts.append(self._note_to_abc(note, unit))
        lines.append(" ".join(body_parts))
        return "\n".join(lines) + "\n"

    def _note_to_abc(self, note: ABCNote, unit: float) -> str:
        if note.is_rest:
            dur_str = self._duration_str(note.duration, unit)
            return f"z{dur_str}"

        acc = ""
        if note.accidental > 0:
            acc = "^" * note.accidental
        elif note.accidental < 0:
            acc = "_" * abs(note.accidental)

        if note.octave >= 5:
            pitch_char = note.pitch.lower()
            oct_marks = "'" * (note.octave - 5)
        else:
            pitch_char = note.pitch.upper()
            oct_marks = "," * (4 - note.octave)

        dur_str = self._duration_str(note.duration, unit)
        return f"{acc}{pitch_char}{oct_marks}{dur_str}"

    @staticmethod
    def _duration_str(duration: float, unit: float) -> str:
        ratio = duration / unit
        if abs(ratio - 1.0) < 1e-9:
            return ""
        if abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1:
            return str(int(round(ratio)))
        if abs(ratio - 0.5) < 1e-9:
            return "/2"
        if abs(ratio - 0.25) < 1e-9:
            return "/4"
        num = round(ratio * 8)
        return f"{num}/8"
