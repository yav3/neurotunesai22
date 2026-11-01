"""Braille music notation: encoding, decoding, interval signs,
octave marks, rhythmic values in braille music code.

Implements core elements of the international braille music standard.
Pure Python — no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class BrailleOctave(Enum):
    OCTAVE_1 = 1
    OCTAVE_2 = 2
    OCTAVE_3 = 3
    OCTAVE_4 = 4
    OCTAVE_5 = 5
    OCTAVE_6 = 6
    OCTAVE_7 = 7


OCTAVE_MARKS: Dict[int, str] = {
    1: "⠈",
    2: "⠘",
    3: "⠸",
    4: "⠐",
    5: "⠨",
    6: "⠰",
    7: "⠠",
}

NOTE_SIGNS_EIGHTH: Dict[str, str] = {
    "C": "⠙",
    "D": "⠑",
    "E": "⠋",
    "F": "⠛",
    "G": "⠓",
    "A": "⠊",
    "B": "⠚",
}

NOTE_SIGNS_QUARTER: Dict[str, str] = {
    "C": "⠹",
    "D": "⠱",
    "E": "⠫",
    "F": "⠻",
    "G": "⠳",
    "A": "⠪",
    "B": "⠺",
}

NOTE_SIGNS_HALF: Dict[str, str] = {
    "C": "⠝",
    "D": "⠕",
    "E": "⠏",
    "F": "⠟",
    "G": "⠗",
    "A": "⠎",
    "B": "⠞",
}

NOTE_SIGNS_WHOLE: Dict[str, str] = {
    "C": "⠽",
    "D": "⠵",
    "E": "⠯",
    "F": "⠿",
    "G": "⠷",
    "A": "⠮",
    "B": "⠾",
}

REST_SIGNS: Dict[str, str] = {
    "whole": "⠍",
    "half": "⠥",
    "quarter": "⠧",
    "eighth": "⠭",
    "sixteenth": "⠭⠣",
}

ACCIDENTAL_SIGNS: Dict[str, str] = {
    "sharp": "⠩",
    "flat": "⠣",
    "natural": "⠡",
    "double_sharp": "⠩⠩",
    "double_flat": "⠣⠣",
}

DOT_SIGN = "⠄"
TIE_SIGN = "⠈⠉"
SLUR_SIGN = "⠉"
BARLINE = "⠀"
DOUBLE_BARLINE = "⠣⠶"
MUSIC_PREFIX = "⠼"
MUSIC_COMMA = "⠄"


@dataclass
class BrailleNote:
    pitch_name: str
    octave: int
    duration: str
    accidental: Optional[str] = None
    dotted: bool = False
    tied: bool = False


@dataclass
class BrailleMeasure:
    notes: List[BrailleNote] = field(default_factory=list)
    in_accord: bool = False


def note_to_braille(note: BrailleNote, prev_octave: Optional[int] = None
                    ) -> str:
    result = ""

    if note.accidental and note.accidental in ACCIDENTAL_SIGNS:
        result += ACCIDENTAL_SIGNS[note.accidental]

    needs_octave = prev_octave is None or abs(note.octave - prev_octave) >= 1
    if needs_octave and note.octave in OCTAVE_MARKS:
        result += OCTAVE_MARKS[note.octave]

    pitch = note.pitch_name.upper()
    if note.duration == "whole":
        sign = NOTE_SIGNS_WHOLE.get(pitch, "⠽")
    elif note.duration == "half":
        sign = NOTE_SIGNS_HALF.get(pitch, "⠝")
    elif note.duration == "quarter":
        sign = NOTE_SIGNS_QUARTER.get(pitch, "⠹")
    else:
        sign = NOTE_SIGNS_EIGHTH.get(pitch, "⠙")
    result += sign

    if note.dotted:
        result += DOT_SIGN

    if note.tied:
        result += TIE_SIGN

    return result


def measure_to_braille(measure: BrailleMeasure) -> str:
    parts = []
    prev_octave = None
    for note in measure.notes:
        braille_str = note_to_braille(note, prev_octave)
        parts.append(braille_str)
        prev_octave = note.octave
    return "".join(parts)


def rest_to_braille(duration: str) -> str:
    return REST_SIGNS.get(duration, REST_SIGNS["quarter"])


def midi_to_braille_note(midi_num: int, duration: str = "quarter",
                         use_sharps: bool = True) -> BrailleNote:
    pitch_names_sharp = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
    pitch_names_flat = ["C", "D", "D", "E", "E", "F", "G", "G", "A", "A", "B", "B"]
    accidentals_sharp = [None, "sharp", None, "sharp", None, None, "sharp", None, "sharp", None, "sharp", None]
    accidentals_flat = [None, "flat", None, "flat", None, None, "flat", None, "flat", None, "flat", None]

    pc = midi_num % 12
    octave = (midi_num // 12) - 1

    if use_sharps:
        pitch = pitch_names_sharp[pc]
        acc = accidentals_sharp[pc]
    else:
        pitch = pitch_names_flat[pc]
        acc = accidentals_flat[pc]

    octave = max(1, min(7, octave))
    return BrailleNote(pitch_name=pitch, octave=octave,
                       duration=duration, accidental=acc)


def melody_to_braille(midi_notes: List[int], durations: List[str],
                      use_sharps: bool = True) -> str:
    parts = []
    prev_octave = None
    for midi_num, dur in zip(midi_notes, durations):
        bn = midi_to_braille_note(midi_num, dur, use_sharps)
        parts.append(note_to_braille(bn, prev_octave))
        prev_octave = bn.octave
    return "".join(parts)


def braille_to_text_description(braille_str: str) -> List[str]:
    reverse_octave = {v: k for k, v in OCTAVE_MARKS.items()}
    reverse_quarter = {v: k for k, v in NOTE_SIGNS_QUARTER.items()}
    reverse_half = {v: k for k, v in NOTE_SIGNS_HALF.items()}
    reverse_whole = {v: k for k, v in NOTE_SIGNS_WHOLE.items()}
    reverse_eighth = {v: k for k, v in NOTE_SIGNS_EIGHTH.items()}

    descriptions = []
    i = 0
    current_octave = 4
    chars = list(braille_str)

    while i < len(chars):
        ch = chars[i]

        if ch in reverse_octave:
            current_octave = reverse_octave[ch]
            i += 1
            continue

        for dur_name, lookup in [("quarter", reverse_quarter),
                                  ("half", reverse_half),
                                  ("whole", reverse_whole),
                                  ("eighth", reverse_eighth)]:
            if ch in lookup:
                note_name = lookup[ch]
                descriptions.append(f"{note_name}{current_octave} {dur_name}")
                break
        i += 1

    return descriptions


def interval_sign(semitones: int) -> str:
    interval_braille = {
        0: "⠌",
        1: "⠌",
        2: "⠬",
        3: "⠼",
        4: "⠔",
        5: "⠴",
        7: "⠤",
    }
    return interval_braille.get(abs(semitones) % 12, "⠤")


def score_to_braille(measures: List[BrailleMeasure]) -> str:
    parts = [MUSIC_PREFIX]
    for i, m in enumerate(measures):
        parts.append(measure_to_braille(m))
        if i < len(measures) - 1:
            parts.append(BARLINE)
    parts.append(DOUBLE_BARLINE)
    return "".join(parts)


def braille_line_length(braille_str: str, max_cells: int = 40) -> List[str]:
    lines = []
    current = ""
    for ch in braille_str:
        current += ch
        if len(current) >= max_cells:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines
