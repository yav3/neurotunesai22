"""Guitar/bass tablature: string tunings, fret mapping, fingering optimization,
tab rendering, and MIDI-to-tab conversion.  Pure numpy."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class InstrumentTuning(str, Enum):
    GUITAR_STANDARD = "guitar_standard"
    GUITAR_DROP_D = "guitar_drop_d"
    GUITAR_DADGAD = "guitar_dadgad"
    GUITAR_OPEN_G = "guitar_open_g"
    GUITAR_OPEN_D = "guitar_open_d"
    BASS_STANDARD = "bass_standard"
    BASS_5STRING = "bass_5string"
    UKULELE_STANDARD = "ukulele_standard"


TUNINGS: Dict[InstrumentTuning, List[int]] = {
    InstrumentTuning.GUITAR_STANDARD: [40, 45, 50, 55, 59, 64],
    InstrumentTuning.GUITAR_DROP_D: [38, 45, 50, 55, 59, 64],
    InstrumentTuning.GUITAR_DADGAD: [38, 45, 50, 55, 57, 62],
    InstrumentTuning.GUITAR_OPEN_G: [38, 43, 50, 55, 59, 62],
    InstrumentTuning.GUITAR_OPEN_D: [38, 45, 50, 54, 57, 62],
    InstrumentTuning.BASS_STANDARD: [28, 33, 38, 43],
    InstrumentTuning.BASS_5STRING: [23, 28, 33, 38, 43],
    InstrumentTuning.UKULELE_STANDARD: [67, 60, 64, 69],
}

MAX_FRET = 24


@dataclass
class TabNote:
    string: int
    fret: int
    duration_beats: float = 1.0
    technique: str = ""


@dataclass
class TabMeasure:
    notes: List[List[TabNote]] = field(default_factory=list)
    time_sig_num: int = 4
    time_sig_den: int = 4


def midi_to_fret_options(
    midi_note: int, tuning: List[int], max_fret: int = MAX_FRET,
) -> List[Tuple[int, int]]:
    options = []
    for s, open_pitch in enumerate(tuning):
        fret = midi_note - open_pitch
        if 0 <= fret <= max_fret:
            options.append((s, fret))
    return options


def fret_distance(a: TabNote, b: TabNote) -> float:
    string_cost = abs(a.string - b.string) * 1.5
    fret_cost = abs(a.fret - b.fret) * 1.0
    return string_cost + fret_cost


def hand_stretch(notes: List[TabNote]) -> int:
    frets = [n.fret for n in notes if n.fret > 0]
    if len(frets) < 2:
        return 0
    return max(frets) - min(frets)


def position_penalty(notes: List[TabNote], max_stretch: int = 4) -> float:
    stretch = hand_stretch(notes)
    if stretch <= max_stretch:
        return 0.0
    return (stretch - max_stretch) * 2.0


def optimize_fingering(
    midi_notes: List[int],
    tuning: InstrumentTuning = InstrumentTuning.GUITAR_STANDARD,
    max_stretch: int = 4,
) -> List[TabNote]:
    strings = TUNINGS[tuning]
    result: List[TabNote] = []
    prev: Optional[TabNote] = None
    for midi in midi_notes:
        options = midi_to_fret_options(midi, strings)
        if not options:
            result.append(TabNote(0, -1))
            continue
        best_cost = np.inf
        best_note = TabNote(options[0][0], options[0][1])
        for s, f in options:
            candidate = TabNote(s, f)
            cost = 0.0
            if prev is not None:
                cost += fret_distance(prev, candidate)
            if f > 12:
                cost += (f - 12) * 0.5
            if cost < best_cost:
                best_cost = cost
                best_note = candidate
        result.append(best_note)
        prev = best_note
    return result


def chord_fingering(
    midi_notes: List[int],
    tuning: InstrumentTuning = InstrumentTuning.GUITAR_STANDARD,
) -> List[TabNote]:
    strings = TUNINGS[tuning]
    n_strings = len(strings)
    assignments: List[Optional[TabNote]] = [None] * n_strings
    sorted_notes = sorted(midi_notes)
    for midi in sorted_notes:
        best_s, best_f = -1, -1
        best_cost = np.inf
        for s, open_pitch in enumerate(strings):
            if assignments[s] is not None:
                continue
            f = midi - open_pitch
            if 0 <= f <= MAX_FRET:
                cost = f * 1.0 + abs(s - len(strings) // 2) * 0.3
                if cost < best_cost:
                    best_cost = cost
                    best_s, best_f = s, f
        if best_s >= 0:
            assignments[best_s] = TabNote(best_s, best_f)
    return [a for a in assignments if a is not None]


def tab_to_ascii(
    measures: List[TabMeasure],
    tuning: InstrumentTuning = InstrumentTuning.GUITAR_STANDARD,
) -> str:
    strings = TUNINGS[tuning]
    n_strings = len(strings)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    lines = []
    for s in range(n_strings - 1, -1, -1):
        label = note_names[strings[s] % 12]
        parts = [f"{label:>2}|"]
        for m in measures:
            beat_slots = m.time_sig_num * 3
            slot = ["-"] * beat_slots
            for beat_group in m.notes:
                for n in beat_group:
                    if n.string == s:
                        pos = min(int(n.duration_beats * 3), beat_slots - 1)
                        slot[pos] = str(n.fret) if n.fret < 10 else f"{n.fret}"
            parts.append("".join(slot) + "|")
        lines.append("".join(parts))
    return "\n".join(lines)


def playability_score(notes: List[TabNote], max_stretch: int = 4) -> float:
    if not notes:
        return 1.0
    penalties = 0.0
    for i, n in enumerate(notes):
        if n.fret < 0:
            penalties += 5.0
        if n.fret > 12:
            penalties += (n.fret - 12) * 0.3
    for i in range(1, len(notes)):
        penalties += fret_distance(notes[i - 1], notes[i]) * 0.1
    for i in range(len(notes) - 1):
        group = notes[i:i + 2]
        penalties += position_penalty(group, max_stretch) * 0.2
    return float(max(0.0, 1.0 - penalties / max(len(notes), 1)))


def capo_transpose(
    notes: List[TabNote], capo_fret: int,
) -> List[TabNote]:
    return [TabNote(n.string, max(0, n.fret - capo_fret), n.duration_beats, n.technique)
            for n in notes]


def suggest_capo(
    midi_notes: List[int],
    tuning: InstrumentTuning = InstrumentTuning.GUITAR_STANDARD,
) -> Tuple[int, float]:
    best_capo = 0
    best_score = -1.0
    for capo in range(13):
        shifted = [m - capo for m in midi_notes]
        fingering = optimize_fingering(shifted, tuning)
        score = playability_score(fingering)
        if score > best_score:
            best_score = score
            best_capo = capo
    return best_capo, best_score


def string_usage_distribution(notes: List[TabNote], n_strings: int = 6) -> np.ndarray:
    counts = np.zeros(n_strings)
    for n in notes:
        if 0 <= n.string < n_strings:
            counts[n.string] += 1
    total = counts.sum()
    if total > 0:
        return counts / total
    return counts


def fret_histogram(notes: List[TabNote], n_frets: int = 25) -> np.ndarray:
    hist = np.zeros(n_frets)
    for n in notes:
        if 0 <= n.fret < n_frets:
            hist[n.fret] += 1
    return hist


def difficulty_estimate(notes: List[TabNote]) -> float:
    if not notes:
        return 0.0
    avg_fret = np.mean([n.fret for n in notes if n.fret >= 0]) if notes else 0
    stretches = []
    for i in range(1, len(notes)):
        stretches.append(fret_distance(notes[i - 1], notes[i]))
    avg_stretch = np.mean(stretches) if stretches else 0
    high_fret_ratio = sum(1 for n in notes if n.fret > 7) / max(len(notes), 1)
    return float(min(1.0, 0.3 * (avg_fret / 12) + 0.4 * (avg_stretch / 8) + 0.3 * high_fret_ratio))
