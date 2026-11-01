"""Motivic improvisation: develop motifs in real-time improvisatory contexts.

Algorithms for motivic manipulation during improvisation: call-and-response
generation, motivic fragmentation and recombination, bebop enclosure patterns,
approach-note strategies, digital patterns, and motivic density control.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ApproachType(str, Enum):
    CHROMATIC_ABOVE = "chromatic_above"
    CHROMATIC_BELOW = "chromatic_below"
    DIATONIC_ABOVE = "diatonic_above"
    DIATONIC_BELOW = "diatonic_below"
    ENCLOSURE = "enclosure"
    DOUBLE_CHROMATIC = "double_chromatic"


class DigitalPattern(str, Enum):
    P1235 = "1235"
    P1325 = "1325"
    P5321 = "5321"
    P1357 = "1357"
    P3517 = "3517"
    P7531 = "7531"


@dataclass
class MotivicCell:
    intervals: Tuple[int, ...]
    rhythm: Tuple[float, ...]
    contour: Tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.intervals) + 1


@dataclass
class ImprovPhrase:
    pitches: np.ndarray
    onsets_ms: np.ndarray
    durations_ms: np.ndarray
    cell_boundaries: List[int]


def extract_cells(pitches: np.ndarray, min_len: int = 3,
                  max_len: int = 7) -> List[MotivicCell]:
    cells: List[MotivicCell] = []
    intervals = np.diff(pitches.astype(int))
    i = 0
    while i + min_len <= len(intervals):
        length = min(max_len, len(intervals) - i)
        seg = tuple(int(x) for x in intervals[i:i + length])
        contour = tuple(1 if d > 0 else (-1 if d < 0 else 0) for d in seg)
        rhythm = tuple(1.0 for _ in seg)
        cells.append(MotivicCell(intervals=seg, rhythm=rhythm, contour=contour))
        i += length
    return cells


def transpose_cell(cell: MotivicCell, semitones: int) -> MotivicCell:
    return MotivicCell(
        intervals=cell.intervals,
        rhythm=cell.rhythm,
        contour=cell.contour,
    )


def invert_cell(cell: MotivicCell) -> MotivicCell:
    return MotivicCell(
        intervals=tuple(-i for i in cell.intervals),
        rhythm=cell.rhythm,
        contour=tuple(-c for c in cell.contour),
    )


def retrograde_cell(cell: MotivicCell) -> MotivicCell:
    return MotivicCell(
        intervals=tuple(-i for i in reversed(cell.intervals)),
        rhythm=cell.rhythm[::-1],
        contour=tuple(-c for c in reversed(cell.contour)),
    )


def augment_cell(cell: MotivicCell, factor: float = 2.0) -> MotivicCell:
    return MotivicCell(
        intervals=cell.intervals,
        rhythm=tuple(r * factor for r in cell.rhythm),
        contour=cell.contour,
    )


def fragment_cell(cell: MotivicCell, start: int = 0,
                  length: int = 3) -> MotivicCell:
    end = min(start + length, len(cell.intervals))
    return MotivicCell(
        intervals=cell.intervals[start:end],
        rhythm=cell.rhythm[start:end],
        contour=cell.contour[start:end],
    )


def enclosure_pattern(target_pitch: int,
                      approach: ApproachType = ApproachType.ENCLOSURE) -> np.ndarray:
    if approach == ApproachType.CHROMATIC_ABOVE:
        return np.array([target_pitch + 1, target_pitch])
    elif approach == ApproachType.CHROMATIC_BELOW:
        return np.array([target_pitch - 1, target_pitch])
    elif approach == ApproachType.DIATONIC_ABOVE:
        return np.array([target_pitch + 2, target_pitch])
    elif approach == ApproachType.DIATONIC_BELOW:
        return np.array([target_pitch - 2, target_pitch])
    elif approach == ApproachType.ENCLOSURE:
        return np.array([target_pitch + 1, target_pitch - 1, target_pitch])
    elif approach == ApproachType.DOUBLE_CHROMATIC:
        return np.array([target_pitch + 2, target_pitch + 1, target_pitch])
    return np.array([target_pitch])


def digital_pattern_pitches(root: int, quality: str = "dom7",
                            pattern: DigitalPattern = DigitalPattern.P1235) -> np.ndarray:
    chord_map = {
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "dom7": [0, 4, 7, 10],
        "dim7": [0, 3, 6, 9],
    }
    tones = chord_map.get(quality, [0, 4, 7, 10])
    degrees = [int(c) - 1 for c in pattern.value]
    result = []
    for d in degrees:
        octave_shift = d // len(tones) * 12
        result.append(root + tones[d % len(tones)] + octave_shift)
    return np.array(result)


def call_and_response(call_pitches: np.ndarray,
                      transformation: str = "invert") -> np.ndarray:
    intervals = np.diff(call_pitches.astype(int))
    if transformation == "invert":
        resp_intervals = -intervals
    elif transformation == "retrograde":
        resp_intervals = -intervals[::-1]
    elif transformation == "transpose":
        resp_intervals = intervals
    elif transformation == "augment":
        resp_intervals = intervals * 2
    else:
        resp_intervals = intervals
    start = call_pitches[-1] + 2
    response = np.zeros(len(resp_intervals) + 1, dtype=int)
    response[0] = start
    for i, iv in enumerate(resp_intervals):
        response[i + 1] = response[i] + iv
    return response


def motivic_density(pitches: np.ndarray, onsets_ms: np.ndarray,
                    window_ms: float = 2000.0) -> np.ndarray:
    if len(onsets_ms) < 2:
        return np.array([0.0])
    total_dur = onsets_ms[-1] - onsets_ms[0]
    if total_dur <= 0:
        return np.array([float(len(pitches))])
    n_windows = max(1, int(total_dur / window_ms))
    density = np.zeros(n_windows)
    for i in range(n_windows):
        start = onsets_ms[0] + i * window_ms
        end = start + window_ms
        count = np.sum((onsets_ms >= start) & (onsets_ms < end))
        density[i] = count / (window_ms / 1000.0)
    return density


def tension_release_curve(pitches: np.ndarray,
                          scale: Tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)) -> np.ndarray:
    scale_set = set(s % 12 for s in scale)
    tension = np.zeros(len(pitches))
    for i, p in enumerate(pitches):
        pc = int(p) % 12
        if pc not in scale_set:
            tension[i] = 1.0
        else:
            tension[i] = 0.0
        if i > 0:
            leap = abs(int(pitches[i]) - int(pitches[i - 1]))
            if leap > 7:
                tension[i] += 0.5
    return np.clip(tension, 0.0, 1.5)


def generate_bebop_line(root: int, quality: str, n_beats: int = 4,
                        bpm: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    chord_map = {"maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10],
                 "dom7": [0, 4, 7, 10], "dim7": [0, 3, 6, 9]}
    tones = chord_map.get(quality, [0, 4, 7, 10])
    eighth_ms = 60_000.0 / bpm / 2.0
    n_notes = n_beats * 2
    pitches = np.zeros(n_notes, dtype=int)
    onsets = np.arange(n_notes) * eighth_ms
    pitches[0] = root + tones[rng.integers(len(tones))]
    for i in range(1, n_notes):
        if rng.random() < 0.7:
            pitches[i] = pitches[i - 1] + rng.choice([-2, -1, 1, 2])
        else:
            pitches[i] = root + tones[rng.integers(len(tones))]
    return pitches, onsets
