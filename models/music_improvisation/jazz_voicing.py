"""Jazz voicing: drop-2/3, shell, rootless, quartal voicings and comping patterns.

Generates and optimises jazz chord voicings: drop-2, drop-3, shell, rootless
(type A/B), quartal stacks.  Voice-leading cost minimisation, comping rhythm
patterns (Freddie Green, bossa nova), and chord-density curves.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class VoicingType(str, Enum):
    DROP2 = "drop2"
    DROP3 = "drop3"
    SHELL = "shell"
    ROOTLESS_A = "rootless_a"
    ROOTLESS_B = "rootless_b"
    QUARTAL = "quartal"
    CLUSTER = "cluster"
    SPREAD = "spread"


CHORD_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "maj7":    (0, 4, 7, 11),
    "min7":    (0, 3, 7, 10),
    "dom7":    (0, 4, 7, 10),
    "min7b5":  (0, 3, 6, 10),
    "dim7":    (0, 3, 6, 9),
    "aug7":    (0, 4, 8, 10),
    "maj6":    (0, 4, 7, 9),
    "min6":    (0, 3, 7, 9),
    "dom9":    (0, 4, 7, 10, 14),
    "min9":    (0, 3, 7, 10, 14),
    "maj9":    (0, 4, 7, 11, 14),
}


@dataclass
class ChordVoicing:
    root: int
    quality: str
    voicing_type: VoicingType
    midi_notes: Tuple[int, ...]

    @property
    def n_notes(self) -> int:
        return len(self.midi_notes)

    @property
    def span(self) -> int:
        return max(self.midi_notes) - min(self.midi_notes) if self.midi_notes else 0


def _close_position(root: int, quality: str) -> Tuple[int, ...]:
    intervals = CHORD_INTERVALS.get(quality, (0, 4, 7, 10))
    return tuple(root + i for i in intervals)


def _invert(notes: Tuple[int, ...], inversion: int) -> Tuple[int, ...]:
    lst = list(notes)
    for _ in range(inversion % len(lst)):
        lst.append(lst.pop(0) + 12)
    return tuple(lst)


def drop2_voicing(root: int, quality: str = "dom7",
                  inversion: int = 0) -> ChordVoicing:
    close = _invert(_close_position(root, quality), inversion)
    lst = list(close)
    if len(lst) >= 4:
        second = lst.pop(-2)
        lst.insert(0, second - 12)
    return ChordVoicing(root, quality, VoicingType.DROP2, tuple(lst))


def drop3_voicing(root: int, quality: str = "dom7",
                  inversion: int = 0) -> ChordVoicing:
    close = _invert(_close_position(root, quality), inversion)
    lst = list(close)
    if len(lst) >= 4:
        third = lst.pop(-3)
        lst.insert(0, third - 12)
    return ChordVoicing(root, quality, VoicingType.DROP3, tuple(lst))


def shell_voicing(root: int, quality: str = "dom7") -> ChordVoicing:
    intervals = CHORD_INTERVALS.get(quality, (0, 4, 7, 10))
    third = intervals[1] if len(intervals) > 1 else 4
    seventh = intervals[3] if len(intervals) > 3 else intervals[-1]
    notes = (root, root + third, root + seventh)
    return ChordVoicing(root, quality, VoicingType.SHELL, notes)


def rootless_voicing(root: int, quality: str = "dom7",
                     variant: str = "A") -> ChordVoicing:
    intervals = CHORD_INTERVALS.get(quality, (0, 4, 7, 10))
    if len(intervals) < 4:
        intervals = intervals + (14,)
    third, fifth, seventh = intervals[1], intervals[2], intervals[3]
    ninth = 14
    if variant.upper() == "A":
        notes = (root + third, root + fifth, root + seventh, root + ninth)
    else:
        notes = (root + seventh, root + ninth, root + third + 12, root + fifth + 12)
    return ChordVoicing(root, quality,
                        VoicingType.ROOTLESS_A if variant.upper() == "A"
                        else VoicingType.ROOTLESS_B,
                        notes)


def quartal_voicing(root: int, n_notes: int = 4,
                    interval: int = 5) -> ChordVoicing:
    notes = tuple(root + i * interval for i in range(n_notes))
    return ChordVoicing(root, "quartal", VoicingType.QUARTAL, notes)


def voice_leading_cost(v1: ChordVoicing, v2: ChordVoicing) -> int:
    n1 = sorted(v1.midi_notes)
    n2 = sorted(v2.midi_notes)
    min_len = min(len(n1), len(n2))
    return sum(abs(n1[i] - n2[i]) for i in range(min_len))


def optimal_voice_leading(voicings: List[ChordVoicing],
                          target: ChordVoicing) -> ChordVoicing:
    if not voicings:
        return target
    costs = [(voice_leading_cost(v, target), i) for i, v in enumerate(voicings)]
    costs.sort()
    return voicings[costs[0][1]]


def comping_rhythm_pattern(style: str = "swing",
                           beats_per_bar: int = 4) -> np.ndarray:
    if style == "swing":
        pattern = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    elif style == "bossa":
        pattern = np.array([1, 0, 0, 1, 0, 0, 1, 0])
    elif style == "latin":
        pattern = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    elif style == "ballad":
        pattern = np.array([1, 0, 0, 0, 0, 0, 1, 0])
    else:
        pattern = np.ones(beats_per_bar * 2, dtype=float)
    return pattern.astype(float)


def freddie_green_pattern(bpm: float, bars: int = 4) -> np.ndarray:
    beat_ms = 60_000.0 / bpm
    total_beats = bars * 4
    return np.arange(total_beats) * beat_ms


def bossa_comping(bpm: float, bars: int = 4) -> np.ndarray:
    beat_ms = 60_000.0 / bpm
    eighth_ms = beat_ms / 2.0
    onsets = []
    for bar in range(bars):
        base = bar * 4 * beat_ms
        onsets.extend([base, base + 1.5 * beat_ms,
                       base + 2 * beat_ms + eighth_ms])
    return np.array(onsets)


def chord_density_curve(n_chords: int, style: str = "ballad") -> np.ndarray:
    x = np.linspace(0, 1, n_chords)
    if style == "ballad":
        return 3 + 2 * np.sin(np.pi * x)
    elif style == "bebop":
        return 4 + np.random.default_rng(42).uniform(-0.5, 0.5, n_chords)
    elif style == "sparse":
        return 3 * np.ones(n_chords)
    return 4 * np.ones(n_chords)


def generate_voicing_sequence(roots: List[int],
                              qualities: List[str],
                              voicing_type: VoicingType = VoicingType.DROP2,
                              ) -> List[ChordVoicing]:
    builders = {
        VoicingType.DROP2: lambda r, q: drop2_voicing(r, q),
        VoicingType.DROP3: lambda r, q: drop3_voicing(r, q),
        VoicingType.SHELL: lambda r, q: shell_voicing(r, q),
        VoicingType.ROOTLESS_A: lambda r, q: rootless_voicing(r, q, "A"),
        VoicingType.ROOTLESS_B: lambda r, q: rootless_voicing(r, q, "B"),
        VoicingType.QUARTAL: lambda r, q: quartal_voicing(r),
    }
    builder = builders.get(voicing_type, builders[VoicingType.DROP2])
    result: List[ChordVoicing] = []
    for r, q in zip(roots, qualities):
        candidate = builder(r, q)
        if result:
            inversions = [builder(r, q)]
            for inv in range(1, 4):
                if voicing_type in (VoicingType.DROP2, VoicingType.DROP3):
                    fn = drop2_voicing if voicing_type == VoicingType.DROP2 else drop3_voicing
                    inversions.append(fn(r, q, inv))
            candidate = optimal_voice_leading(inversions, result[-1])
        result.append(candidate)
    return result
