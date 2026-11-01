"""Voice arrangement: voicing spread, doubling, register classification, spacing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VoiceArrangement:
    voices: list[list[int]]
    n_voices: int

    @property
    def all_pitches(self) -> list[int]:
        return [p for v in self.voices for p in v]


def voicing_spread(pitches: Sequence[int]) -> int:
    if len(pitches) < 2:
        return 0
    return max(pitches) - min(pitches)


def voicing_density(pitches: Sequence[int]) -> float:
    spread = voicing_spread(pitches)
    if spread == 0:
        return float(len(pitches))
    return len(pitches) / spread


def detect_doubling(
    voices: Sequence[Sequence[int]],
) -> list[tuple[int, int, int]]:
    doublings: list[tuple[int, int, int]] = []
    n = len(voices)
    for i in range(n):
        for j in range(i + 1, n):
            common = set(voices[i]) & set(voices[j])
            for p in sorted(common):
                doublings.append((i, j, p))
    return doublings


def arrangement_register(pitches: Sequence[int]) -> str:
    if not pitches:
        return "unknown"
    mean_p = float(np.mean(pitches))
    if mean_p < 48:
        return "bass"
    if mean_p < 60:
        return "tenor"
    if mean_p < 72:
        return "alto"
    if mean_p < 84:
        return "soprano"
    return "high"


def voice_crossing_count(voices: Sequence[Sequence[int]]) -> int:
    n_voices = len(voices)
    if n_voices < 2:
        return 0
    min_len = min(len(v) for v in voices)
    crossings = 0
    for t in range(min_len):
        for i in range(1, n_voices):
            if voices[i][t] > voices[i - 1][t]:
                crossings += 1
    return crossings


def spacing_quality(pitches: Sequence[int]) -> float:
    if len(pitches) < 2:
        return 1.0
    sorted_p = sorted(pitches)
    intervals = [sorted_p[i + 1] - sorted_p[i] for i in range(len(sorted_p) - 1)]
    if not intervals:
        return 1.0
    mean_iv = float(np.mean(intervals))
    if mean_iv < 1e-9:
        return 1.0
    std_iv = float(np.std(intervals))
    return max(0.0, min(1.0, 1.0 - std_iv / mean_iv))


def optimal_doubling(chord: Sequence[int], n_voices: int) -> list[int]:
    if not chord:
        return []
    sorted_chord = sorted(chord)
    result = list(sorted_chord)
    while len(result) < n_voices:
        result.append(sorted_chord[0])
        if len(result) < n_voices and len(sorted_chord) >= 3:
            result.append(sorted_chord[-2])
        elif len(result) < n_voices and len(sorted_chord) >= 2:
            result.append(sorted_chord[1])
    return sorted(result[:n_voices])


def register_balance(voices: Sequence[Sequence[int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in voices:
        reg = arrangement_register(v)
        counts[reg] = counts.get(reg, 0) + 1
    return counts


def open_close_voicing(pitches: Sequence[int]) -> str:
    if len(pitches) < 3:
        return "open"
    sorted_p = sorted(pitches)
    span = sorted_p[-1] - sorted_p[0]
    if span <= 12:
        return "close"
    max_gap = max(sorted_p[i + 1] - sorted_p[i] for i in range(len(sorted_p) - 1))
    if max_gap > 12:
        return "open"
    return "close"
