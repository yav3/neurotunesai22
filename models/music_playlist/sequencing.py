"""Playlist sequencing: track ordering, flow optimization, and DJ-style transitions.

Implements intelligent track sequencing using key compatibility (Camelot wheel),
tempo ramping, harmonic mixing, and energy-arc shaping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class MixStrategy(Enum):
    HARMONIC = "harmonic"
    ENERGY_RAMP = "energy_ramp"
    TEMPO_SORT = "tempo_sort"
    MOOD_ARC = "mood_arc"
    RANDOM_WALK = "random_walk"


CAMELOT_WHEEL: dict[str, list[str]] = {
    "1A": ["1A", "1B", "12A", "2A"],
    "1B": ["1B", "1A", "12B", "2B"],
    "2A": ["2A", "2B", "1A", "3A"],
    "2B": ["2B", "2A", "1B", "3B"],
    "3A": ["3A", "3B", "2A", "4A"],
    "3B": ["3B", "3A", "2B", "4B"],
    "4A": ["4A", "4B", "3A", "5A"],
    "4B": ["4B", "4A", "3B", "5B"],
    "5A": ["5A", "5B", "4A", "6A"],
    "5B": ["5B", "5A", "4B", "6B"],
    "6A": ["6A", "6B", "5A", "7A"],
    "6B": ["6B", "6A", "5B", "7B"],
    "7A": ["7A", "7B", "6A", "8A"],
    "7B": ["7B", "7A", "6B", "8B"],
    "8A": ["8A", "8B", "7A", "9A"],
    "8B": ["8B", "8A", "7B", "9B"],
    "9A": ["9A", "9B", "8A", "10A"],
    "9B": ["9B", "9A", "8B", "10B"],
    "10A": ["10A", "10B", "9A", "11A"],
    "10B": ["10B", "10A", "9B", "11B"],
    "11A": ["11A", "11B", "10A", "12A"],
    "11B": ["11B", "11A", "10B", "12B"],
    "12A": ["12A", "12B", "11A", "1A"],
    "12B": ["12B", "12A", "11B", "1B"],
}


def camelot_compatible(key_a: str, key_b: str) -> bool:
    if key_a not in CAMELOT_WHEEL:
        return False
    return key_b in CAMELOT_WHEEL[key_a]


def camelot_distance(key_a: str, key_b: str) -> int:
    if key_a == key_b:
        return 0
    if key_a not in CAMELOT_WHEEL or key_b not in CAMELOT_WHEEL:
        return 12
    num_a = int(key_a[:-1])
    num_b = int(key_b[:-1])
    mode_a = key_a[-1]
    mode_b = key_b[-1]
    circle_dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    mode_dist = 0 if mode_a == mode_b else 1
    return circle_dist + mode_dist


@dataclass(frozen=True)
class SequenceTrack:
    track_id: str
    tempo_bpm: float
    camelot_key: str = ""
    energy: float = 0.5
    valence: float = 0.5
    duration_s: float = 240.0


def tempo_compatibility(bpm_a: float, bpm_b: float, tolerance: float = 0.06) -> float:
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.0
    ratio = bpm_a / bpm_b
    for target in [1.0, 0.5, 2.0, 0.75, 1.5]:
        if abs(ratio - target) / target < tolerance:
            return max(0.0, 1.0 - abs(ratio - target) / target / tolerance)
    return 0.0


def transition_score(a: SequenceTrack, b: SequenceTrack) -> float:
    key_score = 1.0 if camelot_compatible(a.camelot_key, b.camelot_key) else max(0.0, 1.0 - camelot_distance(a.camelot_key, b.camelot_key) / 6.0)
    tempo_score = tempo_compatibility(a.tempo_bpm, b.tempo_bpm)
    energy_diff = abs(a.energy - b.energy)
    energy_score = max(0.0, 1.0 - energy_diff / 0.4)
    return 0.4 * key_score + 0.35 * tempo_score + 0.25 * energy_score


def greedy_sequence(tracks: Sequence[SequenceTrack], start_idx: int = 0) -> list[int]:
    if len(tracks) == 0:
        return []
    n = len(tracks)
    used = [False] * n
    order = [start_idx]
    used[start_idx] = True
    for _ in range(n - 1):
        current = tracks[order[-1]]
        best_idx = -1
        best_score = -1.0
        for j in range(n):
            if used[j]:
                continue
            s = transition_score(current, tracks[j])
            if s > best_score:
                best_score = s
                best_idx = j
        if best_idx < 0:
            break
        order.append(best_idx)
        used[best_idx] = True
    return order


def tempo_ramp_sequence(tracks: Sequence[SequenceTrack], ascending: bool = True) -> list[int]:
    indexed = list(enumerate(tracks))
    indexed.sort(key=lambda x: x[1].tempo_bpm, reverse=not ascending)
    return [i for i, _ in indexed]


def energy_arc_sequence(tracks: Sequence[SequenceTrack], peak_position: float = 0.6) -> list[int]:
    n = len(tracks)
    if n == 0:
        return []
    indexed = sorted(enumerate(tracks), key=lambda x: x[1].energy)
    peak_idx = int(peak_position * (n - 1))
    result = [0] * n
    lo, hi = 0, n - 1
    for pos in range(n):
        if pos <= peak_idx:
            t = pos / max(peak_idx, 1)
            target_rank = int(t * (n - 1))
        else:
            t = (pos - peak_idx) / max(n - 1 - peak_idx, 1)
            target_rank = int((1.0 - t) * (n - 1))
        target_rank = max(0, min(n - 1, target_rank))
        result[pos] = indexed[target_rank][0]
    seen: set[int] = set()
    deduped: list[int] = []
    remaining = [idx for idx, _ in indexed]
    rem_ptr = 0
    for pos in range(n):
        if result[pos] not in seen:
            deduped.append(result[pos])
            seen.add(result[pos])
        else:
            while rem_ptr < n and remaining[rem_ptr] in seen:
                rem_ptr += 1
            if rem_ptr < n:
                deduped.append(remaining[rem_ptr])
                seen.add(remaining[rem_ptr])
                rem_ptr += 1
    return deduped


@dataclass
class PlaylistSequencer:
    strategy: MixStrategy = MixStrategy.HARMONIC

    def sequence(self, tracks: Sequence[SequenceTrack]) -> list[int]:
        if len(tracks) == 0:
            return []
        if self.strategy == MixStrategy.TEMPO_SORT:
            return tempo_ramp_sequence(tracks)
        if self.strategy == MixStrategy.ENERGY_RAMP:
            return energy_arc_sequence(tracks)
        if self.strategy == MixStrategy.RANDOM_WALK:
            rng = np.random.default_rng(42)
            order = list(range(len(tracks)))
            rng.shuffle(order)
            return list(order)
        return greedy_sequence(tracks)

    def total_flow_score(self, tracks: Sequence[SequenceTrack], order: list[int]) -> float:
        if len(order) < 2:
            return 1.0
        total = 0.0
        for i in range(len(order) - 1):
            total += transition_score(tracks[order[i]], tracks[order[i + 1]])
        return total / (len(order) - 1)
