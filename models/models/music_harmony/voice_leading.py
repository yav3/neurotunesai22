"""Voice leading: cost functions, parallel motion detection, optimal voice assignment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class VoiceLeadingRules:
    max_leap: int = 7
    parallel_fifth_penalty: float = 10.0
    parallel_octave_penalty: float = 10.0
    crossing_penalty: float = 5.0
    leap_penalty_per_semitone: float = 0.5
    common_tone_bonus: float = -1.0


def voice_leading_cost(
    source: Sequence[int], target: Sequence[int],
    rules: VoiceLeadingRules | None = None,
) -> float:
    if len(source) != len(target):
        return float("inf")
    if not source:
        return 0.0
    r = rules or VoiceLeadingRules()
    cost = 0.0
    for s, t in zip(source, target):
        motion = abs(t - s)
        if motion == 0:
            cost += r.common_tone_bonus
        elif motion > r.max_leap:
            cost += r.leap_penalty_per_semitone * motion * 2
        else:
            cost += r.leap_penalty_per_semitone * motion
    cost += _parallel_penalty(source, target, r)
    cost += _crossing_penalty(source, target, r)
    return max(0.0, cost)


def _parallel_penalty(
    source: Sequence[int], target: Sequence[int], r: VoiceLeadingRules,
) -> float:
    penalty = 0.0
    n = len(source)
    for i in range(n):
        for j in range(i + 1, n):
            interval_before = (source[j] - source[i]) % 12
            interval_after = (target[j] - target[i]) % 12
            same_direction = (target[i] - source[i]) * (target[j] - source[j]) > 0
            if same_direction:
                if interval_before == 7 and interval_after == 7:
                    penalty += r.parallel_fifth_penalty
                if interval_before == 0 and interval_after == 0 and source[i] != target[i]:
                    penalty += r.parallel_octave_penalty
    return penalty


def _crossing_penalty(
    source: Sequence[int], target: Sequence[int], r: VoiceLeadingRules,
) -> float:
    penalty = 0.0
    for i in range(len(target) - 1):
        if target[i] > target[i + 1] and source[i] <= source[i + 1]:
            penalty += r.crossing_penalty
        elif target[i] < target[i + 1] and source[i] >= source[i + 1]:
            penalty += r.crossing_penalty
    return penalty


def parallel_fifths_check(
    source: Sequence[int], target: Sequence[int],
) -> list[tuple[int, int]]:
    violations: list[tuple[int, int]] = []
    n = len(source)
    for i in range(n):
        for j in range(i + 1, n):
            if (source[j] - source[i]) % 12 == 7 and (target[j] - target[i]) % 12 == 7:
                if (target[i] - source[i]) * (target[j] - source[j]) > 0:
                    violations.append((i, j))
    return violations


def parallel_octaves_check(
    source: Sequence[int], target: Sequence[int],
) -> list[tuple[int, int]]:
    violations: list[tuple[int, int]] = []
    n = len(source)
    for i in range(n):
        for j in range(i + 1, n):
            if (source[j] - source[i]) % 12 == 0 and (target[j] - target[i]) % 12 == 0:
                if source[i] != target[i] and (target[i] - source[i]) * (target[j] - source[j]) > 0:
                    violations.append((i, j))
    return violations


def voice_lead(
    source: Sequence[int], target_pcs: frozenset[int],
    n_voices: int | None = None,
    rules: VoiceLeadingRules | None = None,
) -> list[int]:
    if not target_pcs:
        return list(source)
    n = n_voices or len(source)
    target_options = sorted(target_pcs)
    candidates: list[int] = []
    for pc in target_options:
        for octave in range(-1, 9):
            candidates.append(pc + octave * 12)
    candidates = [c for c in candidates if 21 <= c <= 108]
    if not candidates:
        return list(source[:n])
    result: list[int] = []
    used: set[int] = set()
    for s in source[:n]:
        best = min(candidates, key=lambda c: abs(c - s) + (5 if c in used else 0))
        result.append(best)
        used.add(best)
    return result
