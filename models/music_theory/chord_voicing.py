"""Chord voicing generation and voice leading optimization."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product

import numpy as np


class VoiceLeadingRule(Enum):
    NEAREST = "nearest"
    SMOOTH = "smooth"
    CONTRARY = "contrary"
    OBLIQUE = "oblique"
    PARALLEL = "parallel"


@dataclass(frozen=True)
class Voicing:
    pitches: tuple[int, ...]
    root: int
    bass: int
    inversion: int

    @property
    def n_voices(self) -> int:
        return len(self.pitches)


def generate_voicings(
    chord_tones: list[int], n_voices: int, low: int = 36, high: int = 84,
) -> list[Voicing]:
    if not chord_tones or n_voices < 1:
        return []
    pcs = [t % 12 for t in chord_tones]
    root_pc = pcs[0]
    candidates: list[list[int]] = []
    for pc in pcs:
        octaves = []
        p = pc
        while p < low:
            p += 12
        while p <= high:
            octaves.append(p)
            p += 12
        candidates.append(octaves)

    if n_voices <= len(candidates):
        pools = candidates[:n_voices]
    else:
        pools = list(candidates)
        idx = 0
        while len(pools) < n_voices:
            pools.append(candidates[idx % len(candidates)])
            idx += 1

    voicings = []
    for combo in product(*pools):
        pitches = tuple(sorted(combo))
        if len(set(pitches)) < n_voices:
            continue
        bass = pitches[0]
        inv = pcs.index(bass % 12) if (bass % 12) in pcs else 0
        v = Voicing(pitches=pitches, root=root_pc, bass=bass, inversion=inv)
        voicings.append(v)
    seen: set[tuple[int, ...]] = set()
    unique = []
    for v in voicings:
        if v.pitches not in seen:
            seen.add(v.pitches)
            unique.append(v)
    return unique


def voice_leading_cost(v1: Voicing, v2: Voicing) -> float:
    n = min(v1.n_voices, v2.n_voices)
    return float(sum(abs(v1.pitches[i] - v2.pitches[i]) for i in range(n)))


def smooth_voice_leading(v1: Voicing, target_chord: list[int]) -> Voicing:
    target_pcs = [t % 12 for t in target_chord]
    root_pc = target_pcs[0]
    new_pitches: list[int] = []
    for old_p in v1.pitches:
        best = old_p
        best_dist = 999
        for pc in target_pcs:
            candidate = old_p + ((pc - old_p % 12) % 12)
            if candidate - old_p > 6:
                candidate -= 12
            dist = abs(candidate - old_p)
            if dist < best_dist:
                best_dist = dist
                best = candidate
        new_pitches.append(best)
    pitches = tuple(sorted(new_pitches))
    bass = pitches[0]
    inv = target_pcs.index(bass % 12) if (bass % 12) in target_pcs else 0
    return Voicing(pitches=pitches, root=root_pc, bass=bass, inversion=inv)


def check_parallel_fifths(v1: Voicing, v2: Voicing) -> list[tuple[int, int]]:
    n = min(v1.n_voices, v2.n_voices)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            int1 = abs(v1.pitches[i] - v1.pitches[j]) % 12
            int2 = abs(v2.pitches[i] - v2.pitches[j]) % 12
            if int1 == 7 and int2 == 7:
                if v1.pitches[i] != v2.pitches[i] or v1.pitches[j] != v2.pitches[j]:
                    pairs.append((i, j))
    return pairs


def check_parallel_octaves(v1: Voicing, v2: Voicing) -> list[tuple[int, int]]:
    n = min(v1.n_voices, v2.n_voices)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            int1 = abs(v1.pitches[i] - v1.pitches[j]) % 12
            int2 = abs(v2.pitches[i] - v2.pitches[j]) % 12
            if int1 == 0 and int2 == 0:
                if v1.pitches[i] != v2.pitches[i] or v1.pitches[j] != v2.pitches[j]:
                    pairs.append((i, j))
    return pairs


def voice_leading_quality(progression: list[Voicing]) -> float:
    if len(progression) < 2:
        return 1.0
    costs = []
    parallels = 0
    for i in range(len(progression) - 1):
        c = voice_leading_cost(progression[i], progression[i + 1])
        costs.append(c)
        parallels += len(check_parallel_fifths(progression[i], progression[i + 1]))
        parallels += len(check_parallel_octaves(progression[i], progression[i + 1]))
    avg_cost = np.mean(costs)
    smoothness = max(0.0, 1.0 - avg_cost / 24.0)
    penalty = parallels * 0.15
    return float(max(0.0, smoothness - penalty))


def drop2_voicing(pitches: list[int]) -> Voicing:
    if len(pitches) < 4:
        s = tuple(sorted(pitches))
        return Voicing(pitches=s, root=s[0] % 12, bass=s[0], inversion=0)
    s = sorted(pitches, reverse=True)
    second = s[1]
    s.remove(second)
    s.append(second - 12 if second - 12 >= 0 else second)
    final = tuple(sorted(s))
    return Voicing(pitches=final, root=final[0] % 12, bass=final[0], inversion=0)


def spread_voicing(pitches: list[int], octave_span: int = 3) -> Voicing:
    if not pitches:
        return Voicing(pitches=(), root=0, bass=0, inversion=0)
    pcs = [p % 12 for p in pitches]
    base = min(pitches)
    step = (octave_span * 12) / max(len(pcs) - 1, 1)
    spread = []
    for i, pc in enumerate(pcs):
        target = base + i * step
        p = pc
        while p < target - 6:
            p += 12
        spread.append(int(round(p)))
    final = tuple(sorted(spread))
    return Voicing(pitches=final, root=pcs[0], bass=final[0], inversion=0)
