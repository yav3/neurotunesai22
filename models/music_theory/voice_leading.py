"""Voice leading analysis after Tymoczko (2011).

Minimal voice leading distance, motion classification (parallel, similar,
contrary, oblique, static), parallel fifths/octaves detection, optimal
voicing search, and smoothness scoring for chord progressions.
"""
from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


class VoiceLeadingType(enum.Enum):
    PARALLEL = "parallel"
    SIMILAR = "similar"
    CONTRARY = "contrary"
    OBLIQUE = "oblique"
    STATIC = "static"


@dataclass(frozen=True)
class VoiceLeading:
    from_pitches: Tuple[int, ...]
    to_pitches: Tuple[int, ...]
    voice_motions: List[VoiceLeadingType]

    @property
    def distance(self) -> float:
        return sum(abs(a - b) for a, b in zip(self.from_pitches, self.to_pitches))


@dataclass(frozen=True)
class ChordTransition:
    from_chord: Tuple[int, ...]
    to_chord: Tuple[int, ...]
    voice_leading: VoiceLeading
    distance: float
    is_smooth: bool


def classify_motion(a1: int, a2: int, b1: int, b2: int) -> VoiceLeadingType:
    """Classify the motion between two voices moving from (a1,b1) to (a2,b2)."""
    da = a2 - a1
    db = b2 - b1
    if da == 0 and db == 0:
        return VoiceLeadingType.STATIC
    if da == 0 or db == 0:
        return VoiceLeadingType.OBLIQUE
    if da == db:
        return VoiceLeadingType.PARALLEL
    if (da > 0 and db < 0) or (da < 0 and db > 0):
        return VoiceLeadingType.CONTRARY
    return VoiceLeadingType.SIMILAR


def voice_leading_distance(chord_a: Tuple[int, ...], chord_b: Tuple[int, ...]) -> float:
    """Minimal voice leading distance via optimal assignment (brute-force for n<=8)."""
    n = len(chord_a)
    m = len(chord_b)
    if n == 0 or m == 0:
        return 0.0
    if n != m:
        if n < m:
            chord_a = chord_a + (chord_a[-1],) * (m - n)
            n = m
        else:
            chord_b = chord_b + (chord_b[-1],) * (n - m)
            m = n
    if n <= 8:
        best = float("inf")
        for perm in itertools.permutations(range(n)):
            d = sum(abs(chord_a[i] - chord_b[perm[i]]) for i in range(n))
            if d < best:
                best = d
        return float(best)
    cost = np.array([[abs(chord_a[i] - chord_b[j]) for j in range(n)] for i in range(n)])
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].sum())


def analyze_voice_leading(
    from_pitches: Tuple[int, ...], to_pitches: Tuple[int, ...]
) -> VoiceLeading:
    """Full voice leading analysis with per-voice motion classification."""
    n = min(len(from_pitches), len(to_pitches))
    motions = []
    for i in range(n):
        for j in range(i + 1, n):
            motions.append(
                classify_motion(from_pitches[i], to_pitches[i], from_pitches[j], to_pitches[j])
            )
    if not motions:
        per_voice = []
        for a, b in zip(from_pitches, to_pitches):
            if a == b:
                per_voice.append(VoiceLeadingType.STATIC)
            elif b > a:
                per_voice.append(VoiceLeadingType.SIMILAR)
            else:
                per_voice.append(VoiceLeadingType.SIMILAR)
        motions = per_voice
    return VoiceLeading(from_pitches, to_pitches, motions)


def smooth_voice_leading(
    chord_a: Tuple[int, ...], chord_b: Tuple[int, ...], max_distance: int = 2
) -> bool:
    """True if every voice moves ≤ max_distance semitones."""
    n = min(len(chord_a), len(chord_b))
    return all(abs(chord_a[i] - chord_b[i]) <= max_distance for i in range(n))


def optimal_voice_leading(
    chord_a: Tuple[int, ...], chord_b: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Find the voicing of chord_b minimising distance from chord_a.

    Tries all octave transpositions of each note in chord_b within ±12 of the
    corresponding chord_a pitch (or the chord_a centroid for size mismatches).
    """
    if not chord_a or not chord_b:
        return chord_b
    centroid = sum(chord_a) / len(chord_a)
    pc_b = [p % 12 for p in chord_b]
    best_voicing = chord_b
    best_dist = voice_leading_distance(chord_a, chord_b)
    candidates_per_voice: List[List[int]] = []
    for pc in pc_b:
        options = []
        for octave in range(-1, 10):
            pitch = pc + 12 * octave
            if abs(pitch - centroid) <= 18:
                options.append(pitch)
        if not options:
            options = [pc + 12 * (int(centroid) // 12)]
        candidates_per_voice.append(options)
    if len(candidates_per_voice) <= 4:
        for combo in itertools.product(*candidates_per_voice):
            d = voice_leading_distance(chord_a, combo)
            if d < best_dist:
                best_dist = d
                best_voicing = combo
    else:
        voicing = []
        targets = list(chord_a) if len(chord_a) == len(chord_b) else [centroid] * len(chord_b)
        for i, opts in enumerate(candidates_per_voice):
            t = targets[i] if i < len(targets) else centroid
            voicing.append(min(opts, key=lambda p: abs(p - t)))
        d = voice_leading_distance(chord_a, tuple(voicing))
        if d < best_dist:
            best_voicing = tuple(voicing)
    return best_voicing


def _interval_class(a: int, b: int) -> int:
    return abs(a - b) % 12


class VoiceLeadingAnalyzer:
    """Analyze voice leading in chord progressions."""

    def analyze_progression(
        self, chords: List[Tuple[int, ...]]
    ) -> List[VoiceLeading]:
        result = []
        for i in range(len(chords) - 1):
            result.append(analyze_voice_leading(chords[i], chords[i + 1]))
        return result

    def total_distance(self, chords: List[Tuple[int, ...]]) -> float:
        return sum(
            voice_leading_distance(chords[i], chords[i + 1])
            for i in range(len(chords) - 1)
        )

    def average_distance(self, chords: List[Tuple[int, ...]]) -> float:
        n = len(chords) - 1
        if n <= 0:
            return 0.0
        return self.total_distance(chords) / n

    def smoothness_score(self, chords: List[Tuple[int, ...]]) -> float:
        """Fraction of voice movements ≤ 2 semitones."""
        total = 0
        smooth = 0
        for i in range(len(chords) - 1):
            n = min(len(chords[i]), len(chords[i + 1]))
            for v in range(n):
                total += 1
                if abs(chords[i][v] - chords[i + 1][v]) <= 2:
                    smooth += 1
        if total == 0:
            return 1.0
        return smooth / total

    def contrary_motion_ratio(self, chords: List[Tuple[int, ...]]) -> float:
        """Fraction of voice pairs moving in contrary motion."""
        total = 0
        contrary = 0
        for i in range(len(chords) - 1):
            n = min(len(chords[i]), len(chords[i + 1]))
            for v1 in range(n):
                for v2 in range(v1 + 1, n):
                    total += 1
                    m = classify_motion(
                        chords[i][v1], chords[i + 1][v1],
                        chords[i][v2], chords[i + 1][v2],
                    )
                    if m == VoiceLeadingType.CONTRARY:
                        contrary += 1
        if total == 0:
            return 0.0
        return contrary / total

    def parallel_fifths_check(
        self, chords: List[Tuple[int, ...]]
    ) -> List[Tuple[int, int, int]]:
        """Detect parallel fifths: (chord_index, voice1, voice2)."""
        violations = []
        for i in range(len(chords) - 1):
            n = min(len(chords[i]), len(chords[i + 1]))
            for v1 in range(n):
                for v2 in range(v1 + 1, n):
                    ic_before = _interval_class(chords[i][v1], chords[i][v2])
                    ic_after = _interval_class(chords[i + 1][v1], chords[i + 1][v2])
                    if ic_before == 7 and ic_after == 7:
                        da = chords[i + 1][v1] - chords[i][v1]
                        db = chords[i + 1][v2] - chords[i][v2]
                        if da != 0 and da == db:
                            violations.append((i, v1, v2))
        return violations

    def parallel_octaves_check(
        self, chords: List[Tuple[int, ...]]
    ) -> List[Tuple[int, int, int]]:
        """Detect parallel octaves: (chord_index, voice1, voice2)."""
        violations = []
        for i in range(len(chords) - 1):
            n = min(len(chords[i]), len(chords[i + 1]))
            for v1 in range(n):
                for v2 in range(v1 + 1, n):
                    ic_before = _interval_class(chords[i][v1], chords[i][v2])
                    ic_after = _interval_class(chords[i + 1][v1], chords[i + 1][v2])
                    if ic_before == 0 and ic_after == 0:
                        da = chords[i + 1][v1] - chords[i][v1]
                        db = chords[i + 1][v2] - chords[i][v2]
                        if da != 0 and da == db:
                            violations.append((i, v1, v2))
        return violations
