"""Voice leading analysis and optimization for arrangement.

Implements part-writing rule detection (parallel 5ths/8ves, hidden 5ths,
voice crossing/overlap, large leaps), motion classification (contrary,
oblique, similar, parallel), spacing analysis, and a composite voice
leading quality score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class VoiceLeadingRule(Enum):
    PARALLEL_FIFTHS = "parallel_fifths"
    PARALLEL_OCTAVES = "parallel_octaves"
    HIDDEN_FIFTHS = "hidden_fifths"
    HIDDEN_OCTAVES = "hidden_octaves"
    VOICE_CROSSING = "voice_crossing"
    VOICE_OVERLAP = "voice_overlap"
    LARGE_LEAP = "large_leap"
    UNRESOLVED_SEVENTH = "unresolved_seventh"


class MotionType(Enum):
    CONTRARY = "contrary"
    OBLIQUE = "oblique"
    SIMILAR = "similar"
    PARALLEL = "parallel"


@dataclass(frozen=True)
class VoiceLeadingError:
    beat_index: int
    rule: VoiceLeadingRule
    voice_a: str
    voice_b: str
    severity: float


@dataclass(frozen=True)
class SpacingReport:
    mean_spacing: float
    max_spacing: int
    min_spacing: int
    tenor_bass_gap: float
    close_position: bool


def classify_motion(motion_a: int, motion_b: int) -> MotionType:
    """Classify the relative motion between two voices."""
    if motion_a == 0 and motion_b == 0:
        return MotionType.OBLIQUE
    if motion_a == 0 or motion_b == 0:
        return MotionType.OBLIQUE
    if (motion_a > 0) != (motion_b > 0):
        return MotionType.CONTRARY
    if motion_a == motion_b:
        return MotionType.PARALLEL
    return MotionType.SIMILAR


def interval_class(semitones: int) -> int:
    """Map any interval to 0-6 (mod-12 interval class)."""
    ic = abs(semitones) % 12
    return min(ic, 12 - ic)


def check_parallel_motion(
    voice1: list[int], voice2: list[int],
) -> list[tuple[int, VoiceLeadingRule]]:
    """Detect parallel 5ths and 8ves between two voices."""
    violations = []
    for i in range(min(len(voice1), len(voice2)) - 1):
        interval_a = abs(voice1[i] - voice2[i]) % 12
        interval_b = abs(voice1[i + 1] - voice2[i + 1]) % 12
        motion1 = voice1[i + 1] - voice1[i]
        motion2 = voice2[i + 1] - voice2[i]
        same_direction = (motion1 > 0 and motion2 > 0) or (motion1 < 0 and motion2 < 0)
        if not same_direction:
            continue
        if interval_a == 7 and interval_b == 7:
            violations.append((i, VoiceLeadingRule.PARALLEL_FIFTHS))
        elif interval_a == 0 and interval_b == 0 and (motion1 != 0 or motion2 != 0):
            violations.append((i, VoiceLeadingRule.PARALLEL_OCTAVES))
    return violations


def detect_hidden_fifths_octaves(
    voice1: list[int], voice2: list[int],
) -> list[tuple[int, VoiceLeadingRule]]:
    """Detect hidden (direct) 5ths and 8ves — similar motion arriving at P5 or P8."""
    violations = []
    for i in range(min(len(voice1), len(voice2)) - 1):
        motion1 = voice1[i + 1] - voice1[i]
        motion2 = voice2[i + 1] - voice2[i]
        if motion1 == 0 or motion2 == 0:
            continue
        if (motion1 > 0) != (motion2 > 0):
            continue
        interval_b = abs(voice1[i + 1] - voice2[i + 1]) % 12
        if interval_b == 7:
            violations.append((i, VoiceLeadingRule.HIDDEN_FIFTHS))
        elif interval_b == 0:
            violations.append((i, VoiceLeadingRule.HIDDEN_OCTAVES))
    return violations


def voice_crossing_check(
    soprano: list[int], alto: list[int], tenor: list[int], bass: list[int],
) -> list[tuple[int, str, str]]:
    """Detect voice crossings at each beat index."""
    voices = {"soprano": soprano, "alto": alto, "tenor": tenor, "bass": bass}
    order = ["soprano", "alto", "tenor", "bass"]
    crossings = []
    n = min(len(v) for v in voices.values())
    for i in range(n):
        for j in range(len(order) - 1):
            upper = voices[order[j]][i]
            lower = voices[order[j + 1]][i]
            if upper < lower:
                crossings.append((i, order[j], order[j + 1]))
    return crossings


def detect_voice_overlap(
    voice_upper: list[int], voice_lower: list[int],
) -> list[int]:
    """Detect voice overlap: upper voice goes below lower voice's previous note."""
    overlaps = []
    for i in range(1, min(len(voice_upper), len(voice_lower))):
        if voice_upper[i] < voice_lower[i - 1]:
            overlaps.append(i)
        if voice_lower[i] > voice_upper[i - 1]:
            overlaps.append(i)
    return sorted(set(overlaps))


def detect_large_leaps(
    voice: list[int], max_leap: int = 12,
) -> list[tuple[int, int]]:
    """Find melodic leaps exceeding max_leap semitones. Returns (index, interval)."""
    leaps = []
    for i in range(len(voice) - 1):
        interval = abs(voice[i + 1] - voice[i])
        if interval > max_leap:
            leaps.append((i, interval))
    return leaps


def spacing_analysis(chord: list[int]) -> SpacingReport:
    """Analyze vertical spacing of a chord voicing."""
    if len(chord) < 2:
        return SpacingReport(0.0, 0, 0, 0.0, True)
    s = sorted(chord)
    intervals = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    tb_gap = float(s[-1] - s[-2]) if len(s) >= 2 else 0.0
    close = all(iv <= 4 for iv in intervals[:-1]) if len(intervals) > 1 else True
    return SpacingReport(
        mean_spacing=float(np.mean(intervals)),
        max_spacing=max(intervals),
        min_spacing=min(intervals),
        tenor_bass_gap=tb_gap,
        close_position=close,
    )


def motion_profile(
    voice1: list[int], voice2: list[int],
) -> dict[str, int]:
    """Count motion types between two voices across a passage."""
    counts = {mt.value: 0 for mt in MotionType}
    for i in range(min(len(voice1), len(voice2)) - 1):
        m1 = voice1[i + 1] - voice1[i]
        m2 = voice2[i + 1] - voice2[i]
        mt = classify_motion(m1, m2)
        counts[mt.value] += 1
    return counts


def smooth_voice_leading(
    from_chord: list[int], to_chord_pcs: list[int],
) -> list[int]:
    """Find voicing of to_chord_pcs minimizing total semitone motion from from_chord."""
    if not from_chord or not to_chord_pcs:
        return list(to_chord_pcs)
    result = []
    used = set()
    for src in from_chord:
        best_pitch = None
        best_dist = 999
        best_key = None
        for pc in to_chord_pcs:
            for octave_shift in range(-1, 2):
                candidate = pc + (src // 12) * 12 + octave_shift * 12
                dist = abs(candidate - src)
                key = (pc, octave_shift)
                if dist < best_dist and key not in used:
                    best_dist = dist
                    best_pitch = candidate
                    best_key = key
        if best_pitch is not None and best_key is not None:
            result.append(best_pitch)
            used.add(best_key)
        else:
            result.append(to_chord_pcs[0] + (src // 12) * 12)
    return result


def voice_leading_cost_matrix(
    chord_a: list[int], chord_b: list[int],
) -> list[list[int]]:
    """Pairwise absolute semitone distances between all voice pairs."""
    return [[abs(a - b) for b in chord_b] for a in chord_a]


def optimal_voicing(
    chord_pcs: list[int],
    prev_voicing: list[int],
    range_low: int = 36,
    range_high: int = 84,
) -> list[int]:
    """Find voicing of pitch classes minimizing motion from previous voicing."""
    candidates = []
    for pc in chord_pcs:
        octave_options = []
        p = pc % 12
        for midi in range(range_low, range_high + 1):
            if midi % 12 == p:
                octave_options.append(midi)
        candidates.append(octave_options)
    if not candidates or not prev_voicing:
        return [c[0] if c else 0 for c in candidates]
    best = None
    best_cost = float("inf")
    from itertools import product as iproduct
    for combo in iproduct(*candidates):
        cost = sum(abs(c - p) for c, p in zip(combo, prev_voicing[:len(combo)]))
        if cost < best_cost:
            best_cost = cost
            best = list(combo)
    return best or [c[0] if c else 0 for c in candidates]


def total_voice_motion(progression: list[list[int]]) -> float:
    """Sum of absolute semitone motion across all voices for a chord progression."""
    if len(progression) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(progression)):
        for a, b in zip(progression[i - 1], progression[i]):
            total += abs(b - a)
    return total


def voice_leading_score(
    progression: list[list[int]],
    parallel_penalty: float = 5.0,
    crossing_penalty: float = 3.0,
    leap_penalty: float = 2.0,
    leap_threshold: int = 12,
) -> float:
    """Composite voice-leading quality score (higher = better, max ~100).

    Penalizes parallel 5ths/8ves, voice crossings, and large leaps.
    """
    if len(progression) < 2:
        return 100.0
    n_voices = len(progression[0])
    n_beats = len(progression)
    total_penalty = 0.0
    for v in range(n_voices):
        voice = [progression[t][v] if v < len(progression[t]) else 0 for t in range(n_beats)]
        leaps = detect_large_leaps(voice, leap_threshold)
        total_penalty += len(leaps) * leap_penalty
    for va in range(n_voices):
        for vb in range(va + 1, n_voices):
            v1 = [progression[t][va] if va < len(progression[t]) else 0 for t in range(n_beats)]
            v2 = [progression[t][vb] if vb < len(progression[t]) else 0 for t in range(n_beats)]
            parallels = check_parallel_motion(v1, v2)
            total_penalty += len(parallels) * parallel_penalty
    for t in range(n_beats):
        chord = progression[t]
        for i in range(len(chord) - 1):
            if chord[i] < chord[i + 1]:
                total_penalty += crossing_penalty
    max_possible = n_beats * n_voices * (parallel_penalty + leap_penalty + crossing_penalty)
    if max_possible == 0:
        return 100.0
    return max(0.0, 100.0 * (1.0 - total_penalty / max_possible))
