"""Species counterpoint: cantus firmus analysis, parallel motion detection,
voice independence metrics, suspension patterns, Fux-style rule checking.

Pure numpy implementation of traditional counterpoint rules and scoring.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class IntervalQuality(Enum):
    PERFECT_CONSONANCE = "perfect_consonance"
    IMPERFECT_CONSONANCE = "imperfect_consonance"
    DISSONANCE = "dissonance"


class MotionType(Enum):
    PARALLEL = "parallel"
    SIMILAR = "similar"
    CONTRARY = "contrary"
    OBLIQUE = "oblique"


PERFECT_CONSONANCES = {0, 7, 12}  # unison, P5, P8
IMPERFECT_CONSONANCES = {3, 4, 8, 9}  # m3, M3, m6, M6
DISSONANCES = {1, 2, 5, 6, 10, 11}  # m2, M2, P4, tritone, m7, M7


@dataclass
class CantusFirmusAnalysis:
    pitches: np.ndarray
    length: int
    range_semitones: int
    climax_position: float
    step_ratio: float
    leap_ratio: float
    largest_leap: int
    starts_on_tonic: bool
    ends_on_tonic: bool
    has_single_climax: bool
    score: float


@dataclass
class CounterpointAnalysis:
    intervals: np.ndarray
    motion_types: List[MotionType]
    parallel_fifths: List[int]
    parallel_octaves: List[int]
    contrary_motion_pct: float
    voice_crossings: int
    dissonance_count: int
    score: float
    violations: List[str] = field(default_factory=list)


@dataclass
class SuspensionPattern:
    position: int
    preparation_interval: int
    dissonance_interval: int
    resolution_interval: int
    is_valid: bool


def classify_interval(semitones: int) -> IntervalQuality:
    """Classify an interval by its consonance/dissonance quality."""
    ic = abs(semitones) % 12
    if ic in PERFECT_CONSONANCES:
        return IntervalQuality.PERFECT_CONSONANCE
    elif ic in IMPERFECT_CONSONANCES:
        return IntervalQuality.IMPERFECT_CONSONANCE
    return IntervalQuality.DISSONANCE


def interval_between(p1: int, p2: int) -> int:
    """Directed interval in semitones."""
    return p2 - p1


def classify_motion(lower1: int, lower2: int,
                    upper1: int, upper2: int) -> MotionType:
    """Classify the type of motion between two voice pairs."""
    lower_motion = lower2 - lower1
    upper_motion = upper2 - upper1
    if lower_motion == 0 and upper_motion == 0:
        return MotionType.OBLIQUE
    if lower_motion == 0 or upper_motion == 0:
        return MotionType.OBLIQUE
    if lower_motion == upper_motion:
        return MotionType.PARALLEL
    if (lower_motion > 0) == (upper_motion > 0):
        return MotionType.SIMILAR
    return MotionType.CONTRARY


def analyze_cantus_firmus(pitches: np.ndarray,
                          tonic: int = 60) -> CantusFirmusAnalysis:
    """Evaluate a cantus firmus against traditional rules."""
    n = len(pitches)
    intervals = np.diff(pitches)
    abs_intervals = np.abs(intervals)

    steps = np.sum(abs_intervals <= 2)
    leaps = np.sum(abs_intervals > 2)
    step_ratio = float(steps / max(n - 1, 1))
    leap_ratio = float(leaps / max(n - 1, 1))
    largest_leap = int(np.max(abs_intervals)) if len(abs_intervals) > 0 else 0

    pitch_range = int(np.max(pitches) - np.min(pitches))
    climax_idx = int(np.argmax(pitches))
    climax_position = climax_idx / max(n - 1, 1)

    starts_on_tonic = (pitches[0] % 12) == (tonic % 12)
    ends_on_tonic = (pitches[-1] % 12) == (tonic % 12)

    climax_count = int(np.sum(pitches == np.max(pitches)))
    has_single_climax = climax_count == 1

    score = 100.0
    if not starts_on_tonic:
        score -= 15
    if not ends_on_tonic:
        score -= 15
    if not has_single_climax:
        score -= 10
    if pitch_range > 16:
        score -= 10
    if pitch_range < 5:
        score -= 10
    if step_ratio < 0.5:
        score -= 15
    if largest_leap > 8:
        score -= 10
    if climax_position < 0.2 or climax_position > 0.8:
        score -= 5

    return CantusFirmusAnalysis(
        pitches=pitches, length=n, range_semitones=pitch_range,
        climax_position=climax_position, step_ratio=step_ratio,
        leap_ratio=leap_ratio, largest_leap=largest_leap,
        starts_on_tonic=starts_on_tonic, ends_on_tonic=ends_on_tonic,
        has_single_climax=has_single_climax, score=max(score, 0.0),
    )


def detect_parallel_fifths(lower: np.ndarray,
                           upper: np.ndarray) -> List[int]:
    """Find positions where parallel perfect fifths occur."""
    positions = []
    for i in range(len(lower) - 1):
        iv1 = abs(upper[i] - lower[i]) % 12
        iv2 = abs(upper[i + 1] - lower[i + 1]) % 12
        if iv1 == 7 and iv2 == 7:
            motion = classify_motion(lower[i], lower[i + 1],
                                     upper[i], upper[i + 1])
            if motion == MotionType.PARALLEL:
                positions.append(i)
    return positions


def detect_parallel_octaves(lower: np.ndarray,
                            upper: np.ndarray) -> List[int]:
    """Find positions where parallel octaves/unisons occur."""
    positions = []
    for i in range(len(lower) - 1):
        iv1 = abs(upper[i] - lower[i]) % 12
        iv2 = abs(upper[i + 1] - lower[i + 1]) % 12
        if iv1 in (0, 12) and iv2 in (0, 12):
            motion = classify_motion(lower[i], lower[i + 1],
                                     upper[i], upper[i + 1])
            if motion == MotionType.PARALLEL:
                positions.append(i)
    return positions


def voice_crossings(lower: np.ndarray, upper: np.ndarray) -> int:
    """Count positions where voices cross."""
    return int(np.sum(lower > upper))


def analyze_counterpoint(lower: np.ndarray,
                         upper: np.ndarray) -> CounterpointAnalysis:
    """Full counterpoint analysis between two voices."""
    n = min(len(lower), len(upper))
    intervals = upper[:n] - lower[:n]
    motions = []
    for i in range(n - 1):
        motions.append(classify_motion(lower[i], lower[i + 1],
                                       upper[i], upper[i + 1]))

    par5 = detect_parallel_fifths(lower[:n], upper[:n])
    par8 = detect_parallel_octaves(lower[:n], upper[:n])
    crossings = voice_crossings(lower[:n], upper[:n])

    contrary_count = sum(1 for m in motions if m == MotionType.CONTRARY)
    contrary_pct = contrary_count / max(len(motions), 1)

    diss_count = sum(1 for iv in intervals
                     if classify_interval(iv) == IntervalQuality.DISSONANCE)

    violations = []
    if par5:
        violations.append(f"parallel_fifths_at_{par5}")
    if par8:
        violations.append(f"parallel_octaves_at_{par8}")
    if crossings > 0:
        violations.append(f"voice_crossings_{crossings}")
    if contrary_pct < 0.3:
        violations.append("insufficient_contrary_motion")

    score = 100.0 - 15 * len(par5) - 15 * len(par8) - 5 * crossings
    if contrary_pct < 0.3:
        score -= 10
    score -= 2 * diss_count

    return CounterpointAnalysis(
        intervals=intervals, motion_types=motions,
        parallel_fifths=par5, parallel_octaves=par8,
        contrary_motion_pct=contrary_pct,
        voice_crossings=crossings, dissonance_count=diss_count,
        score=max(score, 0.0), violations=violations,
    )


def detect_suspensions(lower: np.ndarray, upper: np.ndarray,
                       beats_per_measure: int = 4) -> List[SuspensionPattern]:
    """Detect suspension patterns (preparation–dissonance–resolution)."""
    suspensions = []
    n = min(len(lower), len(upper))
    for i in range(1, n - 1):
        prep_iv = abs(upper[i - 1] - lower[i - 1]) % 12
        diss_iv = abs(upper[i] - lower[i]) % 12
        res_iv = abs(upper[i + 1] - lower[i + 1]) % 12

        if (classify_interval(prep_iv) != IntervalQuality.DISSONANCE and
                classify_interval(diss_iv) == IntervalQuality.DISSONANCE and
                classify_interval(res_iv) != IntervalQuality.DISSONANCE):
            prep_held = (upper[i - 1] == upper[i] or lower[i - 1] == lower[i])
            resolves_down = (upper[i + 1] < upper[i] or lower[i + 1] < lower[i])
            is_valid = prep_held and resolves_down
            suspensions.append(SuspensionPattern(
                position=i, preparation_interval=prep_iv,
                dissonance_interval=diss_iv, resolution_interval=res_iv,
                is_valid=is_valid,
            ))
    return suspensions


def voice_independence(lower: np.ndarray, upper: np.ndarray) -> float:
    """Score 0–1 measuring voice independence (contrary motion, no crossing)."""
    n = min(len(lower), len(upper))
    if n < 2:
        return 0.0
    motions = [classify_motion(lower[i], lower[i + 1], upper[i], upper[i + 1])
               for i in range(n - 1)]
    contrary = sum(1 for m in motions if m == MotionType.CONTRARY)
    oblique = sum(1 for m in motions if m == MotionType.OBLIQUE)
    crossing_penalty = voice_crossings(lower[:n], upper[:n]) / n
    independence = (contrary + 0.5 * oblique) / max(len(motions), 1)
    return float(max(0.0, min(1.0, independence - crossing_penalty)))


def step_skip_ratio(melody: np.ndarray) -> float:
    """Ratio of stepwise motion (<=2 semitones) to total motion."""
    intervals = np.abs(np.diff(melody))
    if len(intervals) == 0:
        return 0.0
    return float(np.sum(intervals <= 2) / len(intervals))
