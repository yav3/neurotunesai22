"""Species counterpoint: rule engine, voice leading analysis, first-species generation."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class MotionType(Enum):
    PARALLEL = "parallel"
    SIMILAR = "similar"
    CONTRARY = "contrary"
    OBLIQUE = "oblique"


CONSONANCES = {0, 3, 4, 7, 8, 9, 12}
PERFECT_CONSONANCES = {0, 7, 12}
IMPERFECT_CONSONANCES = {3, 4, 8, 9}


def classify_motion(v1_interval: int, v2_interval: int) -> MotionType:
    if v1_interval == 0 and v2_interval == 0:
        return MotionType.OBLIQUE
    if v1_interval == 0 or v2_interval == 0:
        return MotionType.OBLIQUE
    if (v1_interval > 0) == (v2_interval > 0):
        if v1_interval == v2_interval:
            return MotionType.PARALLEL
        return MotionType.SIMILAR
    return MotionType.CONTRARY


def is_consonant(interval_semitones: int) -> bool:
    return abs(interval_semitones) % 12 in CONSONANCES


def is_perfect(interval_semitones: int) -> bool:
    return abs(interval_semitones) % 12 in PERFECT_CONSONANCES


def check_parallel_fifths(v1: np.ndarray, v2: np.ndarray) -> list[tuple[int, int]]:
    violations = []
    intervals = (v2 - v1) % 12
    for i in range(len(intervals) - 1):
        if intervals[i] == 7 and intervals[i + 1] == 7:
            m1 = int(v1[i + 1] - v1[i])
            m2 = int(v2[i + 1] - v2[i])
            if m1 != 0 and classify_motion(m1, m2) == MotionType.PARALLEL:
                violations.append((i, 7))
    return violations


def check_parallel_octaves(v1: np.ndarray, v2: np.ndarray) -> list[tuple[int, int]]:
    violations = []
    intervals = (v2 - v1) % 12
    for i in range(len(intervals) - 1):
        if intervals[i] == 0 and intervals[i + 1] == 0:
            m1 = int(v1[i + 1] - v1[i])
            m2 = int(v2[i + 1] - v2[i])
            if m1 != 0 and classify_motion(m1, m2) == MotionType.PARALLEL:
                violations.append((i, 0))
    return violations


def check_voice_crossing(v1: np.ndarray, v2: np.ndarray) -> list[int]:
    return [int(i) for i in range(len(v1)) if v1[i] > v2[i]]


def check_hidden_fifths(v1: np.ndarray, v2: np.ndarray) -> list[int]:
    violations = []
    for i in range(len(v1) - 1):
        m1 = int(v1[i + 1] - v1[i])
        m2 = int(v2[i + 1] - v2[i])
        motion = classify_motion(m1, m2)
        interval = int(v2[i + 1] - v1[i + 1]) % 12
        if motion == MotionType.SIMILAR and interval in PERFECT_CONSONANCES and interval != 0:
            violations.append(i)
    return violations


@dataclass(frozen=True)
class CounterpointRules:
    allow_parallel_thirds: bool = True
    max_leap: int = 12
    prefer_stepwise_fraction: float = 0.6


def first_species(cantus_firmus: np.ndarray,
                  rules: CounterpointRules | None = None) -> np.ndarray:
    if rules is None:
        rules = CounterpointRules()
    n = len(cantus_firmus)
    if n == 0:
        return np.array([], dtype=float)

    cp = np.zeros(n)
    cp[0] = cantus_firmus[0] + 12  # start on octave
    cp[-1] = cantus_firmus[-1] + 12  # end on octave

    consonant_intervals = sorted(CONSONANCES - {0})  # exclude unison for variety

    for i in range(1, n - 1):
        cf_note = cantus_firmus[i]
        best_note = cf_note + 7  # fallback: fifth above
        best_score = -1000.0

        for iv in consonant_intervals:
            candidate = cf_note + iv
            score = 0.0
            motion_interval_cp = candidate - cp[i - 1]
            motion_interval_cf = cf_note - cantus_firmus[i - 1]
            motion = classify_motion(int(motion_interval_cf), int(motion_interval_cp))

            if motion == MotionType.CONTRARY:
                score += 3.0
            elif motion == MotionType.OBLIQUE:
                score += 2.0
            elif motion == MotionType.SIMILAR:
                score += 0.5

            leap = abs(motion_interval_cp)
            if leap <= 2:
                score += 2.0
            elif leap <= 4:
                score += 1.0
            elif leap > rules.max_leap:
                score -= 10.0

            if is_perfect(iv):
                if motion == MotionType.PARALLEL:
                    score -= 20.0
                if motion == MotionType.SIMILAR:
                    score -= 5.0

            if iv in IMPERFECT_CONSONANCES:
                score += 1.0

            if score > best_score:
                best_score = score
                best_note = candidate

        cp[i] = best_note

    return cp


@dataclass
class CounterpointReport:
    parallel_fifths: list[tuple[int, int]] = field(default_factory=list)
    parallel_octaves: list[tuple[int, int]] = field(default_factory=list)
    voice_crossings: list[int] = field(default_factory=list)
    hidden_fifths: list[int] = field(default_factory=list)
    motion_distribution: dict[str, int] = field(default_factory=dict)
    consonance_ratio: float = 0.0


class CounterpointAnalyzer:
    def analyze(self, voice1: np.ndarray, voice2: np.ndarray) -> CounterpointReport:
        report = CounterpointReport()
        report.parallel_fifths = check_parallel_fifths(voice1, voice2)
        report.parallel_octaves = check_parallel_octaves(voice1, voice2)
        report.voice_crossings = check_voice_crossing(voice1, voice2)
        report.hidden_fifths = check_hidden_fifths(voice1, voice2)

        motions = {m.value: 0 for m in MotionType}
        for i in range(len(voice1) - 1):
            m1 = int(voice1[i + 1] - voice1[i])
            m2 = int(voice2[i + 1] - voice2[i])
            mt = classify_motion(m1, m2)
            motions[mt.value] += 1
        report.motion_distribution = motions

        intervals = voice2 - voice1
        consonant_count = sum(1 for iv in intervals if is_consonant(int(iv)))
        report.consonance_ratio = consonant_count / max(len(intervals), 1)

        return report
