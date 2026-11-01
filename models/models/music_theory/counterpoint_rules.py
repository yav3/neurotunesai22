"""Species counterpoint rule checking (Fux-style) with scoring."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum


class Species(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
    FIFTH = 5


PERFECT_CONSONANCES = {0, 7, 12}
IMPERFECT_CONSONANCES = {3, 4, 8, 9}
DISSONANCES = {1, 2, 5, 6, 10, 11}


class MotionKind(Enum):
    PARALLEL = "parallel"
    SIMILAR = "similar"
    CONTRARY = "contrary"
    OBLIQUE = "oblique"


def classify_motion(cf_interval: int, cp_interval: int) -> MotionKind:
    if cf_interval == 0:
        return MotionKind.OBLIQUE
    if cp_interval == 0:
        return MotionKind.OBLIQUE
    if (cf_interval > 0) == (cp_interval > 0):
        if cf_interval == cp_interval:
            return MotionKind.PARALLEL
        return MotionKind.SIMILAR
    return MotionKind.CONTRARY


def interval_class(a: int, b: int) -> int:
    return abs(b - a) % 12


@dataclass
class CounterpointViolation:
    bar: int
    rule: str
    severity: float


def check_parallel_perfects(cf: list[int], cp: list[int]) -> list[CounterpointViolation]:
    violations: list[CounterpointViolation] = []
    for i in range(1, min(len(cf), len(cp))):
        ic_prev = interval_class(cf[i - 1], cp[i - 1])
        ic_curr = interval_class(cf[i], cp[i])
        if ic_curr in PERFECT_CONSONANCES and ic_prev == ic_curr:
            motion = classify_motion(cf[i] - cf[i - 1], cp[i] - cp[i - 1])
            if motion == MotionKind.PARALLEL:
                violations.append(CounterpointViolation(
                    bar=i, rule="parallel_perfect", severity=1.0
                ))
    return violations


def check_hidden_fifths_octaves(cf: list[int], cp: list[int]) -> list[CounterpointViolation]:
    violations: list[CounterpointViolation] = []
    for i in range(1, min(len(cf), len(cp))):
        ic = interval_class(cf[i], cp[i])
        if ic in {0, 7}:
            motion = classify_motion(cf[i] - cf[i - 1], cp[i] - cp[i - 1])
            if motion == MotionKind.SIMILAR:
                cp_step = abs(cp[i] - cp[i - 1])
                if cp_step > 2:
                    violations.append(CounterpointViolation(
                        bar=i, rule="hidden_perfect", severity=0.7
                    ))
    return violations


def check_voice_crossing(cf: list[int], cp: list[int],
                          cp_above: bool = True) -> list[CounterpointViolation]:
    violations: list[CounterpointViolation] = []
    for i in range(min(len(cf), len(cp))):
        if cp_above and cp[i] < cf[i]:
            violations.append(CounterpointViolation(bar=i, rule="voice_crossing", severity=0.8))
        elif not cp_above and cp[i] > cf[i]:
            violations.append(CounterpointViolation(bar=i, rule="voice_crossing", severity=0.8))
    return violations


def check_leap_resolution(cp: list[int], leap_threshold: int = 4) -> list[CounterpointViolation]:
    violations: list[CounterpointViolation] = []
    for i in range(1, len(cp) - 1):
        leap = cp[i] - cp[i - 1]
        if abs(leap) >= leap_threshold:
            resolution = cp[i + 1] - cp[i]
            if np.sign(resolution) == np.sign(leap) or abs(resolution) > 2:
                violations.append(CounterpointViolation(
                    bar=i, rule="unresolved_leap", severity=0.6
                ))
    return violations


def check_tritone_outline(cp: list[int], window: int = 4) -> list[CounterpointViolation]:
    violations: list[CounterpointViolation] = []
    for i in range(len(cp) - window + 1):
        segment = cp[i:i + window]
        outline_interval = interval_class(min(segment), max(segment))
        if outline_interval == 6:
            violations.append(CounterpointViolation(
                bar=i, rule="tritone_outline", severity=0.5
            ))
    return violations


def first_species_check(cf: list[int], cp: list[int],
                         cp_above: bool = True) -> list[CounterpointViolation]:
    violations: list[CounterpointViolation] = []
    violations.extend(check_parallel_perfects(cf, cp))
    violations.extend(check_hidden_fifths_octaves(cf, cp))
    violations.extend(check_voice_crossing(cf, cp, cp_above))
    violations.extend(check_leap_resolution(cp))
    violations.extend(check_tritone_outline(cp))
    for i in range(min(len(cf), len(cp))):
        ic = interval_class(cf[i], cp[i])
        if ic in DISSONANCES:
            violations.append(CounterpointViolation(
                bar=i, rule="dissonant_interval", severity=0.9
            ))
    if len(cf) >= 1 and len(cp) >= 1:
        ic_first = interval_class(cf[0], cp[0])
        if ic_first not in {0, 7}:
            violations.append(CounterpointViolation(
                bar=0, rule="opening_not_perfect", severity=0.8
            ))
    if len(cf) >= 1 and len(cp) >= 1:
        ic_last = interval_class(cf[-1], cp[-1])
        if ic_last not in {0, 7}:
            violations.append(CounterpointViolation(
                bar=len(cf) - 1, rule="closing_not_perfect", severity=0.8
            ))
    return violations


def counterpoint_score(violations: list[CounterpointViolation]) -> float:
    if not violations:
        return 1.0
    total_penalty = sum(v.severity for v in violations)
    return max(0.0, 1.0 - total_penalty * 0.1)


def contrary_motion_ratio(cf: list[int], cp: list[int]) -> float:
    n = min(len(cf), len(cp)) - 1
    if n <= 0:
        return 0.0
    count = 0
    for i in range(n):
        m = classify_motion(cf[i + 1] - cf[i], cp[i + 1] - cp[i])
        if m == MotionKind.CONTRARY:
            count += 1
    return count / n


def melodic_variety(line: list[int]) -> float:
    if len(line) < 2:
        return 0.0
    intervals = [line[i + 1] - line[i] for i in range(len(line) - 1)]
    unique = len(set(intervals))
    return unique / len(intervals)
