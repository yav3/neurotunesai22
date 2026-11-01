"""Constraint satisfaction for music generation.

Solve musical constraint satisfaction problems: voice leading rules,
harmonic rhythm, range constraints, parallel motion avoidance. Uses
backtracking search with arc consistency (AC-3). Pure numpy.

References: Pachet & Roy (2001) "Musical Harmonization with Constraints."
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple


class ConstraintType(Enum):
    HARD = "hard"
    SOFT = "soft"
    PREFERENCE = "preference"


@dataclass
class Variable:
    name: str
    domain: List[int]
    position: int


@dataclass
class Constraint:
    variables: Tuple[str, str]
    predicate: Callable[[int, int], bool]
    constraint_type: ConstraintType = ConstraintType.HARD
    weight: float = 1.0
    description: str = ""


@dataclass
class CSPSolution:
    assignment: Dict[str, int]
    violations: List[str]
    soft_score: float
    nodes_explored: int


@dataclass
class VoiceLeadingConstraints:
    max_leap: int = 7
    avoid_parallel_fifths: bool = True
    avoid_parallel_octaves: bool = True
    prefer_contrary_motion: bool = True
    max_voice_crossing: int = 0


def no_parallel_fifths(pitch_a1: int, pitch_a2: int,
                       pitch_b1: int, pitch_b2: int) -> bool:
    """True if the motion does NOT create parallel fifths."""
    interval1 = abs(pitch_a1 - pitch_b1) % 12
    interval2 = abs(pitch_a2 - pitch_b2) % 12
    if interval1 == 7 and interval2 == 7 and pitch_a1 != pitch_a2:
        return False
    return True


def no_parallel_octaves(pitch_a1: int, pitch_a2: int,
                        pitch_b1: int, pitch_b2: int) -> bool:
    """True if the motion does NOT create parallel octaves/unisons."""
    interval1 = abs(pitch_a1 - pitch_b1) % 12
    interval2 = abs(pitch_a2 - pitch_b2) % 12
    if interval1 == 0 and interval2 == 0 and pitch_a1 != pitch_a2:
        return False
    return True


def leap_constraint(max_semitones: int) -> Callable[[int, int], bool]:
    """Return a predicate that enforces a maximum melodic leap."""
    def check(prev_pitch: int, next_pitch: int) -> bool:
        return abs(next_pitch - prev_pitch) <= max_semitones
    return check


def range_constraint(low: int, high: int) -> Callable[[int, int], bool]:
    """Constraint that both values must be within the given pitch range."""
    def check(a: int, b: int) -> bool:
        return low <= a <= high and low <= b <= high
    return check


def scale_membership(scale_pcs: Set[int]) -> Callable[[int, int], bool]:
    """Both pitches must belong to the given scale pitch classes."""
    def check(a: int, b: int) -> bool:
        return (a % 12) in scale_pcs and (b % 12) in scale_pcs
    return check


def contrary_motion_score(motion_a: int, motion_b: int) -> float:
    """Score for contrary motion preference (1.0 = perfect contrary)."""
    if motion_a == 0 and motion_b == 0:
        return 0.5
    if (motion_a > 0 and motion_b < 0) or (motion_a < 0 and motion_b > 0):
        return 1.0
    if motion_a == 0 or motion_b == 0:
        return 0.7
    return 0.0


def arc_consistency_3(variables: Dict[str, Variable],
                      constraints: List[Constraint]) -> bool:
    """AC-3 algorithm: reduce domains by enforcing arc consistency.

    Returns False if any domain becomes empty (no solution exists).
    """
    constraint_map: Dict[Tuple[str, str], List[Constraint]] = {}
    for c in constraints:
        if c.constraint_type != ConstraintType.HARD:
            continue
        key = c.variables
        constraint_map.setdefault(key, []).append(c)
        rev = (key[1], key[0])
        constraint_map.setdefault(rev, []).append(c)

    queue = list(constraint_map.keys())

    while queue:
        xi_name, xj_name = queue.pop(0)
        xi = variables[xi_name]
        xj = variables[xj_name]

        removed = False
        new_domain = []
        for vi in xi.domain:
            has_support = False
            for vj in xj.domain:
                satisfied = True
                for c in constraint_map.get((xi_name, xj_name), []):
                    if not c.predicate(vi, vj):
                        satisfied = False
                        break
                if satisfied:
                    has_support = True
                    break
            if has_support:
                new_domain.append(vi)
            else:
                removed = True

        if removed:
            xi.domain = new_domain
            if not xi.domain:
                return False
            for c_key in constraint_map:
                if c_key[1] == xi_name and c_key[0] != xj_name:
                    queue.append(c_key)

    return True


def backtrack_search(variables: Dict[str, Variable],
                     constraints: List[Constraint],
                     assignment: Optional[Dict[str, int]] = None,
                     max_nodes: int = 10000) -> Optional[CSPSolution]:
    """Backtracking search with forward checking for musical CSP."""
    if assignment is None:
        assignment = {}

    nodes = [0]

    def _backtrack(assign: Dict[str, int]) -> Optional[Dict[str, int]]:
        if nodes[0] >= max_nodes:
            return None
        nodes[0] += 1

        if len(assign) == len(variables):
            return dict(assign)

        unassigned = [v for v in variables if v not in assign]
        var_name = min(unassigned, key=lambda v: len(variables[v].domain))
        var = variables[var_name]

        for value in var.domain:
            assign[var_name] = value
            consistent = True
            for c in constraints:
                if c.constraint_type != ConstraintType.HARD:
                    continue
                v1, v2 = c.variables
                if v1 in assign and v2 in assign:
                    if not c.predicate(assign[v1], assign[v2]):
                        consistent = False
                        break
            if consistent:
                result = _backtrack(assign)
                if result is not None:
                    return result
            del assign[var_name]

        return None

    result = _backtrack(dict(assignment))
    if result is None:
        return None

    violations = []
    soft_score = 0.0
    soft_count = 0
    for c in constraints:
        v1, v2 = c.variables
        if v1 in result and v2 in result:
            sat = c.predicate(result[v1], result[v2])
            if c.constraint_type == ConstraintType.SOFT:
                soft_count += 1
                if sat:
                    soft_score += c.weight
                else:
                    violations.append(c.description or f"Soft: {v1}-{v2}")
            elif c.constraint_type == ConstraintType.HARD and not sat:
                violations.append(c.description or f"Hard: {v1}-{v2}")

    return CSPSolution(
        assignment=result,
        violations=violations,
        soft_score=soft_score / max(soft_count, 1),
        nodes_explored=nodes[0],
    )


def harmonize_melody(melody_pitches: Sequence[int],
                     chord_tones_per_beat: Sequence[Set[int]],
                     voice_range: Tuple[int, int] = (48, 72),
                     max_leap: int = 5,
                     seed: int = 42) -> Optional[CSPSolution]:
    """Harmonize a melody by assigning bass notes satisfying voice-leading constraints."""
    variables: Dict[str, Variable] = {}
    constraints: List[Constraint] = []

    for i, (mel, tones) in enumerate(zip(melody_pitches, chord_tones_per_beat)):
        name = f"bass_{i}"
        domain = [p for p in range(voice_range[0], voice_range[1] + 1)
                  if (p % 12) in tones]
        variables[name] = Variable(name=name, domain=domain, position=i)

    for i in range(len(melody_pitches) - 1):
        constraints.append(Constraint(
            variables=(f"bass_{i}", f"bass_{i + 1}"),
            predicate=leap_constraint(max_leap),
            constraint_type=ConstraintType.HARD,
            description=f"Leap limit beats {i}-{i + 1}",
        ))

    return backtrack_search(variables, constraints)


def generate_with_constraints(
    n_beats: int,
    scale_pcs: Set[int],
    pitch_range: Tuple[int, int] = (60, 84),
    max_leap: int = 7,
    seed: int = 42,
) -> Optional[CSPSolution]:
    """Generate a melody satisfying scale, range, and leap constraints."""
    variables: Dict[str, Variable] = {}
    constraints: List[Constraint] = []

    domain = [p for p in range(pitch_range[0], pitch_range[1] + 1)
              if (p % 12) in scale_pcs]

    for i in range(n_beats):
        variables[f"note_{i}"] = Variable(
            name=f"note_{i}", domain=list(domain), position=i,
        )

    for i in range(n_beats - 1):
        constraints.append(Constraint(
            variables=(f"note_{i}", f"note_{i + 1}"),
            predicate=leap_constraint(max_leap),
            constraint_type=ConstraintType.HARD,
            description=f"Leap limit {i}-{i + 1}",
        ))

    if not arc_consistency_3(variables, constraints):
        return None

    return backtrack_search(variables, constraints)
