"""Musical syntax processing: GTTM-inspired structural analysis.

Implements computational models of musical syntax based on Lerdahl &
Jackendoff's Generative Theory of Tonal Music (GTTM), Rohrmeier's
generative syntax model, and psycholinguistic parallels (ERAN/ELAN
violation detection). Covers well-formedness rules, preference rules,
prolongational reduction, and syntactic complexity metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class ProlongationType(Enum):
    STRONG = "strong_prolongation"
    WEAK = "weak_prolongation"
    PROGRESSION = "progression"
    DEPARTURE = "departure"


class SyntacticRelation(Enum):
    ELABORATION = "elaboration"
    PROLONGATION = "prolongation"
    CADENCE = "cadence"
    SEQUENCE = "sequence"
    PARENTHETICAL = "parenthetical"


@dataclass(frozen=True)
class SyntacticEvent:
    time: float
    pitch_class: int
    chord_root: int
    chord_quality: str
    metrical_weight: float = 0.5
    is_cadential: bool = False


@dataclass
class SyntacticViolation:
    position: int
    expected_pc: int
    observed_pc: int
    violation_type: str
    severity: float


@dataclass
class ProlongationalNode:
    event_idx: int
    prolongation_type: ProlongationType
    children: list[int] = field(default_factory=list)
    depth: int = 0


DIATONIC_SETS = {
    "major": {0, 2, 4, 5, 7, 9, 11},
    "minor": {0, 2, 3, 5, 7, 8, 10},
}


def build_key_set(root: int, mode: str = "major") -> set[int]:
    template = DIATONIC_SETS.get(mode, DIATONIC_SETS["major"])
    return {(root + pc) % 12 for pc in template}


def detect_syntactic_violations(
    events: Sequence[SyntacticEvent],
    key_root: int = 0,
    mode: str = "major",
) -> list[SyntacticViolation]:
    key_set = build_key_set(key_root, mode)
    violations = []
    for i, ev in enumerate(events):
        if ev.pitch_class not in key_set:
            closest = min(key_set, key=lambda pc: min(
                abs(ev.pitch_class - pc), 12 - abs(ev.pitch_class - pc)))
            severity = ev.metrical_weight
            violations.append(SyntacticViolation(
                position=i,
                expected_pc=closest,
                observed_pc=ev.pitch_class,
                violation_type="chromatic_nonchord",
                severity=severity,
            ))
    return violations


def harmonic_syntax_score(chord_roots: Sequence[int],
                           key_root: int = 0) -> float:
    if len(chord_roots) < 2:
        return 1.0
    common_progressions = {
        (0, 5): 0.9,   # I -> IV
        (0, 7): 0.95,  # I -> V
        (5, 0): 0.85,  # IV -> I
        (7, 0): 1.0,   # V -> I
        (7, 5): 0.5,   # V -> IV (retrogression)
        (2, 7): 0.85,  # ii -> V
        (9, 2): 0.8,   # vi -> ii
        (4, 0): 0.7,   # iii -> I
        (5, 7): 0.9,   # IV -> V
        (2, 0): 0.6,   # ii -> I
    }
    total_score = 0.0
    n = 0
    for i in range(len(chord_roots) - 1):
        rel_from = (chord_roots[i] - key_root) % 12
        rel_to = (chord_roots[i + 1] - key_root) % 12
        pair = (rel_from, rel_to)
        total_score += common_progressions.get(pair, 0.3)
        n += 1
    return total_score / max(n, 1)


def prolongational_reduction(
    events: Sequence[SyntacticEvent],
) -> list[ProlongationalNode]:
    if not events:
        return []
    nodes = []
    metrical_weights = [ev.metrical_weight for ev in events]
    head_idx = int(np.argmax(metrical_weights))
    head_node = ProlongationalNode(
        event_idx=head_idx,
        prolongation_type=ProlongationType.STRONG,
        depth=0,
    )
    nodes.append(head_node)
    for i, ev in enumerate(events):
        if i == head_idx:
            continue
        if ev.chord_root == events[head_idx].chord_root:
            if ev.chord_quality == events[head_idx].chord_quality:
                ptype = ProlongationType.STRONG
            else:
                ptype = ProlongationType.WEAK
        elif ev.is_cadential:
            ptype = ProlongationType.PROGRESSION
        else:
            ptype = ProlongationType.DEPARTURE
        child = ProlongationalNode(
            event_idx=i,
            prolongation_type=ptype,
            depth=1,
        )
        head_node.children.append(len(nodes))
        nodes.append(child)
    return nodes


def syntactic_complexity(events: Sequence[SyntacticEvent],
                          key_root: int = 0) -> float:
    if len(events) < 2:
        return 0.0
    violations = detect_syntactic_violations(events, key_root)
    violation_density = len(violations) / len(events)
    chord_roots = [ev.chord_root for ev in events]
    syntax_score = harmonic_syntax_score(chord_roots, key_root)
    nodes = prolongational_reduction(events)
    departure_count = sum(
        1 for n in nodes if n.prolongation_type == ProlongationType.DEPARTURE
    )
    departure_ratio = departure_count / max(len(nodes), 1)
    return float(np.clip(
        0.3 * violation_density + 0.4 * (1.0 - syntax_score) + 0.3 * departure_ratio,
        0, 1,
    ))


def cadence_detection(chord_roots: Sequence[int], key_root: int = 0) -> list[dict]:
    cadences = []
    for i in range(len(chord_roots) - 1):
        rel_from = (chord_roots[i] - key_root) % 12
        rel_to = (chord_roots[i + 1] - key_root) % 12
        if rel_from == 7 and rel_to == 0:
            cadences.append({"position": i + 1, "type": "authentic", "strength": 1.0})
        elif rel_from == 5 and rel_to == 0:
            cadences.append({"position": i + 1, "type": "plagal", "strength": 0.7})
        elif rel_to == 7:
            cadences.append({"position": i + 1, "type": "half", "strength": 0.6})
        elif rel_from == 7 and rel_to != 0:
            cadences.append({"position": i + 1, "type": "deceptive", "strength": 0.8})
    return cadences


def hierarchical_tension(events: Sequence[SyntacticEvent],
                          key_root: int = 0) -> np.ndarray:
    if not events:
        return np.array([], dtype=np.float64)
    n = len(events)
    tension = np.zeros(n, dtype=np.float64)
    key_set = build_key_set(key_root)
    tonic_distance = {
        0: 0.0, 7: 0.2, 5: 0.25, 2: 0.4, 9: 0.45,
        4: 0.5, 11: 0.55, 10: 0.6, 8: 0.65, 3: 0.7, 1: 0.8, 6: 0.9,
    }
    for i, ev in enumerate(events):
        rel_root = (ev.chord_root - key_root) % 12
        harmonic_t = tonic_distance.get(rel_root, 0.75)
        chromatic_t = 0.0 if ev.pitch_class in key_set else 0.3
        metrical_t = 1.0 - ev.metrical_weight
        tension[i] = 0.5 * harmonic_t + 0.25 * chromatic_t + 0.25 * metrical_t
    return tension


def expectancy_violation_eran(
    observed_pcs: Sequence[int],
    expected_distribution: np.ndarray,
) -> np.ndarray:
    if not observed_pcs:
        return np.array([], dtype=np.float64)
    dist = np.asarray(expected_distribution, dtype=np.float64)
    dist = dist / (dist.sum() + 1e-15)
    violations = np.zeros(len(observed_pcs), dtype=np.float64)
    for i, pc in enumerate(observed_pcs):
        prob = dist[pc % 12]
        violations[i] = -np.log2(prob + 1e-15)
    return violations
