"""Music theory tutoring: knowledge graphs, prerequisites, lesson sequencing, Bloom's taxonomy."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional
import numpy as np


class MasteryLevel(IntEnum):
    NOVICE = 0
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


class BloomLevel(IntEnum):
    REMEMBER = 1
    UNDERSTAND = 2
    APPLY = 3
    ANALYZE = 4
    EVALUATE = 5
    CREATE = 6


@dataclass
class Concept:
    concept_id: str
    name: str
    prerequisites: list[str] = field(default_factory=list)
    bloom_level: BloomLevel = BloomLevel.REMEMBER
    mastery: MasteryLevel = MasteryLevel.NOVICE
    attempts: int = 0
    correct: int = 0


MUSIC_THEORY_GRAPH: dict[str, list[str]] = {
    "pitch": [],
    "rhythm": [],
    "intervals": ["pitch"],
    "scales": ["intervals"],
    "key_signatures": ["scales"],
    "chords": ["intervals", "scales"],
    "triads": ["chords"],
    "seventh_chords": ["triads"],
    "inversions": ["triads"],
    "roman_numerals": ["key_signatures", "triads"],
    "progressions": ["roman_numerals"],
    "cadences": ["progressions"],
    "secondary_dominants": ["cadences"],
    "modulation": ["secondary_dominants", "key_signatures"],
    "form": ["cadences"],
    "counterpoint": ["intervals", "rhythm"],
    "species_counterpoint": ["counterpoint"],
    "voice_leading": ["triads", "counterpoint"],
    "figured_bass": ["inversions", "roman_numerals"],
    "meter": ["rhythm"],
    "simple_meter": ["meter"],
    "compound_meter": ["meter"],
    "syncopation": ["simple_meter"],
    "time_signatures": ["simple_meter", "compound_meter"],
}


def build_concept_graph() -> dict[str, Concept]:
    graph = {}
    for cid, prereqs in MUSIC_THEORY_GRAPH.items():
        graph[cid] = Concept(
            concept_id=cid,
            name=cid.replace("_", " ").title(),
            prerequisites=prereqs,
        )
    return graph


def topological_order(graph: dict[str, Concept]) -> list[str]:
    visited: set[str] = set()
    order: list[str] = []

    def visit(cid: str) -> None:
        if cid in visited:
            return
        visited.add(cid)
        if cid in graph:
            for prereq in graph[cid].prerequisites:
                visit(prereq)
        order.append(cid)

    for cid in graph:
        visit(cid)
    return order


def prerequisites_met(concept_id: str, graph: dict[str, Concept],
                      min_mastery: MasteryLevel = MasteryLevel.BEGINNER) -> bool:
    if concept_id not in graph:
        return True
    concept = graph[concept_id]
    for prereq_id in concept.prerequisites:
        if prereq_id not in graph:
            continue
        if graph[prereq_id].mastery < min_mastery:
            return False
    return True


def knowledge_gaps(graph: dict[str, Concept],
                   target_concept: str) -> list[str]:
    gaps = []
    visited: set[str] = set()

    def check(cid: str) -> None:
        if cid in visited:
            return
        visited.add(cid)
        if cid not in graph:
            return
        c = graph[cid]
        if c.mastery < MasteryLevel.BEGINNER:
            gaps.append(cid)
        for prereq in c.prerequisites:
            check(prereq)

    check(target_concept)
    return gaps


def lesson_sequence(graph: dict[str, Concept],
                    target: str) -> list[str]:
    gaps = knowledge_gaps(graph, target)
    order = topological_order(graph)
    gap_set = set(gaps)
    if target not in gap_set:
        gap_set.add(target)
    return [c for c in order if c in gap_set]


def update_mastery(concept: Concept, correct: bool) -> MasteryLevel:
    concept.attempts += 1
    if correct:
        concept.correct += 1
    if concept.attempts == 0:
        return MasteryLevel.NOVICE
    ratio = concept.correct / concept.attempts
    if concept.attempts < 3:
        concept.mastery = MasteryLevel.NOVICE
    elif ratio >= 0.95 and concept.attempts >= 20:
        concept.mastery = MasteryLevel.EXPERT
    elif ratio >= 0.85 and concept.attempts >= 10:
        concept.mastery = MasteryLevel.ADVANCED
    elif ratio >= 0.7 and concept.attempts >= 5:
        concept.mastery = MasteryLevel.INTERMEDIATE
    elif ratio >= 0.5:
        concept.mastery = MasteryLevel.BEGINNER
    else:
        concept.mastery = MasteryLevel.NOVICE
    return concept.mastery


def bloom_level_for_exercise(exercise_type: str) -> BloomLevel:
    mapping = {
        "identify": BloomLevel.REMEMBER,
        "define": BloomLevel.REMEMBER,
        "explain": BloomLevel.UNDERSTAND,
        "classify": BloomLevel.UNDERSTAND,
        "construct": BloomLevel.APPLY,
        "demonstrate": BloomLevel.APPLY,
        "analyze": BloomLevel.ANALYZE,
        "compare": BloomLevel.ANALYZE,
        "evaluate": BloomLevel.EVALUATE,
        "critique": BloomLevel.EVALUATE,
        "compose": BloomLevel.CREATE,
        "improvise": BloomLevel.CREATE,
    }
    return mapping.get(exercise_type, BloomLevel.REMEMBER)


def recommend_next_concepts(graph: dict[str, Concept], n: int = 3) -> list[str]:
    candidates = []
    for cid, concept in graph.items():
        if concept.mastery < MasteryLevel.INTERMEDIATE and prerequisites_met(cid, graph):
            priority = len(concept.prerequisites) - int(concept.mastery)
            candidates.append((priority, cid))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in candidates[:n]]
