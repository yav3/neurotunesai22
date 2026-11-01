"""Curriculum sequencing for music education — adaptive skill progression and prerequisite modeling."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import numpy as np


class SkillDomain(str, Enum):
    RHYTHM = "rhythm"
    MELODY = "melody"
    HARMONY = "harmony"
    TECHNIQUE = "technique"
    EAR_TRAINING = "ear_training"
    THEORY = "theory"
    SIGHT_READING = "sight_reading"
    IMPROVISATION = "improvisation"
    COMPOSITION = "composition"


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


DIFFICULTY_LEVELS = {
    Difficulty.BEGINNER: 1,
    Difficulty.ELEMENTARY: 2,
    Difficulty.INTERMEDIATE: 3,
    Difficulty.ADVANCED: 4,
    Difficulty.EXPERT: 5,
}


@dataclass
class Skill:
    id: str
    name: str
    domain: SkillDomain
    difficulty: Difficulty
    prerequisites: List[str] = field(default_factory=list)
    estimated_hours: float = 1.0
    mastery_threshold: float = 0.8


@dataclass
class LearnerState:
    skill_mastery: Dict[str, float] = field(default_factory=dict)
    completed_exercises: int = 0
    total_practice_hours: float = 0.0
    learning_rate: float = 1.0
    preferred_domains: List[SkillDomain] = field(default_factory=list)


@dataclass
class Exercise:
    id: str
    skill_id: str
    difficulty: float
    estimated_minutes: float = 15.0
    description: str = ""


@dataclass
class LessonPlan:
    skills: List[str]
    exercises: List[Exercise]
    estimated_hours: float
    difficulty_curve: List[float]


def build_prerequisite_graph(skills: List[Skill]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for s in skills:
        graph[s.id] = set(s.prerequisites)
    return graph


def topological_sort(skills: List[Skill]) -> List[str]:
    graph = build_prerequisite_graph(skills)
    skill_map = {s.id: s for s in skills}
    in_degree: Dict[str, int] = {s.id: 0 for s in skills}
    for sid, prereqs in graph.items():
        for p in prereqs:
            if p in in_degree:
                pass
    for sid, prereqs in graph.items():
        in_degree[sid] = len([p for p in prereqs if p in skill_map])

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    queue.sort(key=lambda s: DIFFICULTY_LEVELS.get(skill_map[s].difficulty, 0))
    result = []
    visited = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        result.append(current)
        for sid, prereqs in graph.items():
            if current in prereqs and sid not in visited:
                remaining = len([p for p in prereqs if p not in visited])
                if remaining == 0:
                    queue.append(sid)
        queue.sort(key=lambda s: DIFFICULTY_LEVELS.get(skill_map[s].difficulty, 0))

    return result


def prerequisite_satisfaction(
    skill: Skill,
    learner: LearnerState,
) -> float:
    if not skill.prerequisites:
        return 1.0
    satisfied = sum(
        1.0 for p in skill.prerequisites
        if learner.skill_mastery.get(p, 0.0) >= 0.7
    )
    return satisfied / len(skill.prerequisites)


def readiness_score(
    skill: Skill,
    learner: LearnerState,
) -> float:
    prereq_sat = prerequisite_satisfaction(skill, learner)
    current_mastery = learner.skill_mastery.get(skill.id, 0.0)
    domain_pref = 1.2 if skill.domain in learner.preferred_domains else 1.0
    difficulty_match = 1.0 - abs(
        DIFFICULTY_LEVELS.get(skill.difficulty, 3) / 5.0
        - (current_mastery + 0.2)
    )
    difficulty_match = max(0.0, difficulty_match)
    return prereq_sat * 0.5 + difficulty_match * 0.3 + (1.0 - current_mastery) * 0.2 * domain_pref


def next_skills(
    skills: List[Skill],
    learner: LearnerState,
    n: int = 5,
) -> List[Tuple[str, float]]:
    candidates = []
    for s in skills:
        mastery = learner.skill_mastery.get(s.id, 0.0)
        if mastery >= s.mastery_threshold:
            continue
        score = readiness_score(s, learner)
        if prerequisite_satisfaction(s, learner) >= 0.7:
            candidates.append((s.id, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]


def spaced_repetition_interval(
    n_reviews: int,
    ease_factor: float = 2.5,
    last_interval: float = 1.0,
) -> float:
    if n_reviews == 0:
        return 1.0
    if n_reviews == 1:
        return 6.0
    return last_interval * ease_factor


def update_mastery(
    current: float,
    performance: float,
    learning_rate: float = 1.0,
) -> float:
    delta = (performance - current) * 0.3 * learning_rate
    new = current + delta
    return float(np.clip(new, 0.0, 1.0))


def difficulty_progression(
    mastery_history: List[float],
    current_difficulty: float,
) -> float:
    if len(mastery_history) < 3:
        return current_difficulty
    recent = mastery_history[-3:]
    avg = np.mean(recent)
    if avg > 0.85:
        return min(current_difficulty + 0.1, 1.0)
    elif avg < 0.5:
        return max(current_difficulty - 0.15, 0.0)
    return current_difficulty


def zone_of_proximal_development(
    skill_mastery: float,
    difficulty: float,
) -> float:
    gap = difficulty - skill_mastery
    if gap < -0.2:
        return 0.3
    if gap > 0.4:
        return 0.2
    return 1.0 - abs(gap - 0.1) / 0.5


def generate_lesson_plan(
    skills: List[Skill],
    learner: LearnerState,
    target_hours: float = 1.0,
    exercises_per_skill: int = 3,
) -> LessonPlan:
    recommended = next_skills(skills, learner, n=5)
    skill_map = {s.id: s for s in skills}

    plan_skills = []
    exercises = []
    total_hours = 0.0
    difficulty_curve = []

    for sid, score in recommended:
        if total_hours >= target_hours:
            break
        s = skill_map.get(sid)
        if s is None:
            continue
        plan_skills.append(sid)
        mastery = learner.skill_mastery.get(sid, 0.0)
        for j in range(exercises_per_skill):
            ex_diff = mastery + 0.1 * (j + 1)
            ex_diff = min(ex_diff, 1.0)
            minutes = s.estimated_hours * 60 / exercises_per_skill
            exercises.append(Exercise(
                id=f"{sid}_ex{j}",
                skill_id=sid,
                difficulty=ex_diff,
                estimated_minutes=minutes,
            ))
            difficulty_curve.append(ex_diff)
            total_hours += minutes / 60.0

    return LessonPlan(
        skills=plan_skills,
        exercises=exercises,
        estimated_hours=total_hours,
        difficulty_curve=difficulty_curve,
    )


def learning_efficiency(
    mastery_gains: List[float],
    practice_hours: List[float],
) -> float:
    if not mastery_gains or not practice_hours:
        return 0.0
    total_gain = sum(mastery_gains)
    total_hours = sum(practice_hours)
    if total_hours < 1e-9:
        return 0.0
    return total_gain / total_hours


def skill_gap_analysis(
    skills: List[Skill],
    learner: LearnerState,
) -> Dict[SkillDomain, float]:
    domain_gaps: Dict[SkillDomain, List[float]] = {}
    for s in skills:
        mastery = learner.skill_mastery.get(s.id, 0.0)
        gap = s.mastery_threshold - mastery
        if s.domain not in domain_gaps:
            domain_gaps[s.domain] = []
        domain_gaps[s.domain].append(max(gap, 0.0))
    return {d: float(np.mean(gaps)) for d, gaps in domain_gaps.items()}


def estimated_completion_time(
    skills: List[Skill],
    learner: LearnerState,
) -> float:
    total = 0.0
    for s in skills:
        mastery = learner.skill_mastery.get(s.id, 0.0)
        remaining = max(s.mastery_threshold - mastery, 0.0)
        hours = s.estimated_hours * remaining / max(s.mastery_threshold, 0.01)
        total += hours / max(learner.learning_rate, 0.1)
    return total


def mastery_decay(
    current_mastery: float,
    days_since_practice: float,
    decay_rate: float = 0.05,
) -> float:
    return current_mastery * math.exp(-decay_rate * days_since_practice)
