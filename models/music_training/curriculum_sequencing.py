"""Adaptive music curriculum sequencing.

Models optimal ordering of practice material for skill acquisition
using spacing effects (Ebbinghaus), zone of proximal development
(Vygotsky), interleaving (contextual interference), and the power
law of practice.  Pure numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class SkillLevel(Enum):
    BEGINNER = 0
    INTERMEDIATE = 1
    ADVANCED = 2
    EXPERT = 3


@dataclass
class LearningObjective:
    skill_name: str
    target_level: float
    prerequisite_skills: List[str] = field(default_factory=list)
    difficulty: float = 0.5


@dataclass
class StudentState:
    skills: Dict[str, float] = field(default_factory=dict)
    practice_hours: float = 0.0
    session_count: int = 0
    last_practiced: Dict[str, float] = field(default_factory=dict)


@dataclass
class CurriculumItem:
    objective: LearningObjective
    estimated_duration: float
    priority: float


def zone_of_proximal_development(current_skill: float, target: float) -> float:
    gap = target - current_skill
    if gap <= 0:
        return 0.0
    optimal_stretch = 0.3
    return float(np.exp(-((gap - optimal_stretch) ** 2) / (2 * 0.15**2)))


def spacing_effect_weight(days_since_last: float, repetition_count: int) -> float:
    stability = 1.0 + 0.5 * repetition_count
    retention = math.exp(-days_since_last / stability)
    review_urgency = 1.0 - retention
    return max(0.0, min(1.0, review_urgency))


def interleaving_benefit(skills: Sequence[str], current_skill: str) -> float:
    if len(skills) <= 1:
        return 0.0
    unique = set(skills)
    if current_skill not in unique:
        return 0.5
    recent_window = skills[-5:] if len(skills) >= 5 else list(skills)
    same_count = sum(1 for s in recent_window if s == current_skill)
    variety = len(set(recent_window))
    return float(variety / len(recent_window)) * (1.0 - same_count / len(recent_window))


def mastery_probability(
    practice_time: float,
    difficulty: float,
    aptitude: float = 1.0,
) -> float:
    rate = aptitude / max(difficulty, 0.01)
    p = 1.0 - math.exp(-rate * practice_time)
    return max(0.0, min(1.0, p))


def skill_decay(current_level: float, days_without_practice: float, floor: float = 0.1) -> float:
    decay_rate = 0.05
    decayed = current_level * math.exp(-decay_rate * days_without_practice)
    return max(floor * current_level, decayed)


def difficulty_calibration(success_rate: float, target_success_rate: float = 0.75) -> float:
    error = success_rate - target_success_rate
    adjustment = -0.5 * error
    return max(0.5, min(2.0, 1.0 + adjustment))


def _topological_sort(objectives: List[LearningObjective]) -> List[LearningObjective]:
    by_name = {o.skill_name: o for o in objectives}
    visited: set = set()
    order: list = []

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        obj = by_name.get(name)
        if obj:
            for pre in obj.prerequisite_skills:
                visit(pre)
            order.append(obj)

    for o in objectives:
        visit(o.skill_name)
    return order


def sequence_curriculum(
    objectives: List[LearningObjective],
    student: StudentState,
) -> List[CurriculumItem]:
    sorted_objs = _topological_sort(objectives)
    items: List[CurriculumItem] = []
    for obj in sorted_objs:
        current = student.skills.get(obj.skill_name, 0.0)
        zpd = zone_of_proximal_development(current, obj.target_level)
        days = student.last_practiced.get(obj.skill_name, 30.0)
        spacing = spacing_effect_weight(days, student.session_count)
        priority = 0.5 * zpd + 0.3 * spacing + 0.2 * obj.difficulty
        duration = obj.difficulty * 30.0 / max(zpd + 0.1, 0.1)
        duration = min(duration, 60.0)
        items.append(CurriculumItem(
            objective=obj,
            estimated_duration=duration,
            priority=priority,
        ))
    items.sort(key=lambda c: c.priority, reverse=True)
    return items


def practice_schedule(
    items: List[CurriculumItem],
    daily_minutes: float = 60.0,
    n_days: int = 7,
) -> List[List[CurriculumItem]]:
    schedule: List[List[CurriculumItem]] = [[] for _ in range(n_days)]
    day_time = np.zeros(n_days)
    for item in items:
        best_day = int(np.argmin(day_time))
        if day_time[best_day] + item.estimated_duration <= daily_minutes * 1.2:
            schedule[best_day].append(item)
            day_time[best_day] += item.estimated_duration
    for day_items in schedule:
        if len(day_items) > 2:
            np.random.default_rng(42).shuffle(day_items)
    return schedule


def skill_progress_forecast(
    student: StudentState,
    curriculum: List[CurriculumItem],
    sessions: int = 30,
    session_minutes: float = 60.0,
) -> Dict[str, np.ndarray]:
    trajectories: Dict[str, list] = {}
    current = dict(student.skills)
    for item in curriculum:
        name = item.objective.skill_name
        if name not in trajectories:
            trajectories[name] = []
    for s in range(sessions):
        for item in curriculum:
            name = item.objective.skill_name
            c = current.get(name, 0.0)
            time_fraction = min(item.estimated_duration / session_minutes, 1.0)
            gain = mastery_probability(time_fraction * 0.5, item.objective.difficulty)
            current[name] = min(c + gain * (item.objective.target_level - c), 1.0)
        for name in trajectories:
            trajectories[name].append(current.get(name, 0.0))
    return {k: np.array(v) for k, v in trajectories.items()}
