"""Music training curriculum design and sequencing.

Skill prerequisite graphs, difficulty estimation, spacing/interleaving
schedules, and mastery-based progression.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class SkillLevel(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class Skill:
    name: str
    level: SkillLevel
    difficulty: float  # 0-1
    prerequisites: List[str] = field(default_factory=list)
    estimated_hours: float = 1.0


@dataclass
class LessonPlan:
    skills_ordered: List[str]
    total_hours: float
    difficulty_curve: np.ndarray


def topological_sort(skills: Dict[str, Skill]) -> List[str]:
    """Topological sort of skills by prerequisite ordering."""
    visited: Set[str] = set()
    result: List[str] = []

    def dfs(name: str):
        if name in visited:
            return
        visited.add(name)
        if name in skills:
            for prereq in skills[name].prerequisites:
                dfs(prereq)
        result.append(name)

    for name in skills:
        dfs(name)
    return result


def difficulty_curve(
    skills: List[Skill],
    smoothing: int = 3,
) -> np.ndarray:
    """Compute the difficulty progression curve for an ordered skill list."""
    raw = np.array([s.difficulty for s in skills])
    if len(raw) < smoothing:
        return raw
    kernel = np.ones(smoothing) / smoothing
    return np.convolve(raw, kernel, mode="same")


def spacing_schedule(
    n_items: int,
    total_days: int,
    spacing_factor: float = 2.0,
) -> np.ndarray:
    """Generate an expanding-interval spacing schedule (spaced repetition).

    Returns review day indices for each item.
    """
    intervals = np.zeros(n_items)
    current_interval = 1.0
    for i in range(n_items):
        intervals[i] = current_interval
        current_interval = min(current_interval * spacing_factor, total_days)
    return np.cumsum(intervals)


def interleaving_schedule(
    topics: List[str],
    n_sessions: int,
    block_size: int = 3,
) -> List[str]:
    """Generate an interleaved practice schedule.

    Alternates between topics in blocks rather than massed practice.
    """
    schedule: List[str] = []
    n = len(topics)
    if n == 0:
        return schedule
    for session in range(n_sessions):
        topic_idx = (session // block_size) % n
        schedule.append(topics[topic_idx])
    return schedule


def mastery_threshold(
    accuracy_history: np.ndarray,
    window: int = 5,
    threshold: float = 0.85,
) -> bool:
    """Check if a learner has reached mastery (sustained accuracy above threshold)."""
    if len(accuracy_history) < window:
        return False
    recent = accuracy_history[-window:]
    return bool(np.all(recent >= threshold))


def estimated_completion_time(
    skills: Dict[str, Skill],
    mastery_rate: float = 0.8,
) -> float:
    """Estimate total hours to complete curriculum with given mastery rate."""
    ordered = topological_sort(skills)
    total = 0.0
    for name in ordered:
        if name in skills:
            s = skills[name]
            total += s.estimated_hours / max(mastery_rate, 0.1)
    return total


def build_curriculum(
    skills: Dict[str, Skill],
) -> LessonPlan:
    """Build an ordered curriculum from a skill graph."""
    ordered = topological_sort(skills)
    ordered_skills = [skills[n] for n in ordered if n in skills]
    curve = difficulty_curve(ordered_skills)
    total_h = sum(s.estimated_hours for s in ordered_skills)
    return LessonPlan(
        skills_ordered=ordered,
        total_hours=total_h,
        difficulty_curve=curve,
    )
