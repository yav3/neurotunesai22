"""Practice scheduling: spaced repetition, difficulty progression, fatigue modeling.

Optimal practice session design using spacing effect, interleaving,
and cognitive load theory. Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class PracticeType(str, Enum):
    TECHNIQUE = "technique"
    REPERTOIRE = "repertoire"
    SIGHT_READING = "sight_reading"
    EAR_TRAINING = "ear_training"
    IMPROVISATION = "improvisation"
    THEORY = "theory"


@dataclass
class PracticeItem:
    name: str
    practice_type: PracticeType
    difficulty: float
    last_practiced_days_ago: float = 0.0
    repetition_count: int = 0
    success_rate: float = 0.5


@dataclass(frozen=True)
class SessionPlan:
    items: tuple[PracticeItem, ...]
    total_minutes: float
    estimated_retention: float
    cognitive_load: float


@dataclass(frozen=True)
class RetentionEstimate:
    retention: float
    optimal_review_days: float
    stability: float


INITIAL_STABILITY = 1.0
STABILITY_GROWTH = 1.5
MIN_RETENTION_THRESHOLD = 0.85


def forgetting_curve(days: float, stability: float) -> float:
    return float(np.exp(-days / max(stability, 0.01)))


def item_stability(item: PracticeItem) -> float:
    return INITIAL_STABILITY * (STABILITY_GROWTH ** item.repetition_count) * (
        0.5 + 0.5 * item.success_rate
    )


def current_retention(item: PracticeItem) -> float:
    s = item_stability(item)
    return forgetting_curve(item.last_practiced_days_ago, s)


def optimal_review_interval(item: PracticeItem, target_retention: float = 0.85) -> float:
    s = item_stability(item)
    if target_retention <= 0 or target_retention >= 1:
        return s
    return float(-s * np.log(target_retention))


def estimate_retention(item: PracticeItem) -> RetentionEstimate:
    s = item_stability(item)
    return RetentionEstimate(
        retention=current_retention(item),
        optimal_review_days=optimal_review_interval(item),
        stability=s,
    )


def priority_score(item: PracticeItem) -> float:
    ret = current_retention(item)
    urgency = max(0.0, 1.0 - ret)
    difficulty_weight = 0.5 + 0.5 * item.difficulty
    return float(urgency * difficulty_weight)


def cognitive_load(items: list[PracticeItem]) -> float:
    if not items:
        return 0.0
    difficulties = np.array([it.difficulty for it in items])
    novelties = np.array([1.0 - current_retention(it) for it in items])
    return float(np.mean(difficulties * 0.6 + novelties * 0.4))


def interleave_schedule(items: list[PracticeItem]) -> list[PracticeItem]:
    if len(items) <= 2:
        return list(items)
    types: dict[PracticeType, list[PracticeItem]] = {}
    for it in items:
        types.setdefault(it.practice_type, []).append(it)
    schedule: list[PracticeItem] = []
    type_lists = list(types.values())
    idx = 0
    while any(type_lists):
        for tl in type_lists:
            if tl:
                schedule.append(tl.pop(0))
        type_lists = [tl for tl in type_lists if tl]
        idx += 1
    return schedule


def fatigue_curve(minutes: float, intensity: float = 0.5) -> float:
    tau = 45.0 / max(intensity, 0.1)
    return float(np.exp(-minutes / tau))


def effective_practice_time(total_minutes: float, intensity: float = 0.5) -> float:
    t = np.linspace(0, total_minutes, max(int(total_minutes), 2))
    efficiency = np.array([fatigue_curve(m, intensity) for m in t])
    return float(np.trapz(efficiency, t))


def plan_session(
    items: list[PracticeItem],
    available_minutes: float = 60.0,
    target_retention: float = 0.85,
) -> SessionPlan:
    scored = sorted(items, key=priority_score, reverse=True)
    minutes_per_item = available_minutes / max(len(scored), 1)
    selected: list[PracticeItem] = []
    remaining = available_minutes
    for it in scored:
        if remaining <= 0:
            break
        selected.append(it)
        remaining -= minutes_per_item
    scheduled = interleave_schedule(selected)
    retentions = [current_retention(it) for it in scheduled]
    avg_ret = float(np.mean(retentions)) if retentions else 0.0
    cl = cognitive_load(scheduled)
    return SessionPlan(
        items=tuple(scheduled),
        total_minutes=available_minutes,
        estimated_retention=avg_ret,
        cognitive_load=cl,
    )


def weekly_plan(
    items: list[PracticeItem],
    daily_minutes: float = 60.0,
    days: int = 7,
) -> list[SessionPlan]:
    plans: list[SessionPlan] = []
    for _ in range(days):
        plan = plan_session(items, daily_minutes)
        plans.append(plan)
        for it in items:
            it.last_practiced_days_ago += 1.0
    return plans


def progression_rate(success_history: list[float], window: int = 10) -> float:
    if len(success_history) < 2:
        return 0.0
    recent = success_history[-window:]
    x = np.arange(len(recent), dtype=float)
    y = np.array(recent)
    if np.std(x) < 1e-10:
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    return slope
