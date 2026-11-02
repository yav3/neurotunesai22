"""Curriculum sequencing: prerequisite graphs, learning paths, mastery gating."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class TopicStatus(Enum):
    LOCKED = "locked"
    AVAILABLE = "available"
    IN_PROGRESS = "in_progress"
    MASTERED = "mastered"


@dataclass
class Topic:
    topic_id: str
    name: str
    prerequisites: list[str] = field(default_factory=list)
    difficulty: float = 0.5
    estimated_hours: float = 1.0
    mastery_threshold: float = 0.8


@dataclass
class LearnerState:
    topic_scores: dict[str, float] = field(default_factory=dict)
    time_spent: dict[str, float] = field(default_factory=dict)

    def score(self, topic_id: str) -> float:
        return self.topic_scores.get(topic_id, 0.0)

    def record(self, topic_id: str, score: float, hours: float) -> None:
        prev = self.topic_scores.get(topic_id, 0.0)
        self.topic_scores[topic_id] = max(prev, score)
        self.time_spent[topic_id] = self.time_spent.get(topic_id, 0.0) + hours


@dataclass(frozen=True)
class LearningPath:
    sequence: list[str]
    estimated_hours: float
    topics_mastered: int
    topics_remaining: int


class CurriculumGraph:
    def __init__(self, topics: Sequence[Topic]) -> None:
        self._topics: dict[str, Topic] = {t.topic_id: t for t in topics}
        self._children: dict[str, list[str]] = {t.topic_id: [] for t in topics}
        for t in topics:
            for pre in t.prerequisites:
                if pre in self._children:
                    self._children[pre].append(t.topic_id)

    @property
    def topics(self) -> dict[str, Topic]:
        return dict(self._topics)

    def topic_status(self, topic_id: str, learner: LearnerState) -> TopicStatus:
        topic = self._topics.get(topic_id)
        if topic is None:
            return TopicStatus.LOCKED
        if learner.score(topic_id) >= topic.mastery_threshold:
            return TopicStatus.MASTERED
        for pre in topic.prerequisites:
            pre_topic = self._topics.get(pre)
            if pre_topic is None:
                continue
            if learner.score(pre) < pre_topic.mastery_threshold:
                return TopicStatus.LOCKED
        if learner.score(topic_id) > 0:
            return TopicStatus.IN_PROGRESS
        return TopicStatus.AVAILABLE

    def available_topics(self, learner: LearnerState) -> list[str]:
        return [
            tid for tid in self._topics
            if self.topic_status(tid, learner) == TopicStatus.AVAILABLE
        ]

    def mastered_topics(self, learner: LearnerState) -> list[str]:
        return [
            tid for tid in self._topics
            if self.topic_status(tid, learner) == TopicStatus.MASTERED
        ]

    def suggest_next(self, learner: LearnerState, n: int = 3) -> list[str]:
        available = self.available_topics(learner)
        in_progress = [
            tid for tid in self._topics
            if self.topic_status(tid, learner) == TopicStatus.IN_PROGRESS
        ]
        candidates = in_progress + available
        candidates.sort(key=lambda tid: self._topics[tid].difficulty)
        return candidates[:n]

    def topological_order(self) -> list[str]:
        visited: set[str] = set()
        order: list[str] = []

        def _visit(tid: str) -> None:
            if tid in visited:
                return
            visited.add(tid)
            for pre in self._topics[tid].prerequisites:
                if pre in self._topics:
                    _visit(pre)
            order.append(tid)

        for tid in self._topics:
            _visit(tid)
        return order

    def learning_path(self, learner: LearnerState) -> LearningPath:
        order = self.topological_order()
        remaining = [
            tid for tid in order
            if self.topic_status(tid, learner) != TopicStatus.MASTERED
        ]
        mastered = len(order) - len(remaining)
        est_hours = sum(self._topics[tid].estimated_hours for tid in remaining)
        return LearningPath(
            sequence=remaining,
            estimated_hours=est_hours,
            topics_mastered=mastered,
            topics_remaining=len(remaining),
        )


def spaced_review_schedule(
    mastery_scores: dict[str, float],
    base_interval_days: float = 1.0,
    ease_factor: float = 2.5,
) -> dict[str, float]:
    schedule: dict[str, float] = {}
    for topic_id, score in mastery_scores.items():
        if score < 0.3:
            interval = base_interval_days * 0.5
        elif score < 0.6:
            interval = base_interval_days
        elif score < 0.9:
            interval = base_interval_days * ease_factor
        else:
            interval = base_interval_days * ease_factor * ease_factor
        schedule[topic_id] = interval
    return schedule


def forgetting_curve(
    initial_strength: float,
    time_days: float,
    stability: float = 1.0,
) -> float:
    return initial_strength * np.exp(-time_days / max(stability, 0.01))


def optimal_review_time(
    initial_strength: float,
    target_retention: float = 0.8,
    stability: float = 1.0,
) -> float:
    if initial_strength <= 0 or target_retention >= initial_strength:
        return 0.0
    ratio = target_retention / initial_strength
    if ratio <= 0:
        return 0.0
    return -stability * float(np.log(ratio))
