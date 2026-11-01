"""Spaced repetition for music learning: SM-2 scheduling, forgetting curves."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReviewItem:
    item_id: str
    ease_factor: float = 2.5
    interval_days: float = 1.0
    repetitions: int = 0
    last_review_day: float = 0.0
    lapse_count: int = 0


@dataclass
class ReviewResult:
    item_id: str
    new_interval: float
    new_ease: float
    retention_probability: float
    next_review_day: float


def sm2_update(item: ReviewItem, quality: int, current_day: float) -> ReviewResult:
    q = max(0, min(5, quality))
    ef = item.ease_factor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    ef = max(1.3, ef)

    if q < 3:
        interval = 1.0
        reps = 0
        lapse = item.lapse_count + 1
    else:
        reps = item.repetitions + 1
        lapse = item.lapse_count
        if reps == 1:
            interval = 1.0
        elif reps == 2:
            interval = 6.0
        else:
            interval = item.interval_days * ef

    retention = retention_probability(interval, ef)

    item.ease_factor = ef
    item.interval_days = interval
    item.repetitions = reps
    item.last_review_day = current_day
    item.lapse_count = lapse

    return ReviewResult(
        item_id=item.item_id,
        new_interval=interval,
        new_ease=ef,
        retention_probability=retention,
        next_review_day=current_day + interval,
    )


def retention_probability(
    days_since_review: float,
    stability: float = 2.5,
    decay_rate: float = 0.2,
) -> float:
    if days_since_review <= 0:
        return 1.0
    return math.exp(-decay_rate * days_since_review / max(stability, 0.01))


def optimal_interval(
    target_retention: float = 0.9,
    stability: float = 2.5,
    decay_rate: float = 0.2,
) -> float:
    if target_retention >= 1.0:
        return 0.0
    if target_retention <= 0.0:
        return float("inf")
    return -math.log(target_retention) * max(stability, 0.01) / decay_rate


@dataclass
class LeitnerBox:
    n_boxes: int = 5
    boxes: dict[int, list[str]] = field(default_factory=dict)
    item_box: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for i in range(self.n_boxes):
            self.boxes.setdefault(i, [])

    def add_item(self, item_id: str) -> None:
        if item_id not in self.item_box:
            self.item_box[item_id] = 0
            self.boxes.setdefault(0, []).append(item_id)

    def review(self, item_id: str, correct: bool) -> int:
        current = self.item_box.get(item_id, 0)
        if item_id in self.boxes.get(current, []):
            self.boxes[current].remove(item_id)
        if correct:
            new_box = min(current + 1, self.n_boxes - 1)
        else:
            new_box = 0
        self.item_box[item_id] = new_box
        self.boxes.setdefault(new_box, []).append(item_id)
        return new_box

    def due_items(self, current_day: int) -> list[str]:
        due = []
        for box_idx in range(self.n_boxes):
            interval = 2 ** box_idx
            if current_day % interval == 0:
                due.extend(self.boxes.get(box_idx, []))
        return due

    def mastery_ratio(self) -> float:
        total = sum(len(v) for v in self.boxes.values())
        if total == 0:
            return 0.0
        top = len(self.boxes.get(self.n_boxes - 1, []))
        return top / total


def forgetting_curve(
    days: np.ndarray,
    stability: float = 2.5,
    decay_rate: float = 0.2,
) -> np.ndarray:
    return np.exp(-decay_rate * days / max(stability, 0.01))


def schedule_session(
    items: list[ReviewItem],
    current_day: float,
    max_reviews: int = 20,
    target_retention: float = 0.85,
) -> list[str]:
    overdue = []
    for item in items:
        days_elapsed = current_day - item.last_review_day
        r = retention_probability(days_elapsed, item.ease_factor)
        if r < target_retention:
            overdue.append((r, item.item_id))
    overdue.sort()
    return [item_id for _, item_id in overdue[:max_reviews]]


def difficulty_estimate(history: list[int]) -> float:
    if not history:
        return 0.5
    weights = np.array([0.5 ** i for i in range(len(history) - 1, -1, -1)])
    scores = np.array(history, dtype=float) / 5.0
    return float(1.0 - np.dot(weights, scores) / weights.sum())
