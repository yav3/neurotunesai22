"""Adaptive difficulty: Bayesian skill estimation, item response theory, zone of proximal development.

Implements adaptive algorithms for music education that adjust exercise
difficulty based on learner performance, using IRT-inspired models and
Elo-like rating updates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SkillDomain(Enum):
    PITCH = "pitch"
    RHYTHM = "rhythm"
    HARMONY = "harmony"
    TIMBRE = "timbre"
    SIGHT_READING = "sight_reading"
    EAR_TRAINING = "ear_training"
    IMPROVISATION = "improvisation"
    TECHNIQUE = "technique"


@dataclass
class SkillEstimate:
    domain: SkillDomain
    ability: float = 0.0
    uncertainty: float = 1.0
    n_attempts: int = 0
    n_correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_attempts if self.n_attempts > 0 else 0.0

    @property
    def confidence(self) -> float:
        return max(0.0, 1.0 - self.uncertainty)


def irt_probability(ability: float, difficulty: float, discrimination: float = 1.0) -> float:
    exponent = discrimination * (ability - difficulty)
    exponent = max(-20.0, min(20.0, exponent))
    return 1.0 / (1.0 + math.exp(-exponent))


def irt_information(ability: float, difficulty: float, discrimination: float = 1.0) -> float:
    p = irt_probability(ability, difficulty, discrimination)
    q = 1.0 - p
    return discrimination ** 2 * p * q


@dataclass
class ExerciseItem:
    item_id: str
    domain: SkillDomain
    difficulty: float
    discrimination: float = 1.0
    guessing: float = 0.0


def update_ability_elo(
    current: SkillEstimate,
    item: ExerciseItem,
    correct: bool,
    k_base: float = 0.4,
    k_min: float = 0.05,
) -> SkillEstimate:
    k = max(k_min, k_base / (1 + 0.1 * current.n_attempts))
    expected = irt_probability(current.ability, item.difficulty, item.discrimination)
    outcome = 1.0 if correct else 0.0
    new_ability = current.ability + k * (outcome - expected)
    info = irt_information(current.ability, item.difficulty, item.discrimination)
    new_uncertainty = max(0.1, current.uncertainty - 0.02 * info)
    return SkillEstimate(
        domain=current.domain,
        ability=new_ability,
        uncertainty=new_uncertainty,
        n_attempts=current.n_attempts + 1,
        n_correct=current.n_correct + (1 if correct else 0),
    )


def select_next_item(
    skill: SkillEstimate,
    items: list[ExerciseItem],
    target_probability: float = 0.7,
) -> ExerciseItem | None:
    if not items:
        return None
    best_item = None
    best_score = float("inf")
    for item in items:
        p = irt_probability(skill.ability, item.difficulty, item.discrimination)
        score = abs(p - target_probability)
        if score < best_score:
            best_score = score
            best_item = item
    return best_item


def select_max_information(
    skill: SkillEstimate,
    items: list[ExerciseItem],
) -> ExerciseItem | None:
    if not items:
        return None
    best_item = None
    best_info = -1.0
    for item in items:
        info = irt_information(skill.ability, item.difficulty, item.discrimination)
        if info > best_info:
            best_info = info
            best_item = item
    return best_item


@dataclass
class ZPDRange:
    lower: float
    upper: float
    optimal: float

    def contains(self, difficulty: float) -> bool:
        return self.lower <= difficulty <= self.upper


def compute_zpd(skill: SkillEstimate, comfort_margin: float = 0.5) -> ZPDRange:
    lower = skill.ability - comfort_margin * skill.uncertainty
    upper = skill.ability + 1.5 * skill.uncertainty
    optimal = skill.ability + 0.5 * skill.uncertainty
    return ZPDRange(lower=lower, upper=upper, optimal=optimal)


def filter_zpd_items(
    skill: SkillEstimate,
    items: list[ExerciseItem],
    comfort_margin: float = 0.5,
) -> list[ExerciseItem]:
    zpd = compute_zpd(skill, comfort_margin)
    return [item for item in items if zpd.contains(item.difficulty)]


@dataclass
class LearnerProfile:
    skills: dict[SkillDomain, SkillEstimate] = field(default_factory=dict)
    history: list[tuple[str, bool]] = field(default_factory=list)

    def get_skill(self, domain: SkillDomain) -> SkillEstimate:
        if domain not in self.skills:
            self.skills[domain] = SkillEstimate(domain=domain)
        return self.skills[domain]

    def record_attempt(self, item: ExerciseItem, correct: bool) -> None:
        current = self.get_skill(item.domain)
        updated = update_ability_elo(current, item, correct)
        self.skills[item.domain] = updated
        self.history.append((item.item_id, correct))

    def overall_ability(self) -> float:
        if not self.skills:
            return 0.0
        return sum(s.ability for s in self.skills.values()) / len(self.skills)


@dataclass
class DifficultySchedule:
    initial_difficulty: float = 0.0
    ramp_rate: float = 0.1
    plateau_threshold: int = 5
    mastery_threshold: float = 0.85

    def next_difficulty(self, skill: SkillEstimate) -> float:
        if skill.n_attempts < self.plateau_threshold:
            return self.initial_difficulty + self.ramp_rate * skill.n_attempts
        if skill.accuracy >= self.mastery_threshold:
            return skill.ability + 0.5
        return skill.ability


def generate_difficulty_curve(
    n_items: int,
    start_difficulty: float = -1.0,
    end_difficulty: float = 2.0,
    shape: str = "sigmoid",
) -> np.ndarray:
    t = np.linspace(0, 1, n_items)
    if shape == "linear":
        curve = t
    elif shape == "sigmoid":
        curve = 1.0 / (1.0 + np.exp(-6 * (t - 0.5)))
    elif shape == "logarithmic":
        curve = np.log1p(t * (np.e - 1))
    else:
        curve = t
    return start_difficulty + (end_difficulty - start_difficulty) * curve


@dataclass
class AdaptiveSession:
    profile: LearnerProfile
    domain: SkillDomain
    items: list[ExerciseItem]
    target_probability: float = 0.7
    max_items: int = 20
    items_presented: int = 0

    def next_item(self) -> ExerciseItem | None:
        if self.items_presented >= self.max_items:
            return None
        skill = self.profile.get_skill(self.domain)
        zpd_items = filter_zpd_items(skill, self.items)
        if zpd_items:
            selected = select_next_item(skill, zpd_items, self.target_probability)
        else:
            selected = select_next_item(skill, self.items, self.target_probability)
        self.items_presented += 1
        return selected

    def submit_answer(self, item: ExerciseItem, correct: bool) -> None:
        self.profile.record_attempt(item, correct)

    @property
    def session_accuracy(self) -> float:
        recent = self.profile.history[-self.items_presented :] if self.items_presented > 0 else []
        if not recent:
            return 0.0
        return sum(1 for _, c in recent if c) / len(recent)
