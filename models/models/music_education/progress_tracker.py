"""Learning progress tracking: skill assessment, mastery curves, recommendations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class SkillAttempt:
    skill: str
    correct: bool
    timestamp: float = 0.0


@dataclass
class SkillAssessment:
    skill_name: str
    score: float = 0.0
    attempts: int = 0
    correct_count: int = 0
    last_attempt_time: float = 0.0


class LearningProfile:
    def __init__(self, decay: float = 0.95) -> None:
        self._history: list[SkillAttempt] = []
        self._skills: dict[str, SkillAssessment] = {}
        self._decay = decay

    def record_attempt(self, skill: str, correct: bool, timestamp: float = 0.0) -> None:
        self._history.append(SkillAttempt(skill, correct, timestamp))
        if skill not in self._skills:
            self._skills[skill] = SkillAssessment(skill_name=skill)
        sa = self._skills[skill]
        sa.attempts += 1
        if correct:
            sa.correct_count += 1
        sa.last_attempt_time = timestamp
        sa.score = self._compute_weighted_score(skill)

    def _compute_weighted_score(self, skill: str) -> float:
        attempts = [a for a in self._history if a.skill == skill]
        if not attempts:
            return 0.0
        n = len(attempts)
        weights = np.array([self._decay ** (n - 1 - i) for i in range(n)])
        scores = np.array([1.0 if a.correct else 0.0 for a in attempts])
        return float(np.sum(weights * scores) / np.sum(weights))

    def skill_level(self, skill: str) -> float:
        if skill not in self._skills:
            return 0.0
        return self._skills[skill].score

    def all_skills(self) -> dict[str, SkillAssessment]:
        return dict(self._skills)

    def weak_areas(self, threshold: float = 0.5) -> list[str]:
        return [s for s, a in self._skills.items() if a.score < threshold and a.attempts >= 3]

    def strong_areas(self, threshold: float = 0.8) -> list[str]:
        return [s for s, a in self._skills.items() if a.score >= threshold and a.attempts >= 3]

    def recommend_next(self, available_skills: Sequence[str]) -> str:
        best_skill = available_skills[0] if available_skills else ""
        lowest_score = float("inf")
        for skill in available_skills:
            score = self.skill_level(skill)
            if score < lowest_score:
                lowest_score = score
                best_skill = skill
        return best_skill

    def mastery_curve(self, skill: str) -> list[float]:
        attempts = [a for a in self._history if a.skill == skill]
        if not attempts:
            return []
        running = []
        correct_so_far = 0
        for i, a in enumerate(attempts):
            if a.correct:
                correct_so_far += 1
            running.append(correct_so_far / (i + 1))
        return running

    def streak(self, skill: str) -> int:
        attempts = [a for a in reversed(self._history) if a.skill == skill]
        count = 0
        for a in attempts:
            if a.correct:
                count += 1
            else:
                break
        return count

    def total_practice_time(self) -> float:
        if len(self._history) < 2:
            return 0.0
        times = sorted(a.timestamp for a in self._history if a.timestamp > 0)
        if len(times) < 2:
            return 0.0
        return times[-1] - times[0]
