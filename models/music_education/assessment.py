"""Assessment: rubrics, skill evaluation, performance scoring."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RubricCriterion:
    name: str
    weight: float = 1.0
    levels: dict[int, str] = field(default_factory=lambda: {
        1: "Beginning", 2: "Developing", 3: "Proficient", 4: "Advanced",
    })

    def max_score(self) -> float:
        return max(self.levels.keys()) * self.weight if self.levels else 0.0


class PerformanceRubric:
    def __init__(self, criteria: list[RubricCriterion] | None = None):
        self.criteria = criteria or [
            RubricCriterion("pitch_accuracy", weight=1.5),
            RubricCriterion("rhythm_accuracy", weight=1.5),
            RubricCriterion("dynamics", weight=1.0),
            RubricCriterion("articulation", weight=1.0),
            RubricCriterion("expression", weight=1.0),
        ]

    def max_total(self) -> float:
        return sum(c.max_score() for c in self.criteria)

    def score(self, ratings: dict[str, int]) -> float:
        total = 0.0
        for c in self.criteria:
            level = ratings.get(c.name, 1)
            level = min(max(level, min(c.levels.keys())), max(c.levels.keys()))
            total += level * c.weight
        return total

    def normalized_score(self, ratings: dict[str, int]) -> float:
        mx = self.max_total()
        if mx <= 0:
            return 0.0
        return self.score(ratings) / mx

    def feedback(self, ratings: dict[str, int]) -> list[dict]:
        fb = []
        for c in self.criteria:
            level = ratings.get(c.name, 1)
            level = min(max(level, min(c.levels.keys())), max(c.levels.keys()))
            fb.append({
                "criterion": c.name,
                "level": level,
                "description": c.levels.get(level, ""),
                "weighted_score": level * c.weight,
            })
        return fb


class SkillAssessment:
    def __init__(self):
        self._scores: list[float] = []
        self._rubric = PerformanceRubric()

    def add_assessment(self, ratings: dict[str, int]) -> float:
        score = self._rubric.normalized_score(ratings)
        self._scores.append(score)
        return score

    def average(self) -> float:
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    def trend(self) -> float:
        if len(self._scores) < 2:
            return 0.0
        recent = self._scores[-3:]
        early = self._scores[:3]
        return sum(recent) / len(recent) - sum(early) / len(early)

    def mastery_level(self) -> str:
        avg = self.average()
        if avg >= 0.9:
            return "master"
        elif avg >= 0.75:
            return "advanced"
        elif avg >= 0.5:
            return "proficient"
        elif avg >= 0.25:
            return "developing"
        return "beginning"
