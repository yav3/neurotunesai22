"""Practice planning: spaced repetition, skill tracking, session planning."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SkillRecord:
    name: str
    level: float = 0.0
    attempts: int = 0
    successes: int = 0
    last_practiced: float = 0.0
    ease_factor: float = 2.5
    interval_days: float = 1.0


class SpacedRepetition:
    def next_interval(self, record: SkillRecord, quality: int) -> float:
        q = min(max(quality, 0), 5)
        ef = record.ease_factor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        ef = max(ef, 1.3)
        if q < 3:
            return 1.0
        if record.attempts <= 1:
            return 1.0
        elif record.attempts == 2:
            return 6.0
        else:
            return record.interval_days * ef

    def update(self, record: SkillRecord, quality: int, current_day: float = 0.0) -> SkillRecord:
        new_interval = self.next_interval(record, quality)
        q = min(max(quality, 0), 5)
        ef = record.ease_factor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        return SkillRecord(
            name=record.name,
            level=min(record.level + (0.1 if quality >= 3 else -0.05), 1.0),
            attempts=record.attempts + 1,
            successes=record.successes + (1 if quality >= 3 else 0),
            last_practiced=current_day,
            ease_factor=max(ef, 1.3),
            interval_days=new_interval,
        )


class SkillTracker:
    def __init__(self):
        self._skills: dict[str, SkillRecord] = {}
        self._sr = SpacedRepetition()

    def add_skill(self, name: str) -> SkillRecord:
        if name not in self._skills:
            self._skills[name] = SkillRecord(name=name)
        return self._skills[name]

    def record_practice(self, name: str, quality: int, current_day: float = 0.0) -> SkillRecord:
        if name not in self._skills:
            self.add_skill(name)
        self._skills[name] = self._sr.update(self._skills[name], quality, current_day)
        return self._skills[name]

    def due_skills(self, current_day: float) -> list[SkillRecord]:
        due = []
        for rec in self._skills.values():
            if current_day >= rec.last_practiced + rec.interval_days:
                due.append(rec)
        return sorted(due, key=lambda r: r.last_practiced)

    def overall_level(self) -> float:
        if not self._skills:
            return 0.0
        return sum(r.level for r in self._skills.values()) / len(self._skills)

    def get_skill(self, name: str) -> SkillRecord | None:
        return self._skills.get(name)


class PracticePlanner:
    def __init__(self, session_minutes: int = 30):
        self.session_minutes = session_minutes

    def plan(self, tracker: SkillTracker, current_day: float = 0.0) -> list[dict]:
        due = tracker.due_skills(current_day)
        if not due:
            due = sorted(tracker._skills.values(), key=lambda r: r.level)
        total_weight = sum(max(1.0 - r.level, 0.1) for r in due)
        plan = []
        for rec in due:
            weight = max(1.0 - rec.level, 0.1)
            minutes = max(5, int(self.session_minutes * weight / total_weight))
            plan.append({"skill": rec.name, "minutes": minutes, "level": rec.level})
        return plan
