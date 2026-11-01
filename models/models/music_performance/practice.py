"""Practice tools: session tracking, progress, difficulty estimation, spaced repetition scheduling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class PracticeSession:
    start_time: float = 0.0
    end_time: float = 0.0
    tempo_range: tuple[float, float] = (0.0, 0.0)
    error_count: int = 0
    sections_practiced: list[str] = field(default_factory=list)
    notes: str = ""

    def duration_minutes(self) -> float:
        return max(0.0, (self.end_time - self.start_time) / 60.0)

    def error_rate(self) -> float:
        dur = self.duration_minutes()
        if dur < 1e-6:
            return 0.0
        return self.error_count / dur


class ProgressTracker:
    def __init__(self) -> None:
        self._sessions: list[PracticeSession] = []

    def add_session(self, session: PracticeSession) -> None:
        self._sessions.append(session)

    @property
    def n_sessions(self) -> int:
        return len(self._sessions)

    def total_practice_minutes(self) -> float:
        return sum(s.duration_minutes() for s in self._sessions)

    def tempo_history(self) -> list[tuple[float, float]]:
        return [(s.start_time, s.tempo_range[1]) for s in self._sessions if s.tempo_range[1] > 0]

    def error_rate_history(self) -> list[tuple[float, float]]:
        return [(s.start_time, s.error_rate()) for s in self._sessions]

    def tempo_improvement(self) -> float:
        history = self.tempo_history()
        if len(history) < 2:
            return 0.0
        first_tempo = history[0][1]
        last_tempo = history[-1][1]
        if first_tempo < 1e-6:
            return 0.0
        return (last_tempo - first_tempo) / first_tempo

    def error_reduction(self) -> float:
        history = self.error_rate_history()
        if len(history) < 2:
            return 0.0
        first_rate = history[0][1]
        last_rate = history[-1][1]
        if first_rate < 1e-6:
            return 0.0
        return (first_rate - last_rate) / first_rate

    def section_coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self._sessions:
            for sec in s.sections_practiced:
                counts[sec] = counts.get(sec, 0) + 1
        return counts


@dataclass
class DifficultyEstimator:
    @staticmethod
    def estimate(
        note_density: float,
        avg_interval_semitones: float,
        rhythmic_complexity: float,
        tempo_bpm: float,
        max_difficulty: float = 10.0,
    ) -> float:
        density_score = min(note_density / 12.0, 1.0)
        interval_score = min(avg_interval_semitones / 12.0, 1.0)
        rhythm_score = min(rhythmic_complexity, 1.0)
        tempo_score = min(tempo_bpm / 200.0, 1.0)

        raw = (
            0.30 * density_score
            + 0.25 * interval_score
            + 0.25 * rhythm_score
            + 0.20 * tempo_score
        )
        return float(min(raw * max_difficulty, max_difficulty))

    @staticmethod
    def classify(difficulty: float) -> str:
        if difficulty < 2.0:
            return "beginner"
        elif difficulty < 4.0:
            return "intermediate"
        elif difficulty < 6.5:
            return "advanced"
        elif difficulty < 8.5:
            return "expert"
        else:
            return "virtuoso"


class PracticeScheduler:
    def __init__(
        self,
        sections: list[str],
        difficulties: list[float] | None = None,
    ) -> None:
        self._sections = list(sections)
        self._difficulties = (
            list(difficulties)
            if difficulties is not None
            else [5.0] * len(sections)
        )
        self._mastery: dict[str, float] = {s: 0.0 for s in sections}
        self._last_practiced: dict[str, int] = {s: 0 for s in sections}
        self._session_count = 0

    def record_practice(self, section: str, quality: float) -> None:
        if section not in self._mastery:
            return
        self._mastery[section] = min(1.0, self._mastery[section] + quality * 0.2)
        self._session_count += 1
        self._last_practiced[section] = self._session_count

    def priority_score(self, section: str) -> float:
        if section not in self._mastery:
            return 0.0
        mastery = self._mastery[section]
        idx = self._sections.index(section)
        difficulty = self._difficulties[idx]
        staleness = self._session_count - self._last_practiced.get(section, 0)

        difficulty_weight = difficulty / 10.0
        mastery_need = 1.0 - mastery
        staleness_bonus = min(staleness / 10.0, 1.0)

        if mastery >= 0.9:
            return float(0.2 * staleness_bonus + 0.1 * difficulty_weight)
        return float(
            0.40 * mastery_need + 0.35 * difficulty_weight + 0.25 * staleness_bonus
        )

    def generate_plan(self, n_sections: int = 5) -> list[dict]:
        scores = [(s, self.priority_score(s)) for s in self._sections]
        scores.sort(key=lambda x: x[1], reverse=True)
        plan: list[dict] = []
        for section, score in scores[:n_sections]:
            idx = self._sections.index(section)
            plan.append(
                {
                    "section": section,
                    "priority": float(score),
                    "difficulty": self._difficulties[idx],
                    "mastery": self._mastery[section],
                    "sessions_since_last": self._session_count
                    - self._last_practiced.get(section, 0),
                }
            )
        return plan
