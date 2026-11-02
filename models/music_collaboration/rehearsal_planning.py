"""Rehearsal planning: scheduling, section focus, difficulty estimation, progress tracking.

Provides algorithms for optimal rehearsal scheduling based on piece difficulty,
section weakness detection, and progress modeling over rehearsal sessions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RehearsalFocus(Enum):
    FULL_RUN = "full_run"
    SECTION_DRILL = "section_drill"
    SIGHT_READ = "sight_read"
    TEMPO_RAMP = "tempo_ramp"
    DYNAMICS_POLISH = "dynamics_polish"
    ENSEMBLE_BLEND = "ensemble_blend"
    MEMORY_WORK = "memory_work"


@dataclass(frozen=True)
class SectionDifficulty:
    section_id: str
    technical_difficulty: float
    ensemble_difficulty: float
    rhythmic_complexity: float
    dynamic_range: float
    tempo_factor: float

    @property
    def composite(self) -> float:
        return (0.35 * self.technical_difficulty
                + 0.25 * self.ensemble_difficulty
                + 0.20 * self.rhythmic_complexity
                + 0.10 * self.dynamic_range
                + 0.10 * self.tempo_factor)


@dataclass
class RehearsalPlan:
    session_minutes: int
    sections: list[tuple[str, RehearsalFocus, int]]
    warmup_minutes: int = 5
    break_minutes: int = 5
    total_difficulty: float = 0.0


@dataclass
class ProgressRecord:
    section_id: str
    session_index: int
    accuracy: float
    tempo_percent: float
    notes: str = ""


def estimate_section_difficulty(
    note_density: float,
    tempo_bpm: float,
    key_changes: int,
    time_sig_changes: int,
    dynamic_markings: int,
    max_interval_semitones: int,
) -> SectionDifficulty:
    tech = min(1.0, note_density * tempo_bpm / 20000.0 + max_interval_semitones / 24.0)
    ens = min(1.0, (key_changes + time_sig_changes) / 8.0)
    rhy = min(1.0, time_sig_changes / 4.0 + 0.3 * (tempo_bpm > 160))
    dyn = min(1.0, dynamic_markings / 12.0)
    tempo_f = min(1.0, tempo_bpm / 220.0)
    return SectionDifficulty(
        section_id="auto",
        technical_difficulty=tech,
        ensemble_difficulty=ens,
        rhythmic_complexity=rhy,
        dynamic_range=dyn,
        tempo_factor=tempo_f,
    )


def plan_rehearsal(
    sections: list[SectionDifficulty],
    total_minutes: int = 90,
    warmup: int = 5,
    break_every: int = 25,
) -> RehearsalPlan:
    if not sections:
        return RehearsalPlan(session_minutes=total_minutes, sections=[])
    sorted_secs = sorted(sections, key=lambda s: s.composite, reverse=True)
    available = total_minutes - warmup
    n_breaks = max(0, available // break_every - 1)
    available -= n_breaks * 5
    plan_sections: list[tuple[str, RehearsalFocus, int]] = []
    total_diff = sum(s.composite for s in sorted_secs)
    for sec in sorted_secs:
        frac = sec.composite / max(total_diff, 1e-6)
        minutes = max(5, int(available * frac))
        if sec.technical_difficulty > 0.7:
            focus = RehearsalFocus.SECTION_DRILL
        elif sec.ensemble_difficulty > 0.6:
            focus = RehearsalFocus.ENSEMBLE_BLEND
        elif sec.tempo_factor > 0.7:
            focus = RehearsalFocus.TEMPO_RAMP
        else:
            focus = RehearsalFocus.FULL_RUN
        plan_sections.append((sec.section_id, focus, minutes))
    return RehearsalPlan(
        session_minutes=total_minutes,
        sections=plan_sections,
        warmup_minutes=warmup,
        break_minutes=5 * n_breaks,
        total_difficulty=total_diff,
    )


def learning_curve(
    sessions: list[ProgressRecord],
) -> dict[str, list[float]]:
    by_section: dict[str, list[float]] = {}
    for rec in sessions:
        by_section.setdefault(rec.section_id, []).append(rec.accuracy)
    return by_section


def weakest_sections(
    progress: dict[str, list[float]],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    latest: list[tuple[str, float]] = []
    for sid, accs in progress.items():
        latest.append((sid, accs[-1] if accs else 0.0))
    latest.sort(key=lambda x: x[1])
    return latest[:top_n]


def tempo_ramp_schedule(
    target_bpm: float,
    current_bpm: float,
    sessions_remaining: int,
) -> list[float]:
    if sessions_remaining <= 0 or current_bpm >= target_bpm:
        return [target_bpm]
    schedule = []
    for i in range(sessions_remaining):
        t = (i + 1) / sessions_remaining
        bpm = current_bpm + (target_bpm - current_bpm) * (1 - (1 - t) ** 2)
        schedule.append(round(bpm, 1))
    return schedule


def rehearsal_efficiency(
    plan: RehearsalPlan,
    progress_before: dict[str, float],
    progress_after: dict[str, float],
) -> float:
    if not plan.sections:
        return 0.0
    improvements = []
    for sid, _, minutes in plan.sections:
        before = progress_before.get(sid, 0.0)
        after = progress_after.get(sid, before)
        if minutes > 0:
            improvements.append((after - before) / (minutes / 10.0 + 1.0))
    return float(np.mean(improvements)) if improvements else 0.0
