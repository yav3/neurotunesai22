"""Guided Imagery and Music (GIM): protocol modeling and program design.

Models the Bonny Method of Guided Imagery and Music — structured music
listening sequences designed to evoke imagery, affect, and insight. Includes
canonical program arcs, readiness matching, and imagery density tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class GIMPhase(Enum):
    PRELUDE = "prelude"
    INDUCTION = "induction"
    MUSIC_LISTENING = "music_listening"
    RETURN = "return"
    POSTLUDE = "postlude"


@dataclass
class ProgramPiece:
    title: str
    composer: str
    tempo_bpm: float
    key_mode: str
    dynamic_level: float
    duration_min: float
    affective_target: float


@dataclass
class GIMProgram:
    name: str
    pieces: list[ProgramPiece]
    intensity_level: int
    description: str

    @property
    def total_duration(self) -> float:
        return sum(p.duration_min for p in self.pieces)

    def affective_arc(self) -> np.ndarray:
        return np.array([p.affective_target for p in self.pieces])


BONNY_PROGRAMS: dict[str, GIMProgram] = {
    "nurturing": GIMProgram(
        "Nurturing", [
            ProgramPiece("Prelude to the Afternoon of a Faun", "Debussy", 60, "major", 0.3, 10.0, 0.4),
            ProgramPiece("Canon in D", "Pachelbel", 66, "major", 0.4, 8.0, 0.6),
            ProgramPiece("Adagio for Strings", "Barber", 50, "minor", 0.5, 9.0, 0.5),
        ], intensity_level=1, description="Gentle, supportive; suitable for beginning travelers"),
    "positive_affect": GIMProgram(
        "Positive Affect", [
            ProgramPiece("Spring (Vivaldi)", "Vivaldi", 92, "major", 0.6, 7.0, 0.7),
            ProgramPiece("Brandenburg Concerto No. 3", "Bach", 108, "major", 0.7, 8.0, 0.8),
            ProgramPiece("Eine kleine Nachtmusik", "Mozart", 120, "major", 0.6, 6.0, 0.7),
        ], intensity_level=2, description="Uplifting, energizing; builds positive emotional resources"),
    "peak_experience": GIMProgram(
        "Peak Experience", [
            ProgramPiece("Prelude in C# minor", "Rachmaninoff", 76, "minor", 0.5, 5.0, 0.5),
            ProgramPiece("Symphony No. 5, mvt 4", "Beethoven", 132, "major", 0.9, 10.0, 0.9),
            ProgramPiece("Nimrod (Enigma Variations)", "Elgar", 52, "major", 0.7, 5.0, 0.7),
        ], intensity_level=4, description="Intense emotional and transpersonal experience"),
    "emotional_expression": GIMProgram(
        "Emotional Expression", [
            ProgramPiece("Pavane pour une infante defunte", "Ravel", 58, "minor", 0.3, 7.0, 0.3),
            ProgramPiece("Adagietto, Symphony No. 5", "Mahler", 44, "major", 0.5, 11.0, 0.5),
            ProgramPiece("Clair de Lune", "Debussy", 56, "major", 0.3, 6.0, 0.4),
        ], intensity_level=3, description="Deep emotional exploration and release"),
    "death_rebirth": GIMProgram(
        "Death-Rebirth", [
            ProgramPiece("Requiem - Lacrimosa", "Mozart", 60, "minor", 0.6, 4.0, 0.2),
            ProgramPiece("Night on Bald Mountain", "Mussorgsky", 120, "minor", 0.9, 8.0, 0.1),
            ProgramPiece("Resurrection Symphony, finale", "Mahler", 100, "major", 0.9, 12.0, 0.9),
        ], intensity_level=5, description="Confrontation with loss, transformation, renewal"),
}


@dataclass
class ImageryScorecard:
    """Track imagery content across a GIM session."""
    imagery_events: list[dict] = field(default_factory=list)

    def record(self, time_min: float, content: str, valence: float,
               category: str = "general") -> None:
        self.imagery_events.append({
            "time": time_min, "content": content,
            "valence": valence, "category": category,
        })

    @property
    def density(self) -> float:
        if not self.imagery_events:
            return 0.0
        span = self.imagery_events[-1]["time"] - self.imagery_events[0]["time"]
        return len(self.imagery_events) / max(span, 0.1)

    def valence_trajectory(self) -> np.ndarray:
        return np.array([e["valence"] for e in self.imagery_events])

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.imagery_events:
            counts[e["category"]] = counts.get(e["category"], 0) + 1
        return counts


def program_arc(program: GIMProgram) -> dict[str, float]:
    arc = program.affective_arc()
    if len(arc) == 0:
        return {"peak": 0.0, "range": 0.0, "climb": 0.0}
    return {
        "peak": float(np.max(arc)),
        "trough": float(np.min(arc)),
        "range": float(np.max(arc) - np.min(arc)),
        "climb": float(arc[-1] - arc[0]),
        "peak_position": float(np.argmax(arc) / max(len(arc) - 1, 1)),
    }


def match_client_readiness(readiness_level: int,
                           programs: dict[str, GIMProgram] | None = None
                           ) -> list[str]:
    """Recommend GIM programs matching client readiness (1-5 scale)."""
    if programs is None:
        programs = BONNY_PROGRAMS
    readiness_level = max(1, min(5, readiness_level))
    ranked = sorted(programs.items(),
                    key=lambda kv: abs(kv[1].intensity_level - readiness_level))
    return [name for name, _ in ranked if abs(_.intensity_level - readiness_level) <= 1]


@dataclass
class SessionPlan:
    """Full GIM session plan with phase timing."""
    program: GIMProgram
    prelude_min: float = 5.0
    induction_min: float = 10.0
    return_min: float = 5.0
    postlude_min: float = 10.0

    @property
    def total_session_min(self) -> float:
        return (self.prelude_min + self.induction_min
                + self.program.total_duration
                + self.return_min + self.postlude_min)

    def phase_schedule(self) -> list[tuple[GIMPhase, float, float]]:
        t = 0.0
        schedule = []
        for phase, dur in [
            (GIMPhase.PRELUDE, self.prelude_min),
            (GIMPhase.INDUCTION, self.induction_min),
            (GIMPhase.MUSIC_LISTENING, self.program.total_duration),
            (GIMPhase.RETURN, self.return_min),
            (GIMPhase.POSTLUDE, self.postlude_min),
        ]:
            schedule.append((phase, t, t + dur))
            t += dur
        return schedule
