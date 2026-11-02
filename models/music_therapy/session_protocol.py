"""Session protocol planning for music therapy sessions."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class TherapyGoal(Enum):
    AROUSAL_REGULATION = "arousal_regulation"
    EMOTIONAL_PROCESSING = "emotional_processing"
    SOCIAL_BONDING = "social_bonding"
    MOTOR_REHABILITATION = "motor_rehabilitation"
    COGNITIVE_STIMULATION = "cognitive_stimulation"


class SessionPhase(Enum):
    OPENING = "opening"
    WARMUP = "warmup"
    WORKING = "working"
    COOLDOWN = "cooldown"
    CLOSING = "closing"


class InterventionType(Enum):
    RECEPTIVE = "receptive"
    ACTIVE_RHYTHM = "active_rhythm"
    SONGWRITING = "songwriting"
    IMPROVISATION = "improvisation"
    MOVEMENT = "movement"
    VOCAL = "vocal"


@dataclass
class Intervention:
    intervention_type: InterventionType
    duration_min: float
    tempo_bpm: float = 80.0
    mode: str = "ionian"
    description: str = ""


@dataclass
class SessionProtocol:
    goal: TherapyGoal
    total_duration_min: float
    phases: list[tuple[SessionPhase, list[Intervention]]] = field(default_factory=list)

    @property
    def n_interventions(self) -> int:
        return sum(len(intv) for _, intv in self.phases)

    @property
    def planned_duration(self) -> float:
        return sum(i.duration_min for _, intlist in self.phases for i in intlist)


def build_protocol(
    goal: TherapyGoal, duration_min: float = 45.0
) -> SessionProtocol:
    protocol = SessionProtocol(goal=goal, total_duration_min=duration_min)
    phase_fracs = {
        SessionPhase.OPENING: 0.1,
        SessionPhase.WARMUP: 0.15,
        SessionPhase.WORKING: 0.5,
        SessionPhase.COOLDOWN: 0.15,
        SessionPhase.CLOSING: 0.1,
    }
    for phase, frac in phase_fracs.items():
        dur = duration_min * frac
        if phase == SessionPhase.WORKING:
            intv_type = {
                TherapyGoal.AROUSAL_REGULATION: InterventionType.ACTIVE_RHYTHM,
                TherapyGoal.EMOTIONAL_PROCESSING: InterventionType.SONGWRITING,
                TherapyGoal.SOCIAL_BONDING: InterventionType.IMPROVISATION,
                TherapyGoal.MOTOR_REHABILITATION: InterventionType.MOVEMENT,
                TherapyGoal.COGNITIVE_STIMULATION: InterventionType.VOCAL,
            }.get(goal, InterventionType.RECEPTIVE)
            interventions = [Intervention(intv_type, dur)]
        elif phase in (SessionPhase.OPENING, SessionPhase.CLOSING):
            interventions = [Intervention(InterventionType.RECEPTIVE, dur, tempo_bpm=65.0)]
        else:
            interventions = [Intervention(InterventionType.ACTIVE_RHYTHM, dur)]
        protocol.phases.append((phase, interventions))
    return protocol


def arousal_regulation_protocol(duration_min: float = 45.0) -> SessionProtocol:
    return build_protocol(TherapyGoal.AROUSAL_REGULATION, duration_min)


def emotional_processing_protocol(duration_min: float = 45.0) -> SessionProtocol:
    return build_protocol(TherapyGoal.EMOTIONAL_PROCESSING, duration_min)
