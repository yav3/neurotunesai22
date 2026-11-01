"""Therapeutic session design — VAD scoring, EIF, session planning.

Track–goal score:
    score = 1/(1 + ‖v_track − v_goal‖₂) · bpm_score

Emotional Impact Factor (EIF):
    eif_total   = 0.4·v_norm + 0.3·|a| + 0.3·d
    eif_patient = eif_total · (1 + |v − v_patient|)

VAD phase interpolation (entry → peak → resolution):
    Piecewise linear over n_phases.

Energy progression (5-segment, from therapeuticEngine.ts):
    Maps discrete energy levels [1–5] per session segment.

Goal profiles (from therapeuticGoals.ts):
    6 goals × {valence, arousal, dominance, BPM range, EEG bands}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GoalProfile:
    goal_id: str
    valence_target: float
    arousal_target: float
    dominance_target: float
    bpm_min: float
    bpm_max: float
    bpm_optimal: float
    eeg_bands: list[str] = field(default_factory=list)


@dataclass
class PhaseSpec:
    name: str
    duration_minutes: float
    vad_target: np.ndarray
    bpm_range: tuple[float, float]


@dataclass
class SessionPlan:
    phases: list[PhaseSpec]
    vad_trajectory: np.ndarray
    total_duration_minutes: float
    fibonacci_durations: np.ndarray | None = None


GOAL_PROFILES: dict[str, GoalProfile] = {
    "focus-enhancement": GoalProfile(
        "focus-enhancement", 0.6, 0.7, 0.6, 78, 100, 85, ["beta", "gamma"],
    ),
    "anxiety-support": GoalProfile(
        "anxiety-support", 0.7, 0.1, 0.4, 40, 70, 55, ["alpha", "theta"],
    ),
    "sleep-preparation": GoalProfile(
        "sleep-preparation", 0.5, -0.5, 0.2, 30, 60, 45, ["delta", "theta"],
    ),
    "depression-support": GoalProfile(
        "depression-support", 0.6, 0.1, 0.4, 30, 80, 50, ["beta", "alpha"],
    ),
    "energy-boost": GoalProfile(
        "energy-boost", 0.8, 0.9, 0.7, 120, 160, 140, ["beta", "gamma"],
    ),
    "pain-support": GoalProfile(
        "pain-support", 0.6, 0.2, 0.3, 50, 70, 60, ["theta", "alpha"],
    ),
}

ENERGY_PROGRESSIONS: dict[str, list[int]] = {
    "anxiety-support": [4, 3, 2, 2, 1],
    "focus-enhancement": [2, 3, 4, 4, 3],
    "sleep-preparation": [3, 2, 1, 1, 1],
    "depression-support": [2, 3, 4, 5, 4],
    "energy-boost": [3, 4, 5, 5, 4],
    "pain-support": [3, 2, 2, 1, 1],
}

VAD_PROGRESSIONS: dict[str, dict[str, tuple[float, float, float]]] = {
    "anxiety-support": {
        "entry": (0.2, 0.8, 0.3),
        "peak": (0.5, 0.2, 0.6),
        "resolution": (0.6, 0.3, 0.7),
    },
    "focus-enhancement": {
        "entry": (0.5, 0.4, 0.5),
        "peak": (0.5, 0.7, 0.8),
        "resolution": (0.7, 0.4, 0.6),
    },
    "sleep-preparation": {
        "entry": (0.5, 0.6, 0.5),
        "peak": (0.4, 0.2, 0.3),
        "resolution": (0.3, 0.1, 0.2),
    },
    "depression-support": {
        "entry": (0.2, 0.3, 0.3),
        "peak": (0.6, 0.5, 0.6),
        "resolution": (0.7, 0.6, 0.7),
    },
    "energy-boost": {
        "entry": (0.5, 0.5, 0.5),
        "peak": (0.8, 0.9, 0.8),
        "resolution": (0.7, 0.6, 0.6),
    },
    "pain-support": {
        "entry": (0.3, 0.6, 0.4),
        "peak": (0.6, 0.4, 0.6),
        "resolution": (0.7, 0.3, 0.7),
    },
}


class TherapeuticSolver:

    @staticmethod
    def vad_distance(track_vad: np.ndarray, goal_vad: np.ndarray) -> float:
        """‖v_track − v_goal‖₂."""
        return float(np.linalg.norm(track_vad - goal_vad))

    @staticmethod
    def track_goal_score(
        track_vad: np.ndarray, track_bpm: float, goal: GoalProfile,
    ) -> float:
        """score = 1/(1+d_vad) · bpm_score."""
        goal_vad = np.array([goal.valence_target, goal.arousal_target, goal.dominance_target])
        d = TherapeuticSolver.vad_distance(track_vad, goal_vad)
        vad_score = 1.0 / (1.0 + d)

        if goal.bpm_min <= track_bpm <= goal.bpm_max:
            bpm_score = 1.0
        else:
            nearest = goal.bpm_min if track_bpm < goal.bpm_min else goal.bpm_max
            bpm_score = float(np.exp(-abs(track_bpm - nearest) / 20.0))

        return vad_score * bpm_score

    @staticmethod
    def emotional_impact_factor(
        valence: float,
        arousal: float,
        dominance: float,
        patient_valence: float = 0.5,
    ) -> dict[str, float]:
        eif_valence = (valence + 1) / 2
        eif_arousal = abs(arousal)
        eif_total = 0.4 * eif_valence + 0.3 * eif_arousal + 0.3 * dominance
        eif_patient = eif_total * (1 + abs(valence - patient_valence))
        return {
            "eif_valence": eif_valence,
            "eif_arousal": eif_arousal,
            "eif_total": eif_total,
            "eif_patient": eif_patient,
        }

    @staticmethod
    def phase_vad_interpolation(
        entry: np.ndarray,
        peak: np.ndarray,
        resolution: np.ndarray,
        n_phases: int,
    ) -> np.ndarray:
        """Piecewise linear: first half entry→peak, second half peak→resolution."""
        if n_phases <= 1:
            return entry.reshape(1, -1)
        mid = n_phases // 2
        first = np.linspace(0, 1, mid + 1).reshape(-1, 1)
        second = np.linspace(0, 1, n_phases - mid).reshape(-1, 1)
        seg1 = entry + first * (peak - entry)
        seg2 = peak + second * (resolution - peak)
        return np.vstack([seg1[:-1], seg2])

    @staticmethod
    def energy_to_arousal(energy_level: int) -> float:
        """Map 1–5 energy → 0–1 arousal."""
        return (max(1, min(5, energy_level)) - 1) / 4.0

    @staticmethod
    def plan_session(
        goal_id: str,
        duration_minutes: float,
        n_phases: int = 5,
    ) -> SessionPlan:
        goal = GOAL_PROFILES.get(goal_id, GOAL_PROFILES["focus-enhancement"])
        vad_prog = VAD_PROGRESSIONS.get(goal_id, VAD_PROGRESSIONS["focus-enhancement"])
        energy_prog = ENERGY_PROGRESSIONS.get(goal_id, [3, 3, 3, 3, 3])

        entry = np.array(vad_prog["entry"])
        peak = np.array(vad_prog["peak"])
        resolution = np.array(vad_prog["resolution"])
        vad_traj = TherapeuticSolver.phase_vad_interpolation(entry, peak, resolution, n_phases)

        weights = np.array(energy_prog[:n_phases], dtype=float)
        if len(weights) < n_phases:
            weights = np.pad(weights, (0, n_phases - len(weights)), constant_values=weights[-1])
        weights /= weights.sum()
        durations = weights * duration_minutes

        phases = []
        for i in range(n_phases):
            bpm_lo = goal.bpm_min + (goal.bpm_max - goal.bpm_min) * TherapeuticSolver.energy_to_arousal(energy_prog[min(i, len(energy_prog)-1)]) * 0.5
            bpm_hi = bpm_lo + (goal.bpm_max - goal.bpm_min) * 0.3
            phases.append(PhaseSpec(
                name=f"phase_{i+1}",
                duration_minutes=float(durations[i]),
                vad_target=vad_traj[i],
                bpm_range=(bpm_lo, min(bpm_hi, goal.bpm_max)),
            ))

        return SessionPlan(
            phases=phases,
            vad_trajectory=vad_traj,
            total_duration_minutes=duration_minutes,
        )

    @staticmethod
    def score_track_for_phase(
        track_vad: np.ndarray, track_bpm: float, phase: PhaseSpec,
    ) -> float:
        d = TherapeuticSolver.vad_distance(track_vad, phase.vad_target)
        vad_score = 1.0 / (1.0 + d)
        lo, hi = phase.bpm_range
        if lo <= track_bpm <= hi:
            bpm_score = 1.0
        else:
            nearest = lo if track_bpm < lo else hi
            bpm_score = float(np.exp(-abs(track_bpm - nearest) / 20.0))
        return vad_score * bpm_score
