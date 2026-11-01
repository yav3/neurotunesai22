"""Wellness protocols: non-clinical music-based wellness routines and scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class WellnessGoal(Enum):
    STRESS_REDUCTION = "stress_reduction"
    FOCUS_ENHANCEMENT = "focus_enhancement"
    SLEEP_PREPARATION = "sleep_preparation"
    ENERGY_BOOST = "energy_boost"
    MOOD_ELEVATION = "mood_elevation"
    MEDITATION_SUPPORT = "meditation_support"
    CREATIVE_FLOW = "creative_flow"
    SOCIAL_BONDING = "social_bonding"


@dataclass
class WellnessTrack:
    tempo_bpm: float = 120.0
    mode: float = 0.5
    energy: float = 0.5
    valence: float = 0.5
    acousticness: float = 0.5
    instrumentalness: float = 0.5
    duration_s: float = 240.0


GOAL_PROFILES: dict[WellnessGoal, dict[str, tuple[float, float]]] = {
    WellnessGoal.STRESS_REDUCTION: {
        "tempo_bpm": (55, 75), "energy": (0.1, 0.35), "valence": (0.3, 0.7),
        "acousticness": (0.6, 1.0), "instrumentalness": (0.5, 1.0),
    },
    WellnessGoal.FOCUS_ENHANCEMENT: {
        "tempo_bpm": (100, 130), "energy": (0.3, 0.5), "valence": (0.4, 0.6),
        "acousticness": (0.2, 0.8), "instrumentalness": (0.7, 1.0),
    },
    WellnessGoal.SLEEP_PREPARATION: {
        "tempo_bpm": (50, 70), "energy": (0.0, 0.2), "valence": (0.3, 0.6),
        "acousticness": (0.7, 1.0), "instrumentalness": (0.6, 1.0),
    },
    WellnessGoal.ENERGY_BOOST: {
        "tempo_bpm": (120, 160), "energy": (0.7, 1.0), "valence": (0.6, 1.0),
        "acousticness": (0.0, 0.4), "instrumentalness": (0.0, 0.5),
    },
    WellnessGoal.MOOD_ELEVATION: {
        "tempo_bpm": (100, 140), "energy": (0.5, 0.8), "valence": (0.7, 1.0),
        "acousticness": (0.2, 0.7), "instrumentalness": (0.0, 0.5),
    },
    WellnessGoal.MEDITATION_SUPPORT: {
        "tempo_bpm": (40, 65), "energy": (0.0, 0.15), "valence": (0.4, 0.6),
        "acousticness": (0.8, 1.0), "instrumentalness": (0.8, 1.0),
    },
    WellnessGoal.CREATIVE_FLOW: {
        "tempo_bpm": (90, 125), "energy": (0.3, 0.6), "valence": (0.4, 0.7),
        "acousticness": (0.3, 0.8), "instrumentalness": (0.5, 1.0),
    },
    WellnessGoal.SOCIAL_BONDING: {
        "tempo_bpm": (100, 135), "energy": (0.5, 0.8), "valence": (0.6, 0.9),
        "acousticness": (0.1, 0.6), "instrumentalness": (0.0, 0.3),
    },
}


def goal_fitness(track: WellnessTrack, goal: WellnessGoal) -> float:
    profile = GOAL_PROFILES.get(goal)
    if profile is None:
        return 0.0
    scores: list[float] = []
    for attr, (lo, hi) in profile.items():
        val = getattr(track, attr, None)
        if val is None:
            continue
        if lo <= val <= hi:
            mid = (lo + hi) / 2.0
            half_range = (hi - lo) / 2.0
            scores.append(1.0 - abs(val - mid) / half_range * 0.3)
        else:
            dist = min(abs(val - lo), abs(val - hi))
            scores.append(max(0.0, 1.0 - dist * 2.0))
    return float(np.mean(scores)) if scores else 0.0


def rank_wellness_tracks(
    tracks: list[WellnessTrack],
    goal: WellnessGoal,
) -> list[tuple[int, float]]:
    scored = [(i, goal_fitness(t, goal)) for i, t in enumerate(tracks)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


@dataclass
class WellnessSession:
    goal: WellnessGoal
    duration_minutes: float = 30.0
    phases: list[str] = field(default_factory=list)
    target_tempo_curve: list[float] = field(default_factory=list)
    target_energy_curve: list[float] = field(default_factory=list)


def build_wellness_session(
    goal: WellnessGoal,
    duration_minutes: float = 30.0,
    n_phases: int = 5,
) -> WellnessSession:
    t = np.linspace(0, 1, n_phases)
    profile = GOAL_PROFILES.get(goal, {})
    tempo_range = profile.get("tempo_bpm", (80, 120))
    energy_range = profile.get("energy", (0.3, 0.6))

    if goal == WellnessGoal.STRESS_REDUCTION:
        tempo_curve = [float(np.interp(x, [0, 0.3, 1], [90, tempo_range[1], tempo_range[0]])) for x in t]
        energy_curve = [float(np.interp(x, [0, 0.3, 1], [0.5, energy_range[1], energy_range[0]])) for x in t]
        phases = ["grounding", "matching", "gradual_descent", "deep_calm", "closing"]
    elif goal == WellnessGoal.ENERGY_BOOST:
        tempo_curve = [float(np.interp(x, [0, 0.7, 1], [tempo_range[0], tempo_range[1], tempo_range[1] - 10])) for x in t]
        energy_curve = [float(np.interp(x, [0, 0.7, 1], [energy_range[0], energy_range[1], energy_range[1] - 0.1])) for x in t]
        phases = ["warm_up", "build", "peak", "sustain", "cool_down"]
    elif goal == WellnessGoal.SLEEP_PREPARATION:
        tempo_curve = [float(np.interp(x, [0, 1], [70, tempo_range[0]])) for x in t]
        energy_curve = [float(np.interp(x, [0, 1], [0.3, energy_range[0]])) for x in t]
        phases = ["transition", "settling", "deepening", "stillness", "silence"]
    elif goal == WellnessGoal.FOCUS_ENHANCEMENT:
        mid_tempo = (tempo_range[0] + tempo_range[1]) / 2
        tempo_curve = [float(np.interp(x, [0, 0.2, 0.8, 1], [90, mid_tempo, mid_tempo, mid_tempo - 5])) for x in t]
        energy_curve = [float(np.interp(x, [0, 0.2, 0.8, 1], [0.3, 0.4, 0.4, 0.35])) for x in t]
        phases = ["entry", "settle", "sustained_focus", "deep_work", "emerge"]
    else:
        mid_tempo = (tempo_range[0] + tempo_range[1]) / 2
        mid_energy = (energy_range[0] + energy_range[1]) / 2
        tempo_curve = [float(mid_tempo)] * n_phases
        energy_curve = [float(mid_energy)] * n_phases
        phases = [f"phase_{i+1}" for i in range(n_phases)]

    return WellnessSession(
        goal=goal,
        duration_minutes=duration_minutes,
        phases=phases[:n_phases],
        target_tempo_curve=tempo_curve,
        target_energy_curve=energy_curve,
    )


def session_adherence_score(
    target_tempos: list[float],
    actual_tempos: list[float],
    target_energies: list[float],
    actual_energies: list[float],
) -> float:
    if not target_tempos or not actual_tempos:
        return 0.0
    n = min(len(target_tempos), len(actual_tempos))
    tt = np.array(target_tempos[:n])
    at = np.array(actual_tempos[:n])
    tempo_err = np.mean(np.abs(tt - at) / np.maximum(tt, 1.0))
    n_e = min(len(target_energies), len(actual_energies))
    if n_e > 0:
        te = np.array(target_energies[:n_e])
        ae = np.array(actual_energies[:n_e])
        energy_err = np.mean(np.abs(te - ae))
    else:
        energy_err = 0.0
    return float(np.clip(1.0 - 0.6 * tempo_err - 0.4 * energy_err, 0.0, 1.0))
