"""Music selection for therapeutic contexts: mood matching, energy regulation, playlist building."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class TherapeuticIntent(Enum):
    CALMING = "calming"
    ENERGIZING = "energizing"
    FOCUS = "focus"
    EMOTIONAL_RELEASE = "emotional_release"
    SOCIAL_BONDING = "social_bonding"
    PAIN_MANAGEMENT = "pain_management"
    SLEEP = "sleep"
    MOTIVATION = "motivation"


@dataclass(frozen=True)
class TrackProfile:
    track_id: str
    tempo_bpm: float
    valence: float
    energy: float
    mode: int = 0
    acousticness: float = 0.5
    instrumentalness: float = 0.5


INTENT_TARGETS: dict[TherapeuticIntent, dict[str, tuple[float, float]]] = {
    TherapeuticIntent.CALMING: {
        "tempo_bpm": (55, 80), "valence": (0.3, 0.6),
        "energy": (0.1, 0.35), "acousticness": (0.5, 1.0),
    },
    TherapeuticIntent.ENERGIZING: {
        "tempo_bpm": (110, 140), "valence": (0.6, 1.0),
        "energy": (0.7, 1.0),
    },
    TherapeuticIntent.FOCUS: {
        "tempo_bpm": (90, 120), "energy": (0.2, 0.5),
        "instrumentalness": (0.6, 1.0),
    },
    TherapeuticIntent.EMOTIONAL_RELEASE: {
        "valence": (0.2, 0.7), "energy": (0.3, 0.8),
    },
    TherapeuticIntent.SLEEP: {
        "tempo_bpm": (50, 70), "energy": (0.0, 0.2),
        "acousticness": (0.6, 1.0), "instrumentalness": (0.4, 1.0),
    },
    TherapeuticIntent.MOTIVATION: {
        "tempo_bpm": (100, 135), "valence": (0.5, 1.0),
        "energy": (0.6, 0.95),
    },
}


def intent_fitness(track: TrackProfile, intent: TherapeuticIntent) -> float:
    targets = INTENT_TARGETS.get(intent, {})
    if not targets:
        return 0.5
    scores: list[float] = []
    for attr, (lo, hi) in targets.items():
        val = getattr(track, attr, None)
        if val is None:
            continue
        if lo <= val <= hi:
            scores.append(1.0)
        elif val < lo:
            scores.append(max(0.0, 1.0 - (lo - val) / max(lo, 0.01)))
        else:
            scores.append(max(0.0, 1.0 - (val - hi) / max(1.0 - hi, 0.01)))
    return float(np.mean(scores)) if scores else 0.5


def rank_tracks(
    tracks: Sequence[TrackProfile],
    intent: TherapeuticIntent,
) -> list[tuple[str, float]]:
    scored = [(t.track_id, intent_fitness(t, intent)) for t in tracks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


@dataclass(frozen=True)
class TherapeuticPlaylist:
    tracks: list[str]
    intent: TherapeuticIntent
    mean_fitness: float
    duration_estimate_min: float


def build_therapeutic_playlist(
    tracks: Sequence[TrackProfile],
    intent: TherapeuticIntent,
    target_duration_min: float = 30.0,
    avg_track_min: float = 3.5,
) -> TherapeuticPlaylist:
    ranked = rank_tracks(tracks, intent)
    n_tracks = max(1, int(target_duration_min / avg_track_min))
    selected = ranked[:n_tracks]
    ids = [tid for tid, _ in selected]
    mean_fit = float(np.mean([s for _, s in selected])) if selected else 0.0
    return TherapeuticPlaylist(
        tracks=ids,
        intent=intent,
        mean_fitness=mean_fit,
        duration_estimate_min=len(ids) * avg_track_min,
    )


def iso_principle_sequence(
    tracks: Sequence[TrackProfile],
    start_energy: float,
    target_energy: float,
    n_steps: int = 6,
) -> list[str]:
    energy_ramp = np.linspace(start_energy, target_energy, n_steps)
    sequence: list[str] = []
    used: set[str] = set()
    for target_e in energy_ramp:
        best_id = ""
        best_dist = float("inf")
        for t in tracks:
            if t.track_id in used:
                continue
            dist = abs(t.energy - target_e)
            if dist < best_dist:
                best_dist = dist
                best_id = t.track_id
        if best_id:
            sequence.append(best_id)
            used.add(best_id)
    return sequence


def session_energy_profile(
    tracks: Sequence[TrackProfile],
    track_ids: Sequence[str],
) -> np.ndarray:
    lookup = {t.track_id: t.energy for t in tracks}
    return np.array([lookup.get(tid, 0.5) for tid in track_ids])
