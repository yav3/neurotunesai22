"""Emotional response: BRECVEMA mechanisms, VAD mapping, chills, nostalgia."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class BRECVEMAMechanism(Enum):
    BRAINSTEM_REFLEX = "brainstem_reflex"
    RHYTHMIC_ENTRAINMENT = "rhythmic_entrainment"
    EVALUATIVE_CONDITIONING = "evaluative_conditioning"
    EMOTIONAL_CONTAGION = "emotional_contagion"
    VISUAL_IMAGERY = "visual_imagery"
    EPISODIC_MEMORY = "episodic_memory"
    MUSICAL_EXPECTANCY = "musical_expectancy"
    AESTHETIC_JUDGMENT = "aesthetic_judgment"


@dataclass
class EmotionalResponse:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.5
    chills_probability: float = 0.0
    nostalgia_score: float = 0.0
    tension_resolution: float = 0.0
    active_mechanisms: list[BRECVEMAMechanism] = field(default_factory=list)


def predict_emotional_response(
    tempo: float,
    mode: float,
    loudness_db: float,
    spectral_centroid: float = 2000.0,
    harmonic_complexity: float = 0.5,
) -> EmotionalResponse:
    arousal_raw = (
        0.35 * np.clip((tempo - 60) / 140, 0, 1)
        + 0.25 * np.clip((loudness_db + 60) / 60, 0, 1)
        + 0.20 * np.clip(spectral_centroid / 8000, 0, 1)
        + 0.20 * harmonic_complexity
    )
    valence_raw = (
        0.40 * mode
        + 0.20 * np.clip((tempo - 80) / 120, 0, 1)
        + 0.20 * (1.0 - harmonic_complexity)
        + 0.20 * np.clip(spectral_centroid / 6000, 0, 1)
    )
    arousal = float(np.clip(arousal_raw * 2 - 1, -1, 1))
    valence = float(np.clip(valence_raw * 2 - 1, -1, 1))
    dominance = float(np.clip(0.3 * arousal + 0.3 * valence + 0.5, 0, 1))
    mechanisms = mechanism_activation_profile(tempo, mode, loudness_db, spectral_centroid)
    return EmotionalResponse(
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        active_mechanisms=mechanisms,
    )


def chills_predictor(
    harmonic_surprise: float = 0.0,
    dynamic_contrast: float = 0.0,
    textural_change: float = 0.0,
    melodic_appoggiatura: float = 0.0,
    new_voice_entry: float = 0.0,
) -> float:
    raw = (
        0.30 * harmonic_surprise
        + 0.25 * dynamic_contrast
        + 0.20 * textural_change
        + 0.15 * melodic_appoggiatura
        + 0.10 * new_voice_entry
    )
    return float(1.0 / (1.0 + np.exp(-6.0 * (raw - 0.5))))


def nostalgia_index(
    familiarity: float = 0.0,
    personal_association: float = 0.0,
    temporal_distance_years: float = 0.0,
    tempo: float = 120.0,
) -> float:
    temporal_factor = 1.0 - np.exp(-temporal_distance_years / 10.0) if temporal_distance_years > 0 else 0.0
    tempo_penalty = np.exp(-((tempo - 90) ** 2) / (2 * 30 ** 2))
    raw = (
        0.35 * familiarity
        + 0.30 * personal_association
        + 0.20 * temporal_factor
        + 0.15 * tempo_penalty
    )
    return float(np.clip(raw, 0.0, 1.0))


def mechanism_activation_profile(
    tempo: float,
    mode: float,
    loudness_db: float,
    spectral_centroid: float = 2000.0,
) -> list[BRECVEMAMechanism]:
    active: list[BRECVEMAMechanism] = []
    if loudness_db > -10 or spectral_centroid > 5000:
        active.append(BRECVEMAMechanism.BRAINSTEM_REFLEX)
    if 60 <= tempo <= 180:
        active.append(BRECVEMAMechanism.RHYTHMIC_ENTRAINMENT)
    if mode > 0.5 or mode < 0.2:
        active.append(BRECVEMAMechanism.EMOTIONAL_CONTAGION)
    active.append(BRECVEMAMechanism.MUSICAL_EXPECTANCY)
    return active


def valence_arousal_from_features(
    tempo: float,
    mode: float,
    loudness_db: float,
    pitch_range_semitones: float = 12.0,
    rhythmic_regularity: float = 0.5,
) -> tuple[float, float]:
    arousal = float(np.tanh(
        0.01 * (tempo - 100)
        + 0.02 * (loudness_db + 30)
        + 0.02 * (pitch_range_semitones - 12)
    ))
    valence = float(np.tanh(
        1.0 * (mode - 0.5)
        + 0.005 * (tempo - 100)
        + 0.5 * (rhythmic_regularity - 0.5)
    ))
    return valence, arousal


def emotion_trajectory(
    valence_series: list[float],
    arousal_series: list[float],
) -> dict[str, float]:
    if len(valence_series) < 2 or len(arousal_series) < 2:
        return {"valence_range": 0.0, "arousal_range": 0.0, "emotional_flux": 0.0}
    v = np.array(valence_series)
    a = np.array(arousal_series)
    dv = np.diff(v)
    da = np.diff(a)
    distances = np.sqrt(dv ** 2 + da ** 2)
    return {
        "valence_range": float(np.ptp(v)),
        "arousal_range": float(np.ptp(a)),
        "emotional_flux": float(np.mean(distances)),
    }
