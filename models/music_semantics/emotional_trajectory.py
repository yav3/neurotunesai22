"""Emotional trajectory: modeling how emotional content evolves over
musical time — arc detection, tension-release patterns, climax
identification, and narrative emotional mapping.

Based on the circumplex model (Russell 1980), Huron (2006) expectation
theory, and Juslin & Västfjäll (2008) emotional mechanisms.
Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class EmotionalDimension(Enum):
    VALENCE = "valence"
    AROUSAL = "arousal"
    DOMINANCE = "dominance"
    TENSION = "tension"


class ArcType(Enum):
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    PEAK = "peak"
    VALLEY = "valley"
    PLATEAU = "plateau"
    OSCILLATION = "oscillation"


@dataclass
class EmotionalState:
    valence: float
    arousal: float
    dominance: float = 0.5
    tension: float = 0.0


@dataclass
class EmotionalArc:
    states: List[EmotionalState]
    timestamps: np.ndarray
    arc_type: ArcType = ArcType.PLATEAU
    climax_index: int = 0


@dataclass
class TensionCurve:
    values: np.ndarray
    peaks: List[int] = field(default_factory=list)
    releases: List[int] = field(default_factory=list)


def valence_from_features(mode: float, brightness: float,
                          tempo_norm: float, harmony_consonance: float) -> float:
    v = 0.3 * mode + 0.2 * brightness + 0.2 * tempo_norm + 0.3 * harmony_consonance
    return float(np.clip(v, -1, 1))


def arousal_from_features(tempo_norm: float, loudness_norm: float,
                          onset_density: float, spectral_flux: float) -> float:
    a = 0.3 * tempo_norm + 0.3 * loudness_norm + 0.2 * onset_density + 0.2 * spectral_flux
    return float(np.clip(a, 0, 1))


def tension_from_harmony(dissonance: float, harmonic_rhythm: float,
                         distance_from_tonic: float) -> float:
    t = 0.4 * dissonance + 0.3 * harmonic_rhythm + 0.3 * distance_from_tonic
    return float(np.clip(t, 0, 1))


def emotional_trajectory(valence_curve: np.ndarray,
                         arousal_curve: np.ndarray,
                         timestamps: Optional[np.ndarray] = None) -> EmotionalArc:
    n = len(valence_curve)
    if timestamps is None:
        timestamps = np.arange(n, dtype=float)
    states = []
    for i in range(n):
        v = float(valence_curve[i])
        a = float(arousal_curve[i])
        t = float(np.sqrt(max(0, a ** 2 + (1 - v) ** 2) / 2))
        states.append(EmotionalState(v, a, 0.5, t))

    energy = np.sqrt(valence_curve ** 2 + arousal_curve ** 2)
    climax_idx = int(np.argmax(energy))
    arc_type = _classify_arc(energy)
    return EmotionalArc(states, timestamps, arc_type, climax_idx)


def _classify_arc(energy: np.ndarray) -> ArcType:
    n = len(energy)
    if n < 3:
        return ArcType.PLATEAU
    peak_pos = np.argmax(energy)
    valley_pos = np.argmin(energy)
    first_half_mean = np.mean(energy[:n // 2])
    second_half_mean = np.mean(energy[n // 2:])
    std_e = np.std(energy)
    mean_e = np.mean(energy)

    if std_e < 0.1 * (abs(mean_e) + 1e-10):
        return ArcType.PLATEAU

    if peak_pos < n * 0.3 and second_half_mean < first_half_mean * 0.7:
        return ArcType.RAMP_DOWN
    if peak_pos > n * 0.7 and first_half_mean < second_half_mean * 0.7:
        return ArcType.RAMP_UP

    sign_changes = np.sum(np.abs(np.diff(np.sign(np.diff(energy)))) > 0)
    if sign_changes > n * 0.3:
        return ArcType.OSCILLATION

    if 0.2 * n < peak_pos < 0.8 * n:
        return ArcType.PEAK
    if 0.2 * n < valley_pos < 0.8 * n:
        return ArcType.VALLEY

    return ArcType.PLATEAU


def tension_curve(dissonance_series: np.ndarray,
                  harmonic_rhythm_series: np.ndarray,
                  tonic_distance_series: np.ndarray) -> TensionCurve:
    n = len(dissonance_series)
    values = np.zeros(n)
    for i in range(n):
        values[i] = tension_from_harmony(
            float(dissonance_series[i]),
            float(harmonic_rhythm_series[i]),
            float(tonic_distance_series[i])
        )

    peaks = []
    releases = []
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            releases.append(i)

    return TensionCurve(values, peaks, releases)


def climax_strength(energy: np.ndarray, peak_index: int) -> float:
    if len(energy) == 0:
        return 0.0
    peak_val = float(energy[peak_index])
    mean_val = float(np.mean(energy))
    std_val = float(np.std(energy))
    if std_val < 1e-10:
        return 0.0
    return float((peak_val - mean_val) / std_val)


def emotional_distance(s1: EmotionalState, s2: EmotionalState) -> float:
    dv = s1.valence - s2.valence
    da = s1.arousal - s2.arousal
    dd = s1.dominance - s2.dominance
    return float(np.sqrt(dv ** 2 + da ** 2 + dd ** 2))


def emotional_velocity(arc: EmotionalArc) -> np.ndarray:
    n = len(arc.states)
    if n < 2:
        return np.zeros(max(n - 1, 1))
    vel = np.zeros(n - 1)
    for i in range(n - 1):
        dt = float(arc.timestamps[i + 1] - arc.timestamps[i])
        if dt < 1e-10:
            dt = 1.0
        vel[i] = emotional_distance(arc.states[i], arc.states[i + 1]) / dt
    return vel


def emotional_smoothness(arc: EmotionalArc) -> float:
    vel = emotional_velocity(arc)
    if len(vel) < 2:
        return 1.0
    jerk = np.abs(np.diff(vel))
    return float(1.0 / (1.0 + np.mean(jerk)))


def arc_similarity(arc1: EmotionalArc, arc2: EmotionalArc) -> float:
    n1 = len(arc1.states)
    n2 = len(arc2.states)
    if n1 == 0 or n2 == 0:
        return 0.0
    n = min(n1, n2)
    v1 = np.array([s.valence for s in arc1.states[:n]])
    v2 = np.array([s.valence for s in arc2.states[:n]])
    a1 = np.array([s.arousal for s in arc1.states[:n]])
    a2 = np.array([s.arousal for s in arc2.states[:n]])
    if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
        v_corr = 0.0
    else:
        v_corr = float(np.corrcoef(v1, v2)[0, 1])
    if np.std(a1) < 1e-10 or np.std(a2) < 1e-10:
        a_corr = 0.0
    else:
        a_corr = float(np.corrcoef(a1, a2)[0, 1])
    return 0.5 * v_corr + 0.5 * a_corr


def narrative_segments(tension_values: np.ndarray,
                       threshold: float = 0.3) -> List[Tuple[int, int, str]]:
    n = len(tension_values)
    if n == 0:
        return []
    mean_t = float(np.mean(tension_values))
    segments = []
    start = 0
    current_label = "high" if tension_values[0] > mean_t + threshold else (
        "low" if tension_values[0] < mean_t - threshold else "neutral"
    )
    for i in range(1, n):
        val = float(tension_values[i])
        if val > mean_t + threshold:
            label = "high"
        elif val < mean_t - threshold:
            label = "low"
        else:
            label = "neutral"
        if label != current_label:
            segments.append((start, i, current_label))
            start = i
            current_label = label
    segments.append((start, n, current_label))
    return segments


def emotional_range(arc: EmotionalArc) -> float:
    if len(arc.states) < 2:
        return 0.0
    vals = np.array([[s.valence, s.arousal] for s in arc.states])
    v_range = float(np.max(vals[:, 0]) - np.min(vals[:, 0]))
    a_range = float(np.max(vals[:, 1]) - np.min(vals[:, 1]))
    return float(np.sqrt(v_range ** 2 + a_range ** 2))
