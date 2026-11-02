"""Rhythm-based wellness interventions: entrainment protocols, drum circles, movement."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RhythmPattern(Enum):
    STEADY_PULSE = "steady_pulse"
    CALL_RESPONSE = "call_response"
    POLYRHYTHM = "polyrhythm"
    ACCELERANDO = "accelerando"
    RITARDANDO = "ritardando"
    SYNCOPATED = "syncopated"
    HEARTBEAT = "heartbeat"
    BREATHING = "breathing"


@dataclass
class RhythmicProfile:
    base_tempo: float = 80.0
    tempo_variance: float = 0.0
    subdivision: int = 4
    accent_pattern: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.7, 0.5])
    swing_ratio: float = 0.5


def generate_entrainment_tempo_curve(
    start_tempo: float,
    target_tempo: float,
    duration_s: float,
    sample_rate_hz: float = 2.0,
) -> list[float]:
    if duration_s <= 0 or sample_rate_hz <= 0:
        return []
    n = int(duration_s * sample_rate_hz)
    t = np.linspace(0, 1, max(n, 2))
    curve = start_tempo + (target_tempo - start_tempo) * (3 * t ** 2 - 2 * t ** 3)
    return [float(x) for x in curve]


def heartbeat_synchronization_score(
    beat_intervals_ms: list[float],
    heart_rate_bpm: float,
) -> float:
    if not beat_intervals_ms or heart_rate_bpm <= 0:
        return 0.0
    target_interval_ms = 60000.0 / heart_rate_bpm
    intervals = np.array(beat_intervals_ms)
    ratios = intervals / target_interval_ms
    nearest_integer_ratios = np.round(ratios)
    nearest_integer_ratios[nearest_integer_ratios == 0] = 1.0
    deviations = np.abs(ratios - nearest_integer_ratios) / nearest_integer_ratios
    mean_dev = float(np.mean(deviations))
    return float(np.clip(1.0 - mean_dev * 5.0, 0.0, 1.0))


def breathing_rhythm_profile(
    inhale_s: float = 4.0,
    hold_s: float = 0.0,
    exhale_s: float = 6.0,
    rest_s: float = 0.0,
    cycles: int = 10,
) -> dict[str, float]:
    cycle_s = inhale_s + hold_s + exhale_s + rest_s
    if cycle_s <= 0:
        return {"breaths_per_min": 0.0, "ie_ratio": 0.0, "suggested_tempo": 0.0}
    bpm_breath = 60.0 / cycle_s
    ie_ratio = inhale_s / exhale_s if exhale_s > 0 else float("inf")
    suggested_tempo = 60.0 / cycle_s * 4
    return {
        "breaths_per_min": bpm_breath,
        "ie_ratio": ie_ratio,
        "suggested_tempo": suggested_tempo,
        "total_duration_s": cycle_s * cycles,
    }


@dataclass
class DrumCircleConfig:
    n_participants: int = 8
    base_tempo: float = 90.0
    complexity_level: float = 0.5
    pattern: RhythmPattern = RhythmPattern.STEADY_PULSE


def drum_circle_cohesion(
    onset_times_per_participant: list[list[float]],
    expected_beat_interval_s: float,
) -> dict[str, float]:
    if not onset_times_per_participant or expected_beat_interval_s <= 0:
        return {"cohesion": 0.0, "mean_deviation_ms": 0.0, "weakest_participant": -1}
    deviations_by_participant: list[float] = []
    for onsets in onset_times_per_participant:
        if len(onsets) < 2:
            deviations_by_participant.append(float("inf"))
            continue
        intervals = np.diff(onsets)
        dev = float(np.mean(np.abs(intervals - expected_beat_interval_s)))
        deviations_by_participant.append(dev)
    devs = np.array(deviations_by_participant)
    finite_mask = np.isfinite(devs)
    if not np.any(finite_mask):
        return {"cohesion": 0.0, "mean_deviation_ms": 0.0, "weakest_participant": 0}
    mean_dev = float(np.mean(devs[finite_mask]))
    cohesion = float(np.clip(1.0 - mean_dev / expected_beat_interval_s * 2.0, 0.0, 1.0))
    weakest = int(np.argmax(devs)) if len(devs) > 0 else -1
    return {
        "cohesion": cohesion,
        "mean_deviation_ms": mean_dev * 1000.0,
        "weakest_participant": weakest,
    }


def movement_tempo_matching(
    movement_frequency_hz: float,
    music_tempo_bpm: float,
) -> float:
    if movement_frequency_hz <= 0 or music_tempo_bpm <= 0:
        return 0.0
    music_freq = music_tempo_bpm / 60.0
    ratio = movement_frequency_hz / music_freq
    nearest = round(ratio)
    if nearest <= 0:
        nearest = 1
    deviation = abs(ratio - nearest) / nearest
    return float(np.clip(1.0 - deviation * 4.0, 0.0, 1.0))


def rhythmic_complexity_index(
    onset_times: list[float],
    meter_numerator: int = 4,
    beat_duration_s: float = 0.5,
) -> float:
    if len(onset_times) < 2 or beat_duration_s <= 0:
        return 0.0
    phases = np.array(onset_times) % (meter_numerator * beat_duration_s)
    grid_positions = phases / beat_duration_s
    grid_offsets = np.abs(grid_positions - np.round(grid_positions))
    off_grid_fraction = float(np.mean(grid_offsets > 0.15))
    intervals = np.diff(onset_times)
    if len(intervals) < 2:
        return off_grid_fraction
    cv = float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0.0
    return float(np.clip(0.5 * off_grid_fraction + 0.5 * min(cv, 1.0), 0.0, 1.0))


def adaptive_tempo_recommendation(
    current_heart_rate: float,
    target_state: str = "calm",
    current_tempo: float = 100.0,
) -> float:
    if target_state == "calm":
        target_hr = min(current_heart_rate, 65.0)
    elif target_state == "energized":
        target_hr = max(current_heart_rate, 100.0)
    elif target_state == "focused":
        target_hr = 72.0
    else:
        target_hr = current_heart_rate
    hr_diff = target_hr - current_heart_rate
    tempo_adjustment = hr_diff * 0.3
    new_tempo = current_tempo + tempo_adjustment
    return float(np.clip(new_tempo, 40.0, 200.0))
