"""Motor-impairment adaptation: interval simplification, timing quantization,
polyphony reduction, tremor filtering, fatigue modelling, and score adaptation."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple
import numpy as np


class MotorProfile(str, Enum):
    FULL_RANGE = "full_range"
    LIMITED_RANGE = "limited_range"
    SINGLE_HAND = "single_hand"
    HEAD_SWITCH = "head_switch"
    EYE_GAZE = "eye_gaze"
    TREMOR = "tremor"
    SLOW_RESPONSE = "slow_response"


@dataclass
class MotorCapability:
    profile: MotorProfile
    max_reach_semitones: int
    min_note_duration_ms: float
    max_simultaneous_keys: int
    reaction_time_ms: float
    accuracy_radius_semitones: float


DEFAULT_CAPABILITIES: Dict[MotorProfile, MotorCapability] = {
    MotorProfile.FULL_RANGE: MotorCapability(MotorProfile.FULL_RANGE, 24, 50, 10, 150, 0.0),
    MotorProfile.LIMITED_RANGE: MotorCapability(MotorProfile.LIMITED_RANGE, 12, 100, 5, 200, 1.0),
    MotorProfile.SINGLE_HAND: MotorCapability(MotorProfile.SINGLE_HAND, 8, 120, 3, 200, 1.0),
    MotorProfile.HEAD_SWITCH: MotorCapability(MotorProfile.HEAD_SWITCH, 4, 500, 1, 600, 2.0),
    MotorProfile.EYE_GAZE: MotorCapability(MotorProfile.EYE_GAZE, 6, 400, 1, 500, 2.0),
    MotorProfile.TREMOR: MotorCapability(MotorProfile.TREMOR, 12, 150, 5, 250, 3.0),
    MotorProfile.SLOW_RESPONSE: MotorCapability(MotorProfile.SLOW_RESPONSE, 18, 300, 5, 400, 1.0),
}


def simplify_intervals(melody_midi: np.ndarray, max_reach: int) -> np.ndarray:
    if len(melody_midi) < 2:
        return melody_midi.copy()
    result = melody_midi.astype(float).copy()
    for i in range(1, len(result)):
        diff = result[i] - result[i - 1]
        if abs(diff) > max_reach:
            direction = 1 if diff > 0 else -1
            while abs(result[i] - result[i - 1]) > max_reach:
                result[i] -= direction * 12
    return result


def quantize_timing(onsets_ms: np.ndarray, min_duration_ms: float) -> np.ndarray:
    if len(onsets_ms) < 2:
        return onsets_ms.copy()
    result = onsets_ms.copy().astype(float)
    for i in range(1, len(result)):
        gap = result[i] - result[i - 1]
        if gap < min_duration_ms:
            result[i] = result[i - 1] + min_duration_ms
    return result


def reduce_polyphony(chord_midi: np.ndarray, max_keys: int) -> np.ndarray:
    if len(chord_midi) <= max_keys:
        return chord_midi.copy()
    sorted_notes = np.sort(chord_midi)[::-1]
    return sorted_notes[:max_keys]


def adaptive_tempo(bpm: float, reaction_time_ms: float) -> float:
    min_ioi_ms = reaction_time_ms * 1.5
    max_bpm = 60000.0 / min_ioi_ms
    return min(bpm, max_bpm)


def target_prediction(
    recent_midi: np.ndarray, recent_times: np.ndarray,
) -> Tuple[int, float]:
    if len(recent_midi) < 2:
        return int(recent_midi[-1]) if len(recent_midi) > 0 else 60, 0.0
    dt = recent_times[-1] - recent_times[-2]
    dp = recent_midi[-1] - recent_midi[-2]
    if dt < 1e-6:
        return int(recent_midi[-1]), recent_times[-1]
    velocity = dp / dt
    predicted_pitch = recent_midi[-1] + velocity * dt
    predicted_time = recent_times[-1] + dt
    return int(round(predicted_pitch)), float(predicted_time)


def tremor_filter(midi_values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(midi_values) < window:
        return midi_values.copy()
    half = window // 2
    result = midi_values.copy().astype(float)
    for i in range(half, len(midi_values) - half):
        segment = midi_values[i - half : i + half + 1]
        result[i] = np.median(segment)
    return result


def error_tolerance_score(
    intended: np.ndarray, actual: np.ndarray, radius: float,
) -> float:
    n = min(len(intended), len(actual))
    if n == 0:
        return 1.0
    diffs = np.abs(intended[:n] - actual[:n])
    return float(np.mean(diffs <= radius))


def fatigue_model(
    durations_ms: np.ndarray, rest_threshold_ms: float = 500.0,
) -> np.ndarray:
    fatigue = np.zeros(len(durations_ms))
    cumulative = 0.0
    for i, d in enumerate(durations_ms):
        if d >= rest_threshold_ms:
            cumulative = max(0.0, cumulative - d * 0.001)
        else:
            cumulative += d * 0.002
        fatigue[i] = cumulative
    max_f = fatigue.max() if fatigue.max() > 0 else 1.0
    return fatigue / max_f


def difficulty_estimate(
    melody_midi: np.ndarray, onsets_ms: np.ndarray,
) -> float:
    if len(melody_midi) < 2:
        return 0.0
    intervals = np.abs(np.diff(melody_midi))
    iois = np.diff(onsets_ms)
    avg_interval = float(np.mean(intervals))
    avg_speed = float(np.mean(1000.0 / (iois + 1.0)))
    return min(1.0, avg_interval / 24.0 * 0.5 + avg_speed * 0.5)


def adapt_score(
    melody_midi: np.ndarray,
    onsets_ms: np.ndarray,
    capability: MotorCapability,
) -> Tuple[np.ndarray, np.ndarray]:
    adapted_pitch = simplify_intervals(melody_midi, capability.max_reach_semitones)
    adapted_timing = quantize_timing(onsets_ms, capability.min_note_duration_ms)
    if capability.profile == MotorProfile.TREMOR:
        adapted_pitch = tremor_filter(adapted_pitch)
    return adapted_pitch, adapted_timing
