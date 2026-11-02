"""Embodied music cognition: movement-sound coupling, gesture analysis, entrainment."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BodyPart:
    name: str
    mass_fraction: float
    resonant_freq_hz: float

BODY_PARTS = {
    "head": BodyPart("head", 0.08, 2.0),
    "torso": BodyPart("torso", 0.50, 0.5),
    "arm": BodyPart("arm", 0.05, 1.5),
    "hand": BodyPart("hand", 0.006, 4.0),
    "leg": BodyPart("leg", 0.16, 0.8),
    "foot": BodyPart("foot", 0.014, 2.5),
}


@dataclass
class MovementPhase:
    onset: float
    peak: float
    offset: float
    magnitude: float
    body_part: str = "unspecified"


def movement_periodicity(positions: np.ndarray, sr: float) -> Tuple[float, float]:
    if len(positions) < 4:
        return (0.0, 0.0)
    positions = positions - np.mean(positions)
    autocorr = np.correlate(positions, positions, mode="full")
    autocorr = autocorr[len(positions) - 1:]
    autocorr /= autocorr[0] + 1e-12
    peaks = []
    for i in range(1, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            if autocorr[i] > 0.1:
                peaks.append(i)
    if not peaks:
        return (0.0, 0.0)
    period_samples = peaks[0]
    freq = sr / period_samples
    strength = float(autocorr[period_samples])
    return (freq, strength)


def entrainment_strength(movement_onsets: np.ndarray, beat_times: np.ndarray) -> float:
    if len(movement_onsets) == 0 or len(beat_times) == 0:
        return 0.0
    phases = []
    for onset in movement_onsets:
        diffs = beat_times - onset
        closest_idx = np.argmin(np.abs(diffs))
        if closest_idx < len(beat_times) - 1:
            beat_period = beat_times[closest_idx + 1] - beat_times[closest_idx]
        elif closest_idx > 0:
            beat_period = beat_times[closest_idx] - beat_times[closest_idx - 1]
        else:
            continue
        if beat_period > 0:
            phase = (onset - beat_times[closest_idx]) / beat_period
            phase = phase % 1.0
            phases.append(phase)
    if not phases:
        return 0.0
    angles = np.array(phases) * 2 * np.pi
    R = np.abs(np.mean(np.exp(1j * angles)))
    return float(R)


def phase_coupling(signal1: np.ndarray, signal2: np.ndarray) -> float:
    if len(signal1) != len(signal2) or len(signal1) == 0:
        return 0.0
    analytic1 = np.fft.ifft(np.fft.fft(signal1) * 2)
    analytic2 = np.fft.ifft(np.fft.fft(signal2) * 2)
    phase_diff = np.angle(analytic1) - np.angle(analytic2)
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return float(plv)


def kinetic_energy(velocity: np.ndarray, mass: float = 1.0) -> np.ndarray:
    return 0.5 * mass * velocity ** 2


def movement_fluency(positions: np.ndarray, sr: float) -> float:
    if len(positions) < 3:
        return 1.0
    velocity = np.diff(positions) * sr
    acceleration = np.diff(velocity) * sr
    jerk = np.diff(acceleration) * sr
    if len(jerk) == 0:
        return 1.0
    jerk_rms = np.sqrt(np.mean(jerk ** 2))
    duration = len(positions) / sr
    path_length = np.sum(np.abs(np.diff(positions)))
    normalized_jerk = jerk_rms * duration ** 2.5 / (path_length + 1e-12)
    return float(1.0 / (1.0 + normalized_jerk / 100.0))


def gesture_segmentation(velocity: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, int]]:
    moving = np.abs(velocity) > threshold
    segments = []
    in_segment = False
    start = 0
    for i in range(len(moving)):
        if moving[i] and not in_segment:
            start = i
            in_segment = True
        elif not moving[i] and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, len(moving)))
    return segments


def body_resonance(freq: float, body_part: str = "torso") -> float:
    bp = BODY_PARTS.get(body_part, BODY_PARTS["torso"])
    ratio = freq / bp.resonant_freq_hz
    Q = 2.0
    response = 1.0 / np.sqrt((1 - ratio ** 2) ** 2 + (ratio / Q) ** 2)
    return float(np.clip(response, 0, 10))


def optimal_tempo_for_movement(body_part: str = "leg",
                                movement_type: str = "walk") -> float:
    base_freqs = {
        "head_nod": BODY_PARTS["head"].resonant_freq_hz,
        "walk": BODY_PARTS["leg"].resonant_freq_hz,
        "arm_swing": BODY_PARTS["arm"].resonant_freq_hz,
        "tap": BODY_PARTS["hand"].resonant_freq_hz,
        "bounce": BODY_PARTS["torso"].resonant_freq_hz,
    }
    freq = base_freqs.get(movement_type, 1.0)
    return freq * 60.0


def movement_audio_synchrony(movement: np.ndarray, audio_envelope: np.ndarray,
                              sr: float) -> float:
    min_len = min(len(movement), len(audio_envelope))
    if min_len < 2:
        return 0.0
    m = movement[:min_len]
    a = audio_envelope[:min_len]
    m = (m - np.mean(m)) / (np.std(m) + 1e-12)
    a = (a - np.mean(a)) / (np.std(a) + 1e-12)
    corr = np.correlate(m, a, mode="full")
    corr /= min_len
    max_lag = int(sr * 0.5)
    center = min_len - 1
    window = corr[max(0, center - max_lag):center + max_lag + 1]
    if len(window) == 0:
        return 0.0
    best_corr = np.max(np.abs(window))
    return float(np.clip(best_corr, 0, 1))


def groove_body_response(beat_positions: np.ndarray, microtiming: np.ndarray) -> float:
    if len(beat_positions) == 0 or len(microtiming) == 0:
        return 0.0
    min_len = min(len(beat_positions), len(microtiming))
    mt = microtiming[:min_len]
    ioi = np.diff(beat_positions[:min_len + 1]) if len(beat_positions) > min_len else np.diff(beat_positions)
    if len(ioi) < min_len:
        min_len = len(ioi)
        mt = mt[:min_len]
    relative_timing = mt[:min_len] / (ioi[:min_len] + 1e-12)
    swing = np.std(relative_timing)
    is_swing = 0.05 < swing < 0.15
    systematic_deviation = np.abs(np.mean(relative_timing))
    is_systematic = systematic_deviation > 0.02
    if is_swing and is_systematic:
        return 0.9
    elif is_swing:
        return 0.7
    elif is_systematic:
        return 0.5
    return 0.3


@dataclass
class DanceMetrics:
    synchrony: float
    fluency: float
    periodicity_freq: float
    periodicity_strength: float
    body_resonance_match: float


def analyze_dance(position_trajectory: np.ndarray, beat_times: np.ndarray,
                  sr: float, tempo_bpm: float = 120.0) -> DanceMetrics:
    velocity = np.diff(position_trajectory, axis=0) * sr if position_trajectory.ndim > 1 else np.diff(position_trajectory) * sr
    if position_trajectory.ndim > 1:
        speed = np.sqrt(np.sum(velocity ** 2, axis=1))
    else:
        speed = np.abs(velocity)
    onsets = []
    for i in range(1, len(speed) - 1):
        if speed[i] > speed[i - 1] and speed[i] > speed[i + 1] and speed[i] > 0.1:
            onsets.append(i / sr)
    synchrony = entrainment_strength(np.array(onsets), beat_times) if onsets else 0.0
    pos_1d = position_trajectory[:, 0] if position_trajectory.ndim > 1 else position_trajectory
    fluency = movement_fluency(pos_1d, sr)
    freq, strength = movement_periodicity(pos_1d, sr)
    beat_freq = tempo_bpm / 60.0
    resonance = body_resonance(beat_freq, "leg")
    return DanceMetrics(
        synchrony=synchrony, fluency=fluency,
        periodicity_freq=freq, periodicity_strength=strength,
        body_resonance_match=float(np.clip(resonance / 5.0, 0, 1)),
    )
