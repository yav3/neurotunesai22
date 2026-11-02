"""Entrainment protocol: rhythmic auditory stimulation for gait and motor therapy.

Implements the iso-principle tempo ramp, gait-cadence assessment, phase-coupling
analysis, adaptive tempo algorithms, bilateral alternation stimuli, and
multi-session progress tracking for rhythmic entrainment therapy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class EntrainmentPhase(str, Enum):
    ASSESSMENT = "assessment"
    MATCHING = "matching"
    LEADING = "leading"
    TARGET = "target"
    MAINTENANCE = "maintenance"


class RampCurve(str, Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"


@dataclass
class GaitParameter:
    cadence_bpm: float
    stride_length_m: float = 0.0
    symmetry_ratio: float = 1.0
    variability_cv: float = 0.0


@dataclass
class EntrainmentState:
    current_tempo_bpm: float
    target_tempo_bpm: float
    phase: EntrainmentPhase
    elapsed_beats: int = 0
    sync_ratio: float = 0.0


def iso_principle_tempo_ramp(start_bpm: float, target_bpm: float,
                             n_steps: int,
                             curve: RampCurve = RampCurve.LINEAR) -> np.ndarray:
    if n_steps <= 1:
        return np.array([target_bpm])
    t = np.linspace(0, 1, n_steps)
    if curve == RampCurve.LINEAR:
        ramp = t
    elif curve == RampCurve.SIGMOID:
        ramp = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))
        ramp = (ramp - ramp[0]) / (ramp[-1] - ramp[0])
    elif curve == RampCurve.EXPONENTIAL:
        ramp = (np.exp(3 * t) - 1) / (np.exp(3) - 1)
    elif curve == RampCurve.LOGARITHMIC:
        ramp = np.log1p(9 * t) / np.log(10)
    else:
        ramp = t
    return start_bpm + (target_bpm - start_bpm) * ramp


def assess_gait_cadence(step_onsets_ms: np.ndarray) -> GaitParameter:
    if len(step_onsets_ms) < 3:
        return GaitParameter(cadence_bpm=0.0)
    ioi = np.diff(step_onsets_ms)
    ioi = ioi[ioi > 0]
    if len(ioi) == 0:
        return GaitParameter(cadence_bpm=0.0)
    mean_ioi = float(np.mean(ioi))
    cadence = 60_000.0 / mean_ioi
    cv = float(np.std(ioi) / mean_ioi)
    symmetry = 1.0
    if len(ioi) >= 4:
        left = ioi[0::2]
        right = ioi[1::2]
        min_len = min(len(left), len(right))
        if min_len > 0:
            symmetry = float(np.mean(np.minimum(left[:min_len], right[:min_len]) /
                                     np.maximum(left[:min_len], right[:min_len])))
    return GaitParameter(cadence_bpm=cadence, symmetry_ratio=symmetry,
                         variability_cv=cv)


def compute_sync_ratio(patient_onsets_ms: np.ndarray,
                       stimulus_onsets_ms: np.ndarray,
                       tolerance_ms: float = 50.0) -> float:
    if len(patient_onsets_ms) == 0 or len(stimulus_onsets_ms) == 0:
        return 0.0
    synced = 0
    for p in patient_onsets_ms:
        diffs = np.abs(stimulus_onsets_ms - p)
        if np.min(diffs) <= tolerance_ms:
            synced += 1
    return synced / len(patient_onsets_ms)


def phase_coupling_strength(patient_phase: np.ndarray,
                            stimulus_phase: np.ndarray) -> float:
    if len(patient_phase) == 0 or len(stimulus_phase) == 0:
        return 0.0
    n = min(len(patient_phase), len(stimulus_phase))
    diff = patient_phase[:n] - stimulus_phase[:n]
    r = np.abs(np.mean(np.exp(1j * diff)))
    return float(r)


def adapt_tempo(state: EntrainmentState,
                patient_onset_ms: float,
                stimulus_onset_ms: float) -> EntrainmentState:
    error_ms = patient_onset_ms - stimulus_onset_ms
    beat_ms = 60_000.0 / state.current_tempo_bpm
    phase_error = error_ms / beat_ms
    adaptation_rate = 0.05
    new_tempo = state.current_tempo_bpm
    if state.phase == EntrainmentPhase.MATCHING:
        new_tempo += phase_error * adaptation_rate * state.current_tempo_bpm
    elif state.phase == EntrainmentPhase.LEADING:
        direction = 1.0 if state.target_tempo_bpm > state.current_tempo_bpm else -1.0
        step = 0.5
        new_tempo += direction * step
    new_tempo = np.clip(new_tempo, 30.0, 200.0)
    new_phase = state.phase
    if state.phase == EntrainmentPhase.MATCHING and abs(phase_error) < 0.05:
        new_phase = EntrainmentPhase.LEADING
    elif state.phase == EntrainmentPhase.LEADING:
        if abs(new_tempo - state.target_tempo_bpm) < 2.0:
            new_phase = EntrainmentPhase.TARGET
    return EntrainmentState(
        current_tempo_bpm=float(new_tempo),
        target_tempo_bpm=state.target_tempo_bpm,
        phase=new_phase,
        elapsed_beats=state.elapsed_beats + 1,
        sync_ratio=state.sync_ratio,
    )


def metronome_with_groove(bpm: float, duration_s: float,
                          sr: int = 44100,
                          groove_depth: float = 0.0) -> np.ndarray:
    beat_samples = int(60.0 / bpm * sr)
    total_samples = int(duration_s * sr)
    n_beats = total_samples // beat_samples
    click_len = min(int(0.01 * sr), beat_samples)
    output = np.zeros(total_samples)
    rng = np.random.default_rng(42)
    for i in range(n_beats):
        jitter = int(groove_depth * click_len * rng.standard_normal())
        pos = i * beat_samples + jitter
        pos = max(0, min(pos, total_samples - click_len))
        t = np.arange(click_len) / sr
        click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 200)
        amp = 1.0 if i % 4 == 0 else 0.7
        output[pos:pos + click_len] += click * amp
    peak = np.abs(output).max()
    if peak > 0:
        output /= peak
    return output


def bilateral_alternation(bpm: float, duration_s: float,
                          sr: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    beat_samples = int(60.0 / bpm * sr)
    total = int(duration_s * sr)
    n_beats = total // beat_samples
    click_len = min(int(0.01 * sr), beat_samples)
    left = np.zeros(total)
    right = np.zeros(total)
    t = np.arange(click_len) / sr
    click = np.sin(2 * np.pi * 800 * t) * np.exp(-t * 150)
    for i in range(n_beats):
        pos = i * beat_samples
        if pos + click_len > total:
            break
        if i % 2 == 0:
            left[pos:pos + click_len] += click
        else:
            right[pos:pos + click_len] += click
    return left, right


def tempo_stability_window(onsets_ms: np.ndarray,
                           window_size: int = 8) -> np.ndarray:
    if len(onsets_ms) < 3:
        return np.array([0.0])
    ioi = np.diff(onsets_ms)
    if len(ioi) < window_size:
        mu = np.mean(ioi)
        return np.array([np.std(ioi) / max(mu, 1e-6)])
    n_windows = len(ioi) - window_size + 1
    cv = np.zeros(n_windows)
    for i in range(n_windows):
        window = ioi[i:i + window_size]
        mu = np.mean(window)
        cv[i] = np.std(window) / max(mu, 1e-6)
    return cv


def entrainment_session_plan(initial_cadence: float,
                             target_cadence: float,
                             session_minutes: float = 20.0) -> List[Tuple[EntrainmentPhase, float, float]]:
    total_s = session_minutes * 60
    plan: List[Tuple[EntrainmentPhase, float, float]] = []
    plan.append((EntrainmentPhase.ASSESSMENT, initial_cadence, total_s * 0.1))
    plan.append((EntrainmentPhase.MATCHING, initial_cadence, total_s * 0.15))
    ramp = iso_principle_tempo_ramp(initial_cadence, target_cadence, 5)
    lead_dur = total_s * 0.35 / len(ramp)
    for tempo in ramp:
        plan.append((EntrainmentPhase.LEADING, float(tempo), lead_dur))
    plan.append((EntrainmentPhase.TARGET, target_cadence, total_s * 0.25))
    plan.append((EntrainmentPhase.MAINTENANCE, target_cadence, total_s * 0.15))
    return plan


def progress_metric(sync_ratios_per_session: List[float]) -> Dict[str, float]:
    if len(sync_ratios_per_session) < 2:
        return {"slope": 0.0, "latest": sync_ratios_per_session[-1] if sync_ratios_per_session else 0.0}
    x = np.arange(len(sync_ratios_per_session), dtype=float)
    y = np.array(sync_ratios_per_session)
    slope = float(np.polyfit(x, y, 1)[0])
    return {
        "slope": slope,
        "latest": float(y[-1]),
        "mean": float(np.mean(y)),
        "best": float(np.max(y)),
        "improving": slope > 0.01,
    }
