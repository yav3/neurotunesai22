"""Rhythmic entrainment for therapeutic applications: auditory-motor coupling, gait training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "EntrainmentState", "EntrainmentSession", "GaitTrainingProtocol",
    "auditory_motor_coupling", "phase_locking_value",
    "entrainment_strength", "tempo_adaptation_sequence",
    "isochronous_sequence", "kuramoto_oscillator",
    "gradual_tempo_shift", "optimal_entrainment_bpm",
    "gait_training_protocol", "bilateral_entrainment_ratio",
]


@dataclass
class EntrainmentState:
    phase: float
    frequency: float
    coupling_strength: float
    phase_coherence: float
    is_entrained: bool


def kuramoto_oscillator(
    natural_frequencies: np.ndarray,
    coupling: float,
    n_steps: int = 1000,
    dt: float = 0.01,
    initial_phases: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = len(natural_frequencies)
    omega = np.asarray(natural_frequencies, dtype=float)
    if initial_phases is not None:
        phases = np.asarray(initial_phases, dtype=float).copy()
    else:
        phases = np.random.default_rng(42).uniform(0, 2 * np.pi, n)
    history = np.zeros((n_steps, n))
    for t in range(n_steps):
        history[t] = phases.copy()
        for i in range(n):
            interaction = np.sum(np.sin(phases - phases[i]))
            dphi = omega[i] + (coupling / n) * interaction
            phases[i] += dphi * dt
    return history


def phase_locking_value(
    phase_a: np.ndarray,
    phase_b: np.ndarray,
) -> float:
    diff = np.asarray(phase_a) - np.asarray(phase_b)
    plv = np.abs(np.mean(np.exp(1j * diff)))
    return float(plv)


def auditory_motor_coupling(
    stimulus_onsets: np.ndarray,
    response_onsets: np.ndarray,
    stimulus_period: float,
) -> EntrainmentState:
    stim = np.asarray(stimulus_onsets, dtype=float)
    resp = np.asarray(response_onsets, dtype=float)
    if len(stim) < 2 or len(resp) < 2:
        return EntrainmentState(0.0, 0.0, 0.0, 0.0, False)
    stim_phases = (stim % stimulus_period) / stimulus_period * 2 * np.pi
    matched_phases = []
    for r in resp:
        idx = np.argmin(np.abs(stim - r))
        async_val = r - stim[idx]
        phase = (async_val / stimulus_period) * 2 * np.pi
        matched_phases.append(phase)
    matched_phases = np.array(matched_phases)
    mean_phase = np.angle(np.mean(np.exp(1j * matched_phases)))
    coherence = float(np.abs(np.mean(np.exp(1j * matched_phases))))
    resp_ioi = np.diff(resp)
    resp_freq = 1.0 / np.mean(resp_ioi) if len(resp_ioi) > 0 and np.mean(resp_ioi) > 0 else 0.0
    coupling = coherence
    return EntrainmentState(
        phase=float(mean_phase),
        frequency=float(resp_freq),
        coupling_strength=coupling,
        phase_coherence=coherence,
        is_entrained=coherence > 0.7,
    )


def entrainment_strength(
    tap_times: np.ndarray,
    stimulus_period: float,
) -> float:
    taps = np.asarray(tap_times, dtype=float)
    if len(taps) < 2:
        return 0.0
    phases = (taps % stimulus_period) / stimulus_period * 2 * np.pi
    r = float(np.abs(np.mean(np.exp(1j * phases))))
    return r


def isochronous_sequence(
    bpm: float,
    n_beats: int,
    start_time: float = 0.0,
) -> np.ndarray:
    period = 60.0 / bpm
    return start_time + np.arange(n_beats) * period


def gradual_tempo_shift(
    start_bpm: float,
    target_bpm: float,
    n_beats: int,
    curve: str = "linear",
) -> np.ndarray:
    if curve == "exponential":
        ratio = target_bpm / max(start_bpm, 1e-9)
        bpms = start_bpm * np.power(ratio, np.linspace(0, 1, n_beats))
    elif curve == "sigmoid":
        t = np.linspace(-6, 6, n_beats)
        sigmoid = 1.0 / (1.0 + np.exp(-t))
        bpms = start_bpm + (target_bpm - start_bpm) * sigmoid
    else:
        bpms = np.linspace(start_bpm, target_bpm, n_beats)
    periods = 60.0 / bpms
    onsets = np.cumsum(periods) - periods[0]
    return onsets


def tempo_adaptation_sequence(
    patient_natural_bpm: float,
    target_bpm: float,
    session_duration_sec: float = 300.0,
    max_shift_per_minute: float = 5.0,
) -> np.ndarray:
    bpm_diff = target_bpm - patient_natural_bpm
    minutes_needed = abs(bpm_diff) / max(max_shift_per_minute, 0.1)
    adaptation_sec = min(minutes_needed * 60, session_duration_sec * 0.7)
    n_adaptation_beats = int(adaptation_sec * (patient_natural_bpm + target_bpm) / 2 / 60)
    n_adaptation_beats = max(n_adaptation_beats, 4)
    adaptation_onsets = gradual_tempo_shift(
        patient_natural_bpm, target_bpm, n_adaptation_beats, curve="sigmoid"
    )
    remaining_sec = session_duration_sec - (adaptation_onsets[-1] if len(adaptation_onsets) > 0 else 0)
    n_steady = max(1, int(remaining_sec * target_bpm / 60))
    steady_onsets = isochronous_sequence(target_bpm, n_steady, adaptation_onsets[-1] + 60.0 / target_bpm)
    return np.concatenate([adaptation_onsets, steady_onsets])


def optimal_entrainment_bpm(
    patient_natural_bpm: float,
    target_activity_bpm: float,
    adaptation_weight: float = 0.7,
) -> float:
    return adaptation_weight * target_activity_bpm + (1 - adaptation_weight) * patient_natural_bpm


@dataclass
class GaitTrainingProtocol:
    patient_cadence: float
    target_cadence: float
    warmup_beats: int
    adaptation_beats: int
    steady_beats: int
    total_onsets: np.ndarray


def gait_training_protocol(
    patient_cadence_bpm: float,
    target_cadence_bpm: float,
    warmup_sec: float = 30.0,
    session_sec: float = 300.0,
) -> GaitTrainingProtocol:
    n_warmup = max(4, int(warmup_sec * patient_cadence_bpm / 60))
    warmup = isochronous_sequence(patient_cadence_bpm, n_warmup)
    remaining = session_sec - warmup_sec
    adaptation_time = remaining * 0.4
    n_adapt = max(4, int(adaptation_time * (patient_cadence_bpm + target_cadence_bpm) / 2 / 60))
    adapt = gradual_tempo_shift(
        patient_cadence_bpm, target_cadence_bpm, n_adapt, curve="sigmoid"
    )
    adapt += warmup[-1] + 60.0 / patient_cadence_bpm
    steady_time = remaining * 0.6
    n_steady = max(4, int(steady_time * target_cadence_bpm / 60))
    steady = isochronous_sequence(
        target_cadence_bpm, n_steady, adapt[-1] + 60.0 / target_cadence_bpm
    )
    total = np.concatenate([warmup, adapt, steady])
    return GaitTrainingProtocol(
        patient_cadence=patient_cadence_bpm,
        target_cadence=target_cadence_bpm,
        warmup_beats=n_warmup,
        adaptation_beats=n_adapt,
        steady_beats=n_steady,
        total_onsets=total,
    )


def bilateral_entrainment_ratio(
    left_taps: np.ndarray,
    right_taps: np.ndarray,
    period: float,
) -> float:
    left_str = entrainment_strength(left_taps, period)
    right_str = entrainment_strength(right_taps, period)
    if max(left_str, right_str) < 1e-9:
        return 1.0
    return float(min(left_str, right_str) / max(left_str, right_str))


@dataclass
class EntrainmentSession:
    protocol: GaitTrainingProtocol
    tap_times: np.ndarray
    phase_coherences: np.ndarray
    is_entrained_timeline: np.ndarray

    @property
    def mean_coherence(self) -> float:
        return float(self.phase_coherences.mean()) if len(self.phase_coherences) > 0 else 0.0

    @property
    def entrainment_ratio(self) -> float:
        if len(self.is_entrained_timeline) == 0:
            return 0.0
        return float(self.is_entrained_timeline.mean())
