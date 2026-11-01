"""Section transitions in arrangement: energy management, transition
types, density curves, frequency spectrum across transitions.

Based on Huron (2006) expectation theory and
Bregman (1990) auditory scene analysis. Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class TransitionType(Enum):
    BUILD = "build"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    FILL = "fill"
    CROSSFADE = "crossfade"
    RISER = "riser"
    IMPACT = "impact"


@dataclass
class SectionProfile:
    energy: float
    density: float
    spectral_centroid: float
    n_voices: int = 4
    label: str = ""


@dataclass
class TransitionParams:
    transition_type: TransitionType = TransitionType.BUILD
    duration_beats: float = 4.0
    curvature: float = 1.0
    intensity: float = 1.0


@dataclass
class TransitionPlan:
    from_section: SectionProfile
    to_section: SectionProfile
    params: TransitionParams
    energy_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    density_curve: np.ndarray = field(default_factory=lambda: np.array([]))


def energy_curve(start: float, end: float, n_steps: int,
                 curvature: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 1, n_steps)
    if curvature == 1.0:
        return start + (end - start) * t
    return start + (end - start) * t ** curvature


def density_ramp(start_voices: int, end_voices: int,
                 n_steps: int) -> np.ndarray:
    return np.linspace(start_voices, end_voices, n_steps)


def spectral_fill(low_centroid: float, high_centroid: float,
                  n_steps: int, curve: str = "linear") -> np.ndarray:
    t = np.linspace(0, 1, n_steps)
    if curve == "exponential":
        return low_centroid * (high_centroid / (low_centroid + 1e-10)) ** t
    return low_centroid + (high_centroid - low_centroid) * t


def build_riser(n_samples: int, sr: float,
                start_freq: float = 200.0,
                end_freq: float = 8000.0,
                noise_amount: float = 0.3) -> np.ndarray:
    t = np.arange(n_samples) / sr
    freq = start_freq * (end_freq / start_freq) ** (np.arange(n_samples) / n_samples)
    phase = np.cumsum(2 * np.pi * freq / sr)
    sweep = np.sin(phase)
    noise = np.random.RandomState(42).randn(n_samples) * noise_amount
    amplitude = np.linspace(0.3, 1.0, n_samples)
    return (sweep + noise) * amplitude


def drop_impact(n_samples: int, sr: float,
                impact_freq: float = 60.0,
                decay_ms: float = 500.0) -> np.ndarray:
    t = np.arange(n_samples) / sr
    decay = np.exp(-t / (decay_ms / 1000.0))
    sine = np.sin(2 * np.pi * impact_freq * t)
    noise_burst = np.zeros(n_samples)
    burst_len = min(int(sr * 0.02), n_samples)
    noise_burst[:burst_len] = np.random.RandomState(42).randn(burst_len) * 0.5
    return (sine * decay + noise_burst * np.exp(-t / 0.01))


def breakdown_strip(full_energy: np.ndarray,
                    keep_fraction: float = 0.25,
                    fade_samples: int = 1024) -> np.ndarray:
    result = full_energy * keep_fraction
    if fade_samples > 0 and len(result) > fade_samples:
        fade = np.linspace(1.0, keep_fraction, fade_samples)
        result[:fade_samples] = full_energy[:fade_samples] * fade
    return result


def compute_transition(from_sec: SectionProfile,
                       to_sec: SectionProfile,
                       params: TransitionParams,
                       n_steps: int = 64) -> TransitionPlan:
    e_curve = energy_curve(from_sec.energy, to_sec.energy, n_steps,
                           params.curvature)
    d_curve = density_ramp(from_sec.n_voices, to_sec.n_voices, n_steps)
    return TransitionPlan(from_sec, to_sec, params, e_curve, d_curve)


def transition_smoothness(curve: np.ndarray) -> float:
    if len(curve) < 3:
        return 1.0
    second_deriv = np.diff(curve, n=2)
    return float(1.0 / (1.0 + np.mean(np.abs(second_deriv))))


def section_contrast(a: SectionProfile, b: SectionProfile) -> float:
    energy_diff = abs(a.energy - b.energy)
    density_diff = abs(a.density - b.density) / (max(a.density, b.density) + 1e-10)
    spectral_diff = abs(a.spectral_centroid - b.spectral_centroid) / (
        max(a.spectral_centroid, b.spectral_centroid) + 1e-10)
    return float((energy_diff + density_diff + spectral_diff) / 3.0)


def plan_arrangement_transitions(
        sections: List[SectionProfile],
        default_duration: float = 4.0) -> List[TransitionPlan]:
    plans = []
    for i in range(len(sections) - 1):
        contrast = section_contrast(sections[i], sections[i + 1])
        if sections[i + 1].energy > sections[i].energy:
            tt = TransitionType.BUILD
            curvature = 1.5
        elif sections[i + 1].energy < sections[i].energy * 0.5:
            tt = TransitionType.DROP
            curvature = 0.5
        else:
            tt = TransitionType.CROSSFADE
            curvature = 1.0
        params = TransitionParams(tt, default_duration, curvature,
                                  contrast)
        plan = compute_transition(sections[i], sections[i + 1], params)
        plans.append(plan)
    return plans


def evaluate_transition_quality(plan: TransitionPlan) -> float:
    smoothness = transition_smoothness(plan.energy_curve)
    contrast = section_contrast(plan.from_section, plan.to_section)
    if len(plan.energy_curve) < 2:
        return 0.5
    monotonic = float(np.all(np.diff(plan.energy_curve) >= -1e-6) or
                       np.all(np.diff(plan.energy_curve) <= 1e-6))
    return float(0.4 * smoothness + 0.3 * contrast + 0.3 * monotonic)


def crossfade_curve(n_samples: int,
                    curve_type: str = "equal_power") -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, n_samples)
    if curve_type == "equal_power":
        fade_out = np.cos(t * np.pi / 2)
        fade_in = np.sin(t * np.pi / 2)
    elif curve_type == "logarithmic":
        fade_out = 1.0 - np.log1p(t * (np.e - 1)) / np.log(np.e)
        fade_in = np.log1p(t * (np.e - 1)) / np.log(np.e)
    else:
        fade_out = 1.0 - t
        fade_in = t
    return fade_out, fade_in
