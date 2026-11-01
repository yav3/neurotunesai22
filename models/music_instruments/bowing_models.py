"""Bowing and plucking models for string instruments.

Physical-modelling parameters for detaché, spiccato, tremolo, pizzicato,
col legno, harmonics, and other articulations. Computes excitation signals
and envelope shapes — pure numpy, no audio I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple

import numpy as np


class BowStroke(Enum):
    DETACHE = "detache"
    LEGATO = "legato"
    STACCATO = "staccato"
    SPICCATO = "spiccato"
    MARTELE = "martele"
    TREMOLO = "tremolo"
    COL_LEGNO = "col_legno"
    SUL_TASTO = "sul_tasto"
    SUL_PONTICELLO = "sul_ponticello"
    PIZZICATO = "pizzicato"
    HARMONIC = "harmonic"


@dataclass
class BowParams:
    force: float = 0.5
    velocity: float = 0.3
    position: float = 0.12
    tilt: float = 0.0
    hair_width: float = 0.008


@dataclass
class StringParams:
    fundamental: float = 440.0
    length: float = 0.32
    tension: float = 70.0
    linear_density: float = 0.001
    damping: float = 0.002


def bow_force_envelope(
    stroke: BowStroke,
    n_samples: int,
    sr: float = 44100.0,
) -> np.ndarray:
    t = np.linspace(0, 1, n_samples)
    if stroke == BowStroke.DETACHE:
        return 0.5 + 0.1 * np.sin(np.pi * t)
    if stroke == BowStroke.LEGATO:
        return np.ones(n_samples) * 0.4
    if stroke == BowStroke.STACCATO:
        env = np.exp(-5.0 * t)
        return env * 0.7
    if stroke == BowStroke.SPICCATO:
        env = np.exp(-8.0 * t) * np.sin(np.pi * t * 2)
        return np.maximum(env, 0) * 0.8
    if stroke == BowStroke.MARTELE:
        attack = np.minimum(t * 20, 1.0)
        decay = np.exp(-3.0 * np.maximum(t - 0.05, 0))
        return attack * decay * 0.9
    if stroke == BowStroke.TREMOLO:
        rate = 12.0
        return 0.3 + 0.3 * np.abs(np.sin(2 * np.pi * rate * t))
    if stroke == BowStroke.COL_LEGNO:
        return np.exp(-15.0 * t) * 0.6
    if stroke == BowStroke.SUL_TASTO:
        return np.ones(n_samples) * 0.25
    if stroke == BowStroke.SUL_PONTICELLO:
        return np.ones(n_samples) * 0.6
    if stroke == BowStroke.PIZZICATO:
        return np.exp(-20.0 * t) * 1.0
    if stroke == BowStroke.HARMONIC:
        return np.ones(n_samples) * 0.15
    return np.ones(n_samples) * 0.5


def spectral_brightness(position: float) -> float:
    if position <= 0 or position >= 1:
        return 0.5
    return 1.0 / position


def harmonic_weighting(
    n_harmonics: int,
    position: float,
    stroke: BowStroke = BowStroke.DETACHE,
) -> np.ndarray:
    harmonics = np.arange(1, n_harmonics + 1, dtype=float)
    weights = np.sin(np.pi * harmonics * position) / harmonics
    weights = np.abs(weights)
    if stroke == BowStroke.SUL_PONTICELLO:
        weights *= harmonics**0.3
    elif stroke == BowStroke.SUL_TASTO:
        weights *= np.exp(-0.3 * harmonics)
    elif stroke == BowStroke.HARMONIC:
        weights *= np.exp(-1.0 * (harmonics - 1))
    elif stroke == BowStroke.COL_LEGNO:
        noise = np.random.default_rng(42).uniform(0.5, 1.5, n_harmonics)
        weights *= noise * np.exp(-0.1 * harmonics)
    total = weights.sum()
    return weights / total if total > 0 else weights


def friction_model(
    bow_velocity: float,
    string_velocity: float,
    force: float,
    mu_s: float = 0.8,
    mu_d: float = 0.3,
    v_s: float = 0.1,
) -> float:
    dv = bow_velocity - string_velocity
    if abs(dv) < 1e-12:
        return mu_s * force * np.sign(dv + 1e-15)
    mu = mu_d + (mu_s - mu_d) * np.exp(-abs(dv) / v_s)
    return mu * force * np.sign(dv)


def pizzicato_excitation(
    n_samples: int,
    pluck_position: float = 0.15,
    pluck_width: float = 0.02,
    sr: float = 44100.0,
) -> np.ndarray:
    x = np.linspace(0, 1, n_samples)
    displacement = np.where(
        np.abs(x - pluck_position) < pluck_width,
        1.0 - np.abs(x - pluck_position) / pluck_width,
        0.0,
    )
    return displacement


def string_decay_time(
    frequency: float,
    damping: float = 0.002,
    tension: float = 70.0,
    length: float = 0.32,
) -> float:
    omega = 2 * np.pi * frequency
    tau = 2.0 / (damping * omega)
    return tau


def natural_harmonic_nodes(
    harmonic_number: int,
    string_length: float = 1.0,
) -> List[float]:
    return [k * string_length / harmonic_number for k in range(1, harmonic_number)]


def synthesize_bowed_tone(
    params: StringParams,
    bow: BowParams,
    stroke: BowStroke,
    duration_s: float = 1.0,
    sr: float = 44100.0,
    n_harmonics: int = 20,
) -> np.ndarray:
    n_samples = int(duration_s * sr)
    t = np.arange(n_samples) / sr

    env = bow_force_envelope(stroke, n_samples, sr)
    weights = harmonic_weighting(n_harmonics, bow.position, stroke)

    signal = np.zeros(n_samples)
    for h in range(n_harmonics):
        freq = params.fundamental * (h + 1)
        decay = np.exp(-params.damping * 2 * np.pi * freq * t)
        signal += weights[h] * np.sin(2 * np.pi * freq * t) * decay

    signal *= env * bow.force
    peak = np.abs(signal).max()
    if peak > 0:
        signal /= peak
    return signal


def vibrato_modulation(
    n_samples: int,
    rate: float = 5.5,
    depth_cents: float = 30.0,
    sr: float = 44100.0,
) -> np.ndarray:
    t = np.arange(n_samples) / sr
    depth_ratio = 2.0 ** (depth_cents / 1200.0) - 1.0
    return depth_ratio * np.sin(2 * np.pi * rate * t)


def bow_noise(
    n_samples: int,
    amount: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, amount, n_samples)
    b = np.array([1.0, -0.97])
    a = np.array([1.0])
    from scipy.signal import lfilter
    return lfilter(b, a, noise)
