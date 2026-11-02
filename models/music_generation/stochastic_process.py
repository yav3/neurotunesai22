"""Stochastic process composition: controlled randomness in music generation.

Generate musical material via stochastic processes: Ornstein-Uhlenbeck for
mean-reverting pitch/dynamic trajectories, Poisson processes for event timing,
Gaussian random fields for textural density, and entropy-guided exploration
for balancing novelty and coherence.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class StochasticConfig:
    seed: int = 42
    temperature: float = 1.0
    entropy_target: float = 0.7


def ornstein_uhlenbeck(n_steps: int, theta: float = 0.15,
                       mu: float = 60.0, sigma: float = 5.0,
                       x0: float = 60.0, dt: float = 1.0,
                       seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps)
    x[0] = x0
    for i in range(1, n_steps):
        dx = theta * (mu - x[i - 1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        x[i] = x[i - 1] + dx
    return x


def poisson_events(rate: float, duration_s: float,
                   seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_expected = int(rate * duration_s * 2)
    intervals = rng.exponential(1.0 / max(rate, 0.01), n_expected)
    times = np.cumsum(intervals)
    return times[times < duration_s]


def inhomogeneous_poisson(rate_func_values: np.ndarray,
                          dt: float = 0.01,
                          seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    max_rate = np.max(rate_func_values)
    if max_rate <= 0:
        return np.array([])
    duration = len(rate_func_values) * dt
    candidate_times = poisson_events(max_rate, duration, seed)
    accepted = []
    for t in candidate_times:
        idx = min(int(t / dt), len(rate_func_values) - 1)
        if rng.random() < rate_func_values[idx] / max_rate:
            accepted.append(t)
    return np.array(accepted)


def brownian_bridge(n_steps: int, start: float, end: float,
                    sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_steps)
    w = np.cumsum(rng.standard_normal(n_steps)) * sigma / np.sqrt(n_steps)
    bridge = w - t * w[-1]
    return start + (end - start) * t + bridge


def pitch_random_walk(n_steps: int, scale: Tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11),
                      start_pitch: int = 60, step_weights: Optional[np.ndarray] = None,
                      seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pitches = np.zeros(n_steps, dtype=int)
    pitches[0] = start_pitch
    scale_arr = np.array(scale)
    for i in range(1, n_steps):
        current_pc = pitches[i - 1] % 12
        current_octave = pitches[i - 1] // 12
        candidates = []
        for oct in range(current_octave - 1, current_octave + 2):
            for pc in scale_arr:
                p = oct * 12 + pc
                if abs(p - pitches[i - 1]) <= 7:
                    candidates.append(p)
        if not candidates:
            pitches[i] = pitches[i - 1]
            continue
        if step_weights is not None and len(step_weights) == len(candidates):
            probs = step_weights / step_weights.sum()
        else:
            distances = np.abs(np.array(candidates) - pitches[i - 1])
            probs = np.exp(-distances / 3.0)
            probs /= probs.sum()
        pitches[i] = rng.choice(candidates, p=probs)
    return pitches


def gaussian_texture_field(n_time: int, n_freq: int,
                           correlation_time: float = 10.0,
                           correlation_freq: float = 5.0,
                           seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_freq, n_time))
    t_kernel = np.exp(-np.arange(n_time) ** 2 / (2 * correlation_time ** 2))
    f_kernel = np.exp(-np.arange(n_freq) ** 2 / (2 * correlation_freq ** 2))
    from scipy.signal import fftconvolve
    smoothed = noise.copy()
    for f in range(n_freq):
        smoothed[f] = fftconvolve(smoothed[f], t_kernel, mode="same")
    for t in range(n_time):
        smoothed[:, t] = fftconvolve(smoothed[:, t], f_kernel, mode="same")
    return smoothed / max(np.abs(smoothed).max(), 1e-10)


def entropy_of_distribution(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def temperature_scaled_choice(logits: np.ndarray, temperature: float = 1.0,
                               seed: int = 42) -> int:
    rng = np.random.default_rng(seed)
    if temperature < 1e-6:
        return int(np.argmax(logits))
    scaled = logits / temperature
    scaled -= scaled.max()
    probs = np.exp(scaled) / np.sum(np.exp(scaled))
    return int(rng.choice(len(logits), p=probs))


def entropy_guided_melody(n_notes: int, scale: Tuple[int, ...],
                          target_entropy: float = 0.7,
                          seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    max_entropy = np.log2(len(scale))
    target_h = target_entropy * max_entropy
    pitches = np.zeros(n_notes, dtype=int)
    pitches[0] = 60 + scale[rng.integers(len(scale))]
    counts = np.ones(len(scale)) * 0.1
    for i in range(1, n_notes):
        probs = counts / counts.sum()
        current_h = entropy_of_distribution(probs)
        if current_h < target_h:
            temp = 2.0
        else:
            temp = 0.5
        proximity = np.array([1.0 / (1.0 + abs(pitches[i - 1] % 12 - s))
                              for s in scale])
        logits = np.log(proximity + 0.01) + np.log(1.0 / (counts + 0.1))
        idx = temperature_scaled_choice(logits, temp, seed + i)
        octave = pitches[i - 1] // 12
        pitches[i] = octave * 12 + scale[idx]
        counts[idx] += 1
    return pitches


def markov_rhythm(n_events: int, states: np.ndarray,
                  transition: np.ndarray,
                  seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros(n_events)
    current = rng.integers(len(states))
    result[0] = states[current]
    for i in range(1, n_events):
        probs = transition[current]
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(states)) / len(states)
        current = rng.choice(len(states), p=probs)
        result[i] = states[current]
    return result
