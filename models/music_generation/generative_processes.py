"""Generative stochastic processes for music: Ornstein-Uhlenbeck, logistic
maps, iterated function systems, self-similar rhythms, Brownian bridges.

Based on Xenakis (1992) formalized music and Roads (2001) microsound.
Complements algorithmic_composition.py with continuous-valued processes.
Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class OUParams:
    mu: float = 60.0
    theta: float = 0.15
    sigma: float = 3.0
    dt: float = 1.0


@dataclass
class LogisticParams:
    r: float = 3.9
    x0: float = 0.1
    pitch_low: float = 48.0
    pitch_high: float = 84.0


@dataclass
class IFSRule:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    prob: float = 1.0


def ornstein_uhlenbeck(n_steps: int, params: OUParams,
                       seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = np.zeros(n_steps)
    x[0] = params.mu
    noise = rng.randn(n_steps) * params.sigma * np.sqrt(params.dt)
    for i in range(1, n_steps):
        x[i] = x[i - 1] + params.theta * (params.mu - x[i - 1]) * params.dt + noise[i]
    return x


def logistic_map(n_steps: int, params: LogisticParams) -> np.ndarray:
    x = np.zeros(n_steps)
    x[0] = params.x0
    for i in range(1, n_steps):
        x[i] = params.r * x[i - 1] * (1.0 - x[i - 1])
    return params.pitch_low + x * (params.pitch_high - params.pitch_low)


def henon_map(n_steps: int, a: float = 1.4, b: float = 0.3,
              x0: float = 0.0, y0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    xs[0], ys[0] = x0, y0
    for i in range(1, n_steps):
        xs[i] = 1.0 - a * xs[i - 1] ** 2 + ys[i - 1]
        ys[i] = b * xs[i - 1]
    return xs, ys


def brownian_bridge(n_steps: int, start: float = 60.0, end: float = 72.0,
                    sigma: float = 3.0, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, n_steps)
    w = np.zeros(n_steps)
    dt = 1.0 / max(n_steps - 1, 1)
    for i in range(1, n_steps):
        w[i] = w[i - 1] + rng.randn() * np.sqrt(dt)
    bridge = w - t * w[-1] if n_steps > 1 else w
    return start + (end - start) * t + sigma * bridge


def ifs_attractor(rules: List[IFSRule], n_points: int = 10000,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    probs = np.array([r.prob for r in rules])
    probs /= probs.sum()
    xs = np.zeros(n_points)
    ys = np.zeros(n_points)
    x, y = 0.0, 0.0
    for i in range(n_points):
        idx = rng.choice(len(rules), p=probs)
        r = rules[idx]
        x_new = r.a * x + r.b * y + r.e
        y_new = r.c * x + r.d * y + r.f
        x, y = x_new, y_new
        xs[i] = x
        ys[i] = y
    return xs, ys


def ifs_to_pitches(xs: np.ndarray, ys: np.ndarray,
                   pitch_range: Tuple[float, float] = (48.0, 84.0),
                   velocity_range: Tuple[float, float] = (40.0, 120.0)
                   ) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_norm = (xs - x_min) / (x_max - x_min + 1e-10)
    y_norm = (ys - y_min) / (y_max - y_min + 1e-10)
    pitches = pitch_range[0] + x_norm * (pitch_range[1] - pitch_range[0])
    velocities = velocity_range[0] + y_norm * (velocity_range[1] - velocity_range[0])
    return pitches, velocities


def self_similar_rhythm(n_beats: int, depth: int = 3,
                        pattern: Optional[np.ndarray] = None) -> np.ndarray:
    if pattern is None:
        pattern = np.array([1.0, 0.0, 1.0])
    grid = np.array([1.0])
    for _ in range(depth):
        new_grid = np.zeros(len(grid) * len(pattern))
        for i, val in enumerate(grid):
            if val > 0:
                start = i * len(pattern)
                new_grid[start:start + len(pattern)] = pattern * val
        grid = new_grid
    if len(grid) >= n_beats:
        return grid[:n_beats]
    repeats = int(np.ceil(n_beats / len(grid)))
    return np.tile(grid, repeats)[:n_beats]


def cantor_rhythm(n_beats: int, depth: int = 4) -> np.ndarray:
    pattern = np.array([1.0, 1.0, 0.0])
    return self_similar_rhythm(n_beats, depth, pattern)


def pink_noise_melody(n_steps: int,
                      pitch_range: Tuple[float, float] = (48.0, 84.0),
                      n_octaves: int = 8, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    values = np.zeros(n_steps)
    octave_values = rng.randn(n_octaves)
    max_key = n_octaves
    for i in range(n_steps):
        last_key = 0
        n = i
        while n > 0 and last_key < max_key:
            if n & 1:
                octave_values[last_key] = rng.randn()
            n >>= 1
            last_key += 1
        values[i] = octave_values.sum()
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 1e-10:
        return np.full(n_steps, (pitch_range[0] + pitch_range[1]) / 2)
    normalized = (values - v_min) / (v_max - v_min)
    return pitch_range[0] + normalized * (pitch_range[1] - pitch_range[0])


def poisson_rhythm(n_beats: int, rate: float = 0.3,
                   seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.random(n_beats) < rate).astype(np.float64)


def markov_rhythm(n_steps: int, transition: np.ndarray,
                  initial_state: int = 0,
                  seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n_states = transition.shape[0]
    states = np.zeros(n_steps, dtype=int)
    states[0] = initial_state
    for i in range(1, n_steps):
        probs = transition[states[i - 1]]
        probs = probs / (probs.sum() + 1e-10)
        states[i] = rng.choice(n_states, p=probs)
    return states


def tendency_mask(n_steps: int, low_curve: np.ndarray,
                  high_curve: np.ndarray,
                  seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, n_steps)
    low_interp = np.interp(t, np.linspace(0, 1, len(low_curve)), low_curve)
    high_interp = np.interp(t, np.linspace(0, 1, len(high_curve)), high_curve)
    uniform = rng.random(n_steps)
    return low_interp + uniform * (high_interp - low_interp)
