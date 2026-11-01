"""Rhythm pattern generation: Euclidean rhythms, polyrhythms, swing quantization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RhythmPattern:
    onsets: list[int]
    steps: int
    name: str = ""

    @property
    def density(self) -> float:
        return len(self.onsets) / self.steps if self.steps > 0 else 0.0

    def to_binary(self) -> list[int]:
        b = [0] * self.steps
        for o in self.onsets:
            if 0 <= o < self.steps:
                b[o] = 1
        return b

    def rotate(self, shift: int) -> RhythmPattern:
        return RhythmPattern(
            onsets=[(o + shift) % self.steps for o in self.onsets],
            steps=self.steps,
            name=self.name,
        )


def euclidean_rhythm(pulses: int, steps: int) -> RhythmPattern:
    if pulses <= 0 or steps <= 0:
        return RhythmPattern(onsets=[], steps=max(steps, 0))
    pulses = min(pulses, steps)
    pattern: list[int] = []
    bucket = 0
    for i in range(steps):
        bucket += pulses
        if bucket >= steps:
            bucket -= steps
            pattern.append(i)
    return RhythmPattern(onsets=pattern, steps=steps, name=f"E({pulses},{steps})")


def polyrhythm(a: int, b: int, steps: int) -> tuple[RhythmPattern, RhythmPattern]:
    pa = euclidean_rhythm(a, steps)
    pb = euclidean_rhythm(b, steps)
    return pa, pb


def swing_quantize(
    onset_times: np.ndarray,
    grid_interval: float,
    swing_ratio: float = 0.67,
) -> np.ndarray:
    if len(onset_times) == 0:
        return onset_times.copy()
    quantized = np.empty_like(onset_times)
    for i, t in enumerate(onset_times):
        beat_pos = t / grid_interval
        beat_idx = int(np.floor(beat_pos))
        frac = beat_pos - beat_idx
        if frac < 0.5:
            grid_t = beat_idx * grid_interval
        else:
            grid_t = (beat_idx + swing_ratio) * grid_interval
        quantized[i] = grid_t
    return quantized


def syncopation_score(pattern: RhythmPattern) -> float:
    if pattern.steps == 0:
        return 0.0
    binary = pattern.to_binary()
    n = len(binary)
    score = 0.0
    for i, val in enumerate(binary):
        if val == 0:
            continue
        # Metric weight: powers of 2 subdivisions
        weight = 0
        pos = i
        div = n
        while div >= 1:
            if pos % div == 0:
                weight = div
                break
            div //= 2
        if weight == 0:
            score += 1.0
        else:
            score += 1.0 / weight
    return score / max(len(pattern.onsets), 1)


def onset_to_ioi(onsets: Sequence[float]) -> list[float]:
    return [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]


def ioi_regularity(ioi_list: Sequence[float]) -> float:
    if len(ioi_list) < 2:
        return 1.0
    arr = np.array(ioi_list)
    mean = np.mean(arr)
    if mean < 1e-12:
        return 1.0
    cv = float(np.std(arr) / mean)
    return float(np.exp(-cv))


@dataclass(frozen=True)
class PolymetricPattern:
    layers: list[RhythmPattern]
    total_steps: int

    def combined_onsets(self) -> list[int]:
        s: set[int] = set()
        for layer in self.layers:
            s.update(layer.onsets)
        return sorted(s)

    @property
    def composite_density(self) -> float:
        return len(self.combined_onsets()) / self.total_steps if self.total_steps > 0 else 0.0


def create_polymetric(meters: Sequence[int], total_steps: int) -> PolymetricPattern:
    layers = [euclidean_rhythm(m, total_steps) for m in meters]
    return PolymetricPattern(layers=layers, total_steps=total_steps)


def groove_template(
    steps: int = 16,
    accents: Sequence[int] | None = None,
    ghost_notes: Sequence[int] | None = None,
) -> np.ndarray:
    velocities = np.full(steps, 0.5)
    if accents:
        for a in accents:
            if 0 <= a < steps:
                velocities[a] = 1.0
    if ghost_notes:
        for g in ghost_notes:
            if 0 <= g < steps:
                velocities[g] = 0.25
    return velocities


def humanize(
    onset_times: np.ndarray,
    timing_std: float = 0.005,
    velocity: np.ndarray | None = None,
    velocity_std: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if rng is None:
        rng = np.random.default_rng()
    times = onset_times + rng.normal(0, timing_std, len(onset_times))
    times = np.maximum(times, 0.0)
    vel_out = None
    if velocity is not None:
        vel_out = np.clip(velocity + rng.normal(0, velocity_std, len(velocity)), 0.0, 1.0)
    return times, vel_out
