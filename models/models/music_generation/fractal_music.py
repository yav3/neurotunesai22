"""Fractal music generation: fBm, chaos maps, Euclidean rhythms, self-similarity.

Generates musical material from fractal processes: fractional Brownian motion
for melody contour (Voss & Clarke 1978), logistic/Lorenz chaotic maps for
deterministic yet unpredictable sequences, Cantor-set rhythms, and Bjorklund's
Euclidean rhythm algorithm.
"""
from __future__ import annotations

import numpy as np


class FBMGenerator:
    """Fractional Brownian motion melody contour via spectral synthesis."""

    def generate(self, n_samples: int, hurst: float = 0.7,
                 rng: np.random.RandomState | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.RandomState()
        hurst = max(0.01, min(hurst, 0.99))
        freqs = np.fft.rfftfreq(n_samples)[1:]
        power = freqs ** (-(2 * hurst + 1))
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        amplitudes = np.sqrt(power)
        spectrum = np.zeros(n_samples // 2 + 1, dtype=complex)
        spectrum[1:] = amplitudes * np.exp(1j * phases)
        signal = np.fft.irfft(spectrum, n=n_samples)
        return signal / (np.std(signal) + 1e-12)

    def to_melody(self, fbm: np.ndarray, pitch_range: tuple[int, int] = (48, 84),
                  scale: list[int] | None = None) -> list[int]:
        low, high = pitch_range
        normalized = (fbm - fbm.min()) / (fbm.max() - fbm.min() + 1e-12)
        continuous = low + normalized * (high - low)
        pitches = np.round(continuous).astype(int)
        if scale is not None:
            scale_set = sorted(set(scale))
            quantized = []
            for p in pitches:
                pc = p % 12
                best = min(scale_set, key=lambda s: abs(s % 12 - pc))
                octave = p // 12
                quantized.append(octave * 12 + best % 12)
            return quantized
        return pitches.tolist()


class ChaosGenerator:
    """Deterministic chaotic maps for music generation."""

    @staticmethod
    def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
        x = np.zeros(n)
        x[0] = x0
        for i in range(1, n):
            x[i] = r * x[i - 1] * (1 - x[i - 1])
        return x

    @staticmethod
    def lorenz(sigma: float = 10.0, rho: float = 28.0, beta: float = 8 / 3,
               dt: float = 0.01, n: int = 5000,
               x0: float = 1.0, y0: float = 1.0, z0: float = 1.0) -> np.ndarray:
        trajectory = np.zeros((n, 3))
        trajectory[0] = [x0, y0, z0]
        for i in range(1, n):
            x, y, z = trajectory[i - 1]
            trajectory[i, 0] = x + dt * sigma * (y - x)
            trajectory[i, 1] = y + dt * (x * (rho - z) - y)
            trajectory[i, 2] = z + dt * (x * y - beta * z)
        return trajectory

    @staticmethod
    def henon_map(a: float = 1.4, b: float = 0.3, n: int = 1000) -> np.ndarray:
        xy = np.zeros((n, 2))
        xy[0] = [0.1, 0.1]
        for i in range(1, n):
            x, y = xy[i - 1]
            xy[i, 0] = 1 - a * x * x + y
            xy[i, 1] = b * x
        return xy

    def map_to_music(self, trajectory: np.ndarray,
                     pitch_range: tuple[int, int] = (48, 84),
                     velocity_range: tuple[int, int] = (40, 120)
                     ) -> list[tuple[int, int, float]]:
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
        n_dim = trajectory.shape[1]
        # pitch from first dimension
        col0 = trajectory[:, 0]
        norm0 = (col0 - col0.min()) / (col0.max() - col0.min() + 1e-12)
        pitches = (pitch_range[0] + norm0 * (pitch_range[1] - pitch_range[0])).astype(int)
        # velocity from second dimension or constant
        if n_dim >= 2:
            col1 = trajectory[:, 1]
            norm1 = (col1 - col1.min()) / (col1.max() - col1.min() + 1e-12)
            velocities = (velocity_range[0] + norm1 * (velocity_range[1] - velocity_range[0])).astype(int)
        else:
            velocities = np.full(len(trajectory), 80)
        # duration from third dimension or constant
        if n_dim >= 3:
            col2 = trajectory[:, 2]
            norm2 = (col2 - col2.min()) / (col2.max() - col2.min() + 1e-12)
            durations = 0.125 + norm2 * 1.875
        else:
            durations = np.full(len(trajectory), 0.5)
        return [(int(p), int(v), float(d)) for p, v, d in zip(pitches, velocities, durations)]


class CantorRhythm:
    """Cantor set and Cantor dust rhythm generators."""

    @staticmethod
    def cantor_set(depth: int) -> list[int]:
        pattern = [1]
        for _ in range(depth):
            new = []
            for bit in pattern:
                if bit == 1:
                    new.extend([1, 0, 1])
                else:
                    new.extend([0, 0, 0])
            pattern = new
        return pattern

    @staticmethod
    def cantor_dust_2d(depth: int) -> np.ndarray:
        grid = np.ones((1, 1), dtype=np.int8)
        for _ in range(depth):
            h, w = grid.shape
            new = np.zeros((h * 3, w * 3), dtype=np.int8)
            for y in range(h):
                for x in range(w):
                    if grid[y, x] == 1:
                        for dy in [0, 2]:
                            for dx in [0, 2]:
                                new[y * 3 + dy, x * 3 + dx] = 1
            grid = new
        return grid


def euclidean_rhythm(pulses: int, steps: int) -> list[int]:
    """Bjorklund's algorithm for maximally even distribution of pulses."""
    if pulses >= steps:
        return [1] * steps
    if pulses <= 0:
        return [0] * steps
    groups: list[list[int]] = [[1] for _ in range(pulses)] + [[0] for _ in range(steps - pulses)]
    while True:
        remainder = len(groups) - pulses
        if remainder <= 1:
            break
        n_merge = min(pulses, remainder)
        new_groups = []
        for i in range(n_merge):
            new_groups.append(groups[i] + groups[pulses + i])
        for i in range(n_merge, pulses):
            new_groups.append(groups[i])
        for i in range(pulses + n_merge, len(groups)):
            new_groups.append(groups[i])
        groups = new_groups
        pulses = n_merge
    result: list[int] = []
    for g in groups:
        result.extend(g)
    return result


def golden_ratio_rhythm(n_beats: int) -> list[float]:
    """Fibonacci-spaced onset positions using golden ratio."""
    phi = (1 + np.sqrt(5)) / 2
    onsets = [(i * phi) % n_beats for i in range(n_beats)]
    onsets.sort()
    return [float(t) for t in onsets]


class SelfSimilarRhythm:
    """Self-similar rhythmic patterns from substitution rules."""

    def __init__(self, rules: dict[str, str] | None = None) -> None:
        self.rules = rules or {"A": "AB", "B": "A"}

    def generate(self, axiom: str, iterations: int) -> str:
        current = axiom
        for _ in range(iterations):
            current = "".join(self.rules.get(ch, ch) for ch in current)
        return current

    def to_rhythm(self, string: str, duration_map: dict[str, float] | None = None
                  ) -> list[float]:
        if duration_map is None:
            duration_map = {"A": 1.0, "B": 0.5}
        return [duration_map.get(ch, 0.5) for ch in string]


class MultifractalMelody:
    """Multifractal spectrum control over melodic complexity."""

    def generate(self, n: int, alpha_min: float = 0.3, alpha_max: float = 0.8,
                 rng: np.random.RandomState | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.RandomState()
        cascade = np.ones(n)
        levels = int(np.log2(max(n, 2)))
        for level in range(levels):
            block = n // (2 ** (level + 1))
            if block < 1:
                break
            alpha = alpha_min + (alpha_max - alpha_min) * level / max(levels - 1, 1)
            for i in range(0, n, 2 * block):
                w = 0.5 + alpha * (rng.rand() - 0.5)
                cascade[i : i + block] *= w
                cascade[i + block : i + 2 * block] *= (1 - w)
        cumulative = np.cumsum(cascade)
        cumulative -= cumulative.mean()
        return cumulative / (np.std(cumulative) + 1e-12)
