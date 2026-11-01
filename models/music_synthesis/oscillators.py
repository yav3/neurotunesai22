"""Oscillators: sine, sawtooth, square, triangle, noise, PWM."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class NoiseType(Enum):
    WHITE = "white"
    PINK = "pink"


@dataclass
class Oscillator:
    frequency: float = 440.0
    sample_rate: int = 44100
    phase: float = 0.0

    def _time_array(self, duration_s: float) -> np.ndarray:
        n = int(self.sample_rate * duration_s)
        return np.arange(n, dtype=np.float64) / self.sample_rate

    def generate(self, duration_s: float) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SineOscillator(Oscillator):
    def generate(self, duration_s: float) -> np.ndarray:
        t = self._time_array(duration_s)
        return np.sin(2.0 * np.pi * self.frequency * t + self.phase).astype(np.float32)


@dataclass
class SawtoothOscillator(Oscillator):
    def generate(self, duration_s: float) -> np.ndarray:
        t = self._time_array(duration_s)
        phase_acc = (self.frequency * t + self.phase / (2.0 * np.pi)) % 1.0
        return (2.0 * phase_acc - 1.0).astype(np.float32)


@dataclass
class SquareOscillator(Oscillator):
    def generate(self, duration_s: float) -> np.ndarray:
        t = self._time_array(duration_s)
        phase_acc = (self.frequency * t + self.phase / (2.0 * np.pi)) % 1.0
        out = np.where(phase_acc < 0.5, 1.0, -1.0)
        return out.astype(np.float32)


@dataclass
class TriangleOscillator(Oscillator):
    def generate(self, duration_s: float) -> np.ndarray:
        t = self._time_array(duration_s)
        phase_acc = (self.frequency * t + self.phase / (2.0 * np.pi)) % 1.0
        out = 2.0 * np.abs(2.0 * phase_acc - 1.0) - 1.0
        return out.astype(np.float32)


@dataclass
class NoiseGenerator:
    noise_type: NoiseType = NoiseType.WHITE
    sample_rate: int = 44100

    def generate(self, duration_s: float, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(self.sample_rate * duration_s)
        white = rng.standard_normal(n)
        if self.noise_type == NoiseType.WHITE:
            return white.astype(np.float32)
        return self._voss_mccartney_pink(white, rng).astype(np.float32)

    @staticmethod
    def _voss_mccartney_pink(white: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(white)
        n_rows = 16
        rows = np.zeros(n_rows, dtype=np.float64)
        running_sum = 0.0
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            idx = 0
            val = i
            while val > 0 and val % 2 == 0:
                idx += 1
                val //= 2
                if idx >= n_rows:
                    break
            if idx < n_rows:
                running_sum -= rows[idx]
                rows[idx] = rng.standard_normal()
                running_sum += rows[idx]
            out[i] = running_sum + white[i]
        mx = np.max(np.abs(out))
        if mx > 1e-12:
            out /= mx
        return out


@dataclass
class PulseWidthModulator(Oscillator):
    duty_cycle: float = 0.5

    def generate(self, duration_s: float) -> np.ndarray:
        t = self._time_array(duration_s)
        phase_acc = (self.frequency * t + self.phase / (2.0 * np.pi)) % 1.0
        dc = np.clip(self.duty_cycle, 0.01, 0.99)
        out = np.where(phase_acc < dc, 1.0, -1.0)
        return out.astype(np.float32)

    def generate_modulated(
        self, duration_s: float, lfo_freq: float = 1.0, lfo_depth: float = 0.3
    ) -> np.ndarray:
        t = self._time_array(duration_s)
        lfo = lfo_depth * np.sin(2.0 * np.pi * lfo_freq * t)
        dc = np.clip(self.duty_cycle + lfo, 0.01, 0.99)
        phase_acc = (self.frequency * t + self.phase / (2.0 * np.pi)) % 1.0
        out = np.where(phase_acc < dc, 1.0, -1.0)
        return out.astype(np.float32)
