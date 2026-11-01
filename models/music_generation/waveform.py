"""Waveform synthesis primitives: oscillators, envelopes, noise generators."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class WaveShape(Enum):
    SINE = "sine"
    SAW = "saw"
    SQUARE = "square"
    TRIANGLE = "triangle"


@dataclass
class Oscillator:
    frequency: float = 440.0
    amplitude: float = 1.0
    phase: float = 0.0
    shape: WaveShape = WaveShape.SINE

    def generate(self, duration_s: float, sr: int = 22050) -> np.ndarray:
        t = np.arange(int(sr * duration_s)) / sr
        phi = 2 * np.pi * self.frequency * t + self.phase
        if self.shape == WaveShape.SINE:
            wave = np.sin(phi)
        elif self.shape == WaveShape.SAW:
            wave = 2.0 * (phi / (2 * np.pi) % 1.0) - 1.0
        elif self.shape == WaveShape.SQUARE:
            wave = np.sign(np.sin(phi))
        elif self.shape == WaveShape.TRIANGLE:
            wave = 2.0 * np.abs(2.0 * (phi / (2 * np.pi) % 1.0) - 1.0) - 1.0
        else:
            wave = np.sin(phi)
        return (self.amplitude * wave).astype(np.float32)


@dataclass
class ADSREnvelope:
    attack: float = 0.01
    decay: float = 0.05
    sustain_level: float = 0.7
    release: float = 0.1

    def apply(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        n = len(signal)
        env = np.ones(n, dtype=np.float32)
        a_samples = int(self.attack * sr)
        d_samples = int(self.decay * sr)
        r_samples = int(self.release * sr)
        s_samples = max(0, n - a_samples - d_samples - r_samples)

        idx = 0
        if a_samples > 0:
            end = min(idx + a_samples, n)
            env[idx:end] = np.linspace(0, 1, end - idx, dtype=np.float32)
            idx = end
        if d_samples > 0 and idx < n:
            end = min(idx + d_samples, n)
            env[idx:end] = np.linspace(1, self.sustain_level, end - idx, dtype=np.float32)
            idx = end
        if s_samples > 0 and idx < n:
            end = min(idx + s_samples, n)
            env[idx:end] = self.sustain_level
            idx = end
        if r_samples > 0 and idx < n:
            env[idx:n] = np.linspace(self.sustain_level, 0, n - idx, dtype=np.float32)

        return signal * env


class WaveformSynthesizer:
    def __init__(self):
        self._oscillators: list[Oscillator] = []

    def add_oscillator(self, osc: Oscillator) -> None:
        self._oscillators.append(osc)

    def synthesize(self, duration_s: float, sr: int = 22050, normalize: bool = True) -> np.ndarray:
        if not self._oscillators:
            return np.zeros(int(sr * duration_s), dtype=np.float32)
        waves = [osc.generate(duration_s, sr) for osc in self._oscillators]
        mixed = sum(waves)
        if normalize and np.max(np.abs(mixed)) > 0:
            mixed = mixed / np.max(np.abs(mixed))
        return mixed.astype(np.float32)

    def clear(self) -> None:
        self._oscillators.clear()


class NoiseGenerator:
    @staticmethod
    def white(n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.standard_normal(n_samples).astype(np.float32)

    @staticmethod
    def pink(n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        white = rng.standard_normal(n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1.0)
        freqs[0] = 1.0
        fft /= np.sqrt(freqs)
        out = np.fft.irfft(fft, n=n_samples)
        if np.max(np.abs(out)) > 0:
            out /= np.max(np.abs(out))
        return out.astype(np.float32)

    @staticmethod
    def brown(n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        white = rng.standard_normal(n_samples)
        out = np.cumsum(white)
        if np.max(np.abs(out)) > 0:
            out /= np.max(np.abs(out))
        return out.astype(np.float32)
