"""Wavetable synthesis: single-cycle waveforms, morphing banks, band-limited playback."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Wavetable:
    samples: np.ndarray = field(default_factory=lambda: np.zeros(2048, dtype=np.float64))

    def __post_init__(self):
        self.samples = np.asarray(self.samples, dtype=np.float64)

    @classmethod
    def from_harmonics(cls, amplitudes: list[float], size: int = 2048) -> Wavetable:
        t = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
        wt = np.zeros(size, dtype=np.float64)
        for k, amp in enumerate(amplitudes, start=1):
            wt += amp * np.sin(k * t)
        mx = np.max(np.abs(wt))
        if mx > 1e-12:
            wt /= mx
        return cls(samples=wt)

    @classmethod
    def sine(cls, size: int = 2048) -> Wavetable:
        return cls.from_harmonics([1.0], size)

    @classmethod
    def saw(cls, n_harmonics: int = 64, size: int = 2048) -> Wavetable:
        amps = [1.0 / k for k in range(1, n_harmonics + 1)]
        return cls.from_harmonics(amps, size)

    @classmethod
    def square(cls, n_harmonics: int = 32, size: int = 2048) -> Wavetable:
        amps = [1.0 / k if k % 2 == 1 else 0.0 for k in range(1, 2 * n_harmonics + 1)]
        return cls.from_harmonics(amps, size)

    @classmethod
    def triangle(cls, n_harmonics: int = 32, size: int = 2048) -> Wavetable:
        amps = []
        for k in range(1, 2 * n_harmonics + 1):
            if k % 2 == 1:
                sign = 1.0 if ((k - 1) // 2) % 2 == 0 else -1.0
                amps.append(sign / (k * k))
            else:
                amps.append(0.0)
        return cls.from_harmonics(amps, size)

    def lookup(self, phase: float) -> float:
        phase = phase % 1.0
        pos = phase * len(self.samples)
        i0 = int(pos) % len(self.samples)
        i1 = (i0 + 1) % len(self.samples)
        frac = pos - int(pos)
        return float(self.samples[i0] * (1.0 - frac) + self.samples[i1] * frac)

    def lookup_cubic(self, phase: float) -> float:
        phase = phase % 1.0
        n = len(self.samples)
        pos = phase * n
        i1 = int(pos) % n
        i0 = (i1 - 1) % n
        i2 = (i1 + 1) % n
        i3 = (i1 + 2) % n
        frac = pos - int(pos)
        y0, y1, y2, y3 = self.samples[i0], self.samples[i1], self.samples[i2], self.samples[i3]
        a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
        b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c = -0.5 * y0 + 0.5 * y2
        d = y1
        return float(((a * frac + b) * frac + c) * frac + d)


@dataclass
class WavetableBank:
    tables: list[Wavetable] = field(default_factory=list)

    @classmethod
    def basic_set(cls) -> WavetableBank:
        return cls(tables=[Wavetable.sine(), Wavetable.triangle(), Wavetable.saw(), Wavetable.square()])

    def lookup(self, position: float, phase: float) -> float:
        if not self.tables:
            return 0.0
        if len(self.tables) == 1:
            return self.tables[0].lookup_cubic(phase)
        position = np.clip(position, 0.0, 1.0)
        scaled = position * (len(self.tables) - 1)
        idx_a = int(scaled)
        idx_b = min(idx_a + 1, len(self.tables) - 1)
        frac = scaled - idx_a
        val_a = self.tables[idx_a].lookup_cubic(phase)
        val_b = self.tables[idx_b].lookup_cubic(phase)
        return val_a * (1.0 - frac) + val_b * frac


class WavetableOscillator:
    def __init__(self, bank: Optional[WavetableBank] = None, sr: int = 44100):
        self.bank = bank or WavetableBank.basic_set()
        self.sr = sr

    def generate(
        self,
        frequency: float,
        duration_s: float,
        position: float = 0.0,
    ) -> np.ndarray:
        n = int(self.sr * duration_s)
        out = np.zeros(n, dtype=np.float64)
        phase_inc = frequency / self.sr
        phase = 0.0
        for i in range(n):
            out[i] = self.bank.lookup(position, phase)
            phase = (phase + phase_inc) % 1.0
        return out.astype(np.float32)

    def generate_sweep(
        self,
        frequency: float,
        duration_s: float,
        pos_start: float = 0.0,
        pos_end: float = 1.0,
    ) -> np.ndarray:
        n = int(self.sr * duration_s)
        out = np.zeros(n, dtype=np.float64)
        phase_inc = frequency / self.sr
        phase = 0.0
        positions = np.linspace(pos_start, pos_end, n)
        for i in range(n):
            out[i] = self.bank.lookup(float(positions[i]), phase)
            phase = (phase + phase_inc) % 1.0
        return out.astype(np.float32)


class WavetableSynthesizer:
    def __init__(self, bank: Optional[WavetableBank] = None, sr: int = 44100):
        self.osc = WavetableOscillator(bank, sr)
        self.sr = sr

    def render_note(
        self,
        frequency: float,
        duration_s: float,
        position: float = 0.0,
        velocity: float = 1.0,
        attack: float = 0.01,
        release: float = 0.05,
    ) -> np.ndarray:
        raw = self.osc.generate(frequency, duration_s, position)
        n = len(raw)
        env = np.ones(n, dtype=np.float64)
        a_samp = int(self.sr * attack)
        r_samp = int(self.sr * release)
        if a_samp > 0:
            env[:a_samp] = np.linspace(0.0, 1.0, a_samp)
        if r_samp > 0 and r_samp < n:
            env[n - r_samp:] = np.linspace(1.0, 0.0, r_samp)
        return (raw * env * velocity).astype(np.float32)
