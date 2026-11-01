"""Additive synthesis: partials, spectra, per-partial envelopes, spectral morphing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AdditivePartial:
    frequency: float = 440.0
    amplitude: float = 1.0
    phase: float = 0.0
    decay_rate: float = 0.0

    def render(self, duration_s: float, sr: int = 44100) -> np.ndarray:
        n = int(sr * duration_s)
        t = np.arange(n, dtype=np.float64) / sr
        env = self.amplitude * np.exp(-self.decay_rate * t) if self.decay_rate > 0 else np.full(n, self.amplitude)
        return (env * np.sin(2.0 * np.pi * self.frequency * t + self.phase)).astype(np.float64)


@dataclass
class AdditiveSpectrum:
    partials: list[AdditivePartial] = field(default_factory=list)

    @classmethod
    def harmonic_series(cls, fundamental: float, n_harmonics: int = 16, rolloff: float = 1.0) -> AdditiveSpectrum:
        partials = []
        for k in range(1, n_harmonics + 1):
            amp = 1.0 / (k ** rolloff)
            partials.append(AdditivePartial(frequency=fundamental * k, amplitude=amp))
        return cls(partials=partials)

    @classmethod
    def odd_harmonics(cls, fundamental: float, n_harmonics: int = 8) -> AdditiveSpectrum:
        partials = []
        for i in range(n_harmonics):
            k = 2 * i + 1
            amp = 1.0 / k
            partials.append(AdditivePartial(frequency=fundamental * k, amplitude=amp))
        return cls(partials=partials)

    @classmethod
    def inharmonic(cls, fundamental: float, n_partials: int = 12, stretch: float = 1.01) -> AdditiveSpectrum:
        partials = []
        for k in range(1, n_partials + 1):
            freq = fundamental * k * (stretch ** (k - 1))
            amp = 1.0 / k
            partials.append(AdditivePartial(frequency=freq, amplitude=amp))
        return cls(partials=partials)

    def render(self, duration_s: float, sr: int = 44100) -> np.ndarray:
        n = int(sr * duration_s)
        out = np.zeros(n, dtype=np.float64)
        for p in self.partials:
            if p.frequency < sr / 2.0:
                out += p.render(duration_s, sr)
        return out

    def normalize(self) -> AdditiveSpectrum:
        if not self.partials:
            return AdditiveSpectrum(partials=[])
        mx = max(abs(p.amplitude) for p in self.partials)
        if mx < 1e-12:
            return AdditiveSpectrum(partials=list(self.partials))
        return AdditiveSpectrum(
            partials=[AdditivePartial(p.frequency, p.amplitude / mx, p.phase, p.decay_rate) for p in self.partials]
        )


class AdditiveSynthesizer:
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def render(self, spectrum: AdditiveSpectrum, duration_s: float, amplitude: float = 1.0) -> np.ndarray:
        out = spectrum.render(duration_s, self.sr)
        mx = np.max(np.abs(out))
        if mx > 1e-12:
            out = out / mx * amplitude
        return out.astype(np.float32)

    def render_with_envelope(
        self, spectrum: AdditiveSpectrum, duration_s: float, envelope: np.ndarray
    ) -> np.ndarray:
        out = spectrum.render(duration_s, self.sr)
        n = len(out)
        if len(envelope) < n:
            envelope = np.pad(envelope, (0, n - len(envelope)))
        elif len(envelope) > n:
            envelope = envelope[:n]
        mx = np.max(np.abs(out))
        if mx > 1e-12:
            out /= mx
        return (out * envelope).astype(np.float32)


class SpectralMorphing:
    @staticmethod
    def morph(
        spectrum_a: AdditiveSpectrum,
        spectrum_b: AdditiveSpectrum,
        alpha: float,
        duration_s: float,
        sr: int = 44100,
    ) -> np.ndarray:
        alpha = np.clip(alpha, 0.0, 1.0)
        all_freqs: dict[float, tuple[float, float, float, float]] = {}
        for p in spectrum_a.partials:
            all_freqs[p.frequency] = (p.amplitude, p.phase, p.decay_rate, 0.0)
        for p in spectrum_b.partials:
            if p.frequency in all_freqs:
                a_amp, a_ph, a_dec, _ = all_freqs[p.frequency]
                all_freqs[p.frequency] = (a_amp, a_ph, a_dec, p.amplitude)
            else:
                all_freqs[p.frequency] = (0.0, 0.0, 0.0, p.amplitude)

        n = int(sr * duration_s)
        out = np.zeros(n, dtype=np.float64)
        t = np.arange(n, dtype=np.float64) / sr
        for freq, (a_amp, a_ph, a_dec, b_amp) in all_freqs.items():
            if freq >= sr / 2.0:
                continue
            amp = (1.0 - alpha) * a_amp + alpha * b_amp
            if amp < 1e-12:
                continue
            env = amp * np.exp(-a_dec * (1.0 - alpha) * t) if a_dec > 0 else np.full(n, amp)
            out += env * np.sin(2.0 * np.pi * freq * t + a_ph * (1.0 - alpha))

        mx = np.max(np.abs(out))
        if mx > 1e-12:
            out /= mx
        return out.astype(np.float32)

    @staticmethod
    def crossfade(
        spectrum_a: AdditiveSpectrum,
        spectrum_b: AdditiveSpectrum,
        duration_s: float,
        sr: int = 44100,
    ) -> np.ndarray:
        n = int(sr * duration_s)
        sig_a = spectrum_a.render(duration_s, sr)
        sig_b = spectrum_b.render(duration_s, sr)
        if len(sig_a) < n:
            sig_a = np.pad(sig_a, (0, n - len(sig_a)))
        if len(sig_b) < n:
            sig_b = np.pad(sig_b, (0, n - len(sig_b)))
        ramp = np.linspace(0.0, 1.0, n)
        out = (1.0 - ramp) * sig_a + ramp * sig_b
        mx = np.max(np.abs(out))
        if mx > 1e-12:
            out /= mx
        return out.astype(np.float32)
