"""Subtractive synthesis: biquad filters, filter envelopes, LFO modulation.

Classic analog synth model — sawtooth/square through resonant filters
with time-varying cutoff.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class FilterType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"


def biquad_coefficients(
    filter_type: FilterType,
    cutoff_hz: float,
    sr: int = 44100,
    q: float = 0.707,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute biquad IIR coefficients (b, a) for the given filter type.

    Robert Bristow-Johnson's Audio EQ Cookbook formulae.
    """
    omega = 2.0 * np.pi * np.clip(cutoff_hz, 20.0, sr / 2.0 - 1.0) / sr
    sin_w = np.sin(omega)
    cos_w = np.cos(omega)
    alpha = sin_w / (2.0 * max(q, 0.01))

    if filter_type == FilterType.LOWPASS:
        b0 = (1.0 - cos_w) / 2.0
        b1 = 1.0 - cos_w
        b2 = b0
    elif filter_type == FilterType.HIGHPASS:
        b0 = (1.0 + cos_w) / 2.0
        b1 = -(1.0 + cos_w)
        b2 = b0
    elif filter_type == FilterType.BANDPASS:
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def apply_biquad(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply a biquad filter to *signal* using direct-form II transposed."""
    out = np.zeros_like(signal)
    z1 = 0.0
    z2 = 0.0
    for i in range(len(signal)):
        x = float(signal[i])
        y = b[0] * x + z1
        z1 = b[1] * x - a[1] * y + z2
        z2 = b[2] * x - a[2] * y
        out[i] = y
    return out


def apply_biquad_tv(
    signal: np.ndarray,
    filter_type: FilterType,
    cutoff_curve: np.ndarray,
    sr: int = 44100,
    q: float = 0.707,
) -> np.ndarray:
    """Apply a biquad filter with a time-varying cutoff (one coefficient set per sample)."""
    n = len(signal)
    if len(cutoff_curve) < n:
        cutoff_curve = np.pad(cutoff_curve, (0, n - len(cutoff_curve)), mode="edge")
    out = np.zeros(n, dtype=np.float64)
    z1 = 0.0
    z2 = 0.0
    for i in range(n):
        b, a = biquad_coefficients(filter_type, float(cutoff_curve[i]), sr, q)
        x = float(signal[i])
        y = b[0] * x + z1
        z1 = b[1] * x - a[1] * y + z2
        z2 = b[2] * x - a[2] * y
        out[i] = y
    return out


@dataclass
class FilterEnvelope:
    """ADSR envelope for filter cutoff modulation."""
    attack: float = 0.05
    decay: float = 0.2
    sustain: float = 0.4
    release: float = 0.3
    amount: float = 4000.0  # Hz range of modulation

    def generate(self, base_cutoff: float, duration_s: float, sr: int = 44100) -> np.ndarray:
        n = int(sr * duration_s)
        if n == 0:
            return np.array([], dtype=np.float64)
        env = np.zeros(n, dtype=np.float64)
        a_samp = int(sr * self.attack)
        d_samp = int(sr * self.decay)
        r_samp = int(sr * self.release)
        sus_end = max(0, n - r_samp)
        for i in range(n):
            if i < a_samp:
                env[i] = i / max(a_samp, 1)
            elif i < a_samp + d_samp:
                frac = (i - a_samp) / max(d_samp, 1)
                env[i] = 1.0 - frac * (1.0 - self.sustain)
            elif i < sus_end:
                env[i] = self.sustain
            else:
                frac = (i - sus_end) / max(r_samp, 1)
                env[i] = self.sustain * (1.0 - min(frac, 1.0))
        return base_cutoff + self.amount * env


@dataclass
class LFO:
    """Low-frequency oscillator for modulation."""
    rate_hz: float = 5.0
    depth: float = 500.0  # Hz peak deviation

    def generate(self, duration_s: float, sr: int = 44100) -> np.ndarray:
        n = int(sr * duration_s)
        t = np.arange(n, dtype=np.float64) / sr
        return self.depth * np.sin(2.0 * np.pi * self.rate_hz * t)


class SubtractiveSynth:
    """Classic subtractive synthesizer: oscillator → filter → output."""

    def __init__(
        self,
        sr: int = 44100,
        filter_type: FilterType = FilterType.LOWPASS,
        cutoff_hz: float = 2000.0,
        resonance: float = 1.0,
        filter_envelope: Optional[FilterEnvelope] = None,
        lfo: Optional[LFO] = None,
    ):
        self.sr = sr
        self.filter_type = filter_type
        self.cutoff_hz = cutoff_hz
        self.resonance = resonance
        self.filter_envelope = filter_envelope
        self.lfo = lfo

    def _sawtooth(self, freq: float, n: int) -> np.ndarray:
        t = np.arange(n, dtype=np.float64) / self.sr
        phase = (freq * t) % 1.0
        return 2.0 * phase - 1.0

    def _square(self, freq: float, n: int) -> np.ndarray:
        t = np.arange(n, dtype=np.float64) / self.sr
        return np.sign(np.sin(2.0 * np.pi * freq * t))

    def render(
        self,
        freq: float,
        duration_s: float,
        waveform: str = "sawtooth",
        velocity: float = 1.0,
    ) -> np.ndarray:
        n = int(self.sr * duration_s)
        if n == 0:
            return np.array([], dtype=np.float32)

        if waveform == "square":
            raw = self._square(freq, n)
        else:
            raw = self._sawtooth(freq, n)

        cutoff_curve = np.full(n, self.cutoff_hz, dtype=np.float64)
        if self.filter_envelope is not None:
            cutoff_curve = self.filter_envelope.generate(self.cutoff_hz, duration_s, self.sr)
        if self.lfo is not None:
            lfo_sig = self.lfo.generate(duration_s, self.sr)
            cutoff_curve = cutoff_curve + lfo_sig[:n]
        cutoff_curve = np.clip(cutoff_curve, 20.0, self.sr / 2.0 - 1.0)

        filtered = apply_biquad_tv(raw, self.filter_type, cutoff_curve, self.sr, self.resonance)
        filtered *= velocity

        mx = np.max(np.abs(filtered))
        if mx > 1.0:
            filtered /= mx
        return filtered.astype(np.float32)
