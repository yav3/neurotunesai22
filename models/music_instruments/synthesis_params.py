"""Synthesizer parameters: ADSR envelopes, oscillator types, patch modeling."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class OscillatorType(Enum):
    SINE = "sine"
    SAW = "sawtooth"
    SQUARE = "square"
    TRIANGLE = "triangle"
    NOISE = "noise"
    WAVETABLE = "wavetable"
    FM = "fm"
    GRANULAR = "granular"


@dataclass
class ADSREnvelope:
    attack_s: float = 0.01
    decay_s: float = 0.1
    sustain: float = 0.7
    release_s: float = 0.3

    def level_at(self, t: float, gate_off_t: float | None = None) -> float:
        if gate_off_t is not None and t >= gate_off_t:
            elapsed = t - gate_off_t
            sustain_level = self.sustain
            if elapsed >= self.release_s:
                return 0.0
            return sustain_level * (1 - elapsed / self.release_s)
        if t < self.attack_s:
            return t / self.attack_s if self.attack_s > 0 else 1.0
        t2 = t - self.attack_s
        if t2 < self.decay_s:
            return 1.0 - (1.0 - self.sustain) * (t2 / self.decay_s) if self.decay_s > 0 else self.sustain
        return self.sustain


@dataclass
class SynthPatch:
    name: str
    osc_type: OscillatorType
    envelope: ADSREnvelope
    filter_cutoff_hz: float = 5000.0
    filter_resonance: float = 0.0
    detune_cents: float = 0.0
    fm_ratio: float = 1.0
    fm_depth: float = 0.0
    lfo_rate_hz: float = 0.0
    lfo_depth: float = 0.0


def default_patch(osc: OscillatorType = OscillatorType.SAW) -> SynthPatch:
    return SynthPatch(name="init", osc_type=osc, envelope=ADSREnvelope())


def fm_ratio(carrier_hz: float, modulator_hz: float) -> float:
    if carrier_hz <= 0:
        return 0.0
    return modulator_hz / carrier_hz


def detune_cents_to_ratio(cents: float) -> float:
    return 2.0 ** (cents / 1200.0)


def filter_cutoff_from_velocity(
    velocity: int,
    min_cutoff: float = 200.0,
    max_cutoff: float = 8000.0,
) -> float:
    v = max(0, min(127, velocity)) / 127.0
    return min_cutoff + (max_cutoff - min_cutoff) * v


def lfo_depth_curve(
    rate_hz: float,
    max_depth: float = 1.0,
    rolloff_hz: float = 10.0,
) -> float:
    return max_depth / (1.0 + (rate_hz / rolloff_hz) ** 2)
