"""Multi-stage envelope generators, LFOs, modulation matrix, portamento.

Pure-numpy signal-level building blocks for subtractive / additive /
FM synthesis chains.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


class EnvelopeStage(Enum):
    ATTACK = "attack"
    DECAY = "decay"
    SUSTAIN = "sustain"
    RELEASE = "release"
    DELAY = "delay"
    HOLD = "hold"


class LFOWaveform(Enum):
    SINE = "sine"
    TRIANGLE = "triangle"
    SAW_UP = "saw_up"
    SAW_DOWN = "saw_down"
    SQUARE = "square"
    SAMPLE_HOLD = "sample_hold"


@dataclass
class ADSRParams:
    attack_s: float = 0.01
    decay_s: float = 0.1
    sustain_level: float = 0.7
    release_s: float = 0.3


@dataclass
class DAHDSRParams:
    delay_s: float = 0.0
    attack_s: float = 0.01
    hold_s: float = 0.0
    decay_s: float = 0.1
    sustain_level: float = 0.7
    release_s: float = 0.3


@dataclass
class LFOParams:
    frequency_hz: float = 5.0
    waveform: LFOWaveform = LFOWaveform.SINE
    depth: float = 1.0
    phase_offset: float = 0.0
    unipolar: bool = False


@dataclass
class ModulationRoute:
    source: str
    destination: str
    amount: float = 1.0


def adsr_envelope(
    params: ADSRParams,
    gate_duration_s: float,
    sr: int = 44100,
) -> np.ndarray:
    """Generate a standard ADSR envelope.

    Returns a sample-level amplitude array covering the gate period
    plus the release tail.
    """
    a_samples = max(1, int(params.attack_s * sr))
    d_samples = max(1, int(params.decay_s * sr))
    r_samples = max(1, int(params.release_s * sr))
    gate_samples = max(1, int(gate_duration_s * sr))
    s_samples = max(0, gate_samples - a_samples - d_samples)

    attack = np.linspace(0.0, 1.0, a_samples, endpoint=False)
    decay = np.linspace(1.0, params.sustain_level, d_samples, endpoint=False)
    sustain = np.full(s_samples, params.sustain_level)
    release_start = params.sustain_level if s_samples > 0 else (decay[-1] if len(decay) else 1.0)
    release = np.linspace(release_start, 0.0, r_samples)

    return np.concatenate([attack, decay, sustain, release])


def dahdsr_envelope(
    params: DAHDSRParams,
    gate_duration_s: float,
    sr: int = 44100,
) -> np.ndarray:
    """Generate a DAHDSR (Delay-Attack-Hold-Decay-Sustain-Release) envelope."""
    delay = np.zeros(max(0, int(params.delay_s * sr)))
    a_samples = max(1, int(params.attack_s * sr))
    h_samples = max(0, int(params.hold_s * sr))
    d_samples = max(1, int(params.decay_s * sr))
    r_samples = max(1, int(params.release_s * sr))
    gate_samples = max(1, int(gate_duration_s * sr))
    s_samples = max(0, gate_samples - len(delay) - a_samples - h_samples - d_samples)

    attack = np.linspace(0.0, 1.0, a_samples, endpoint=False)
    hold = np.ones(h_samples)
    decay = np.linspace(1.0, params.sustain_level, d_samples, endpoint=False)
    sustain = np.full(s_samples, params.sustain_level)
    release_start = params.sustain_level if s_samples > 0 else (decay[-1] if len(decay) else 1.0)
    release = np.linspace(release_start, 0.0, r_samples)

    return np.concatenate([delay, attack, hold, decay, sustain, release])


def exponential_adsr(
    params: ADSRParams,
    gate_duration_s: float,
    sr: int = 44100,
    curve: float = 3.0,
) -> np.ndarray:
    """ADSR with exponential attack/decay/release curves."""
    a_n = max(1, int(params.attack_s * sr))
    d_n = max(1, int(params.decay_s * sr))
    r_n = max(1, int(params.release_s * sr))
    gate_n = max(1, int(gate_duration_s * sr))
    s_n = max(0, gate_n - a_n - d_n)

    t_a = np.linspace(0, 1, a_n, endpoint=False)
    attack = 1.0 - np.exp(-curve * t_a)
    attack /= attack[-1] if attack[-1] > 0 else 1.0

    t_d = np.linspace(0, 1, d_n, endpoint=False)
    decay = 1.0 - (1.0 - params.sustain_level) * (1.0 - np.exp(-curve * t_d))

    sustain = np.full(s_n, params.sustain_level)

    release_start = params.sustain_level if s_n > 0 else (decay[-1] if len(decay) else 1.0)
    t_r = np.linspace(0, 1, r_n)
    release = release_start * np.exp(-curve * t_r)

    return np.concatenate([attack, decay, sustain, release])


def multistage_envelope(
    levels: List[float],
    times_s: List[float],
    sr: int = 44100,
) -> np.ndarray:
    """Arbitrary multi-breakpoint envelope.

    ``levels`` has N+1 entries (start + one per segment);
    ``times_s`` has N entries (duration of each segment).
    """
    if len(levels) < 2 or len(times_s) < 1:
        return np.array([levels[0]] if levels else [0.0])
    segments = []
    for i, t in enumerate(times_s):
        n = max(1, int(t * sr))
        seg = np.linspace(levels[i], levels[i + 1], n, endpoint=(i == len(times_s) - 1))
        segments.append(seg)
    return np.concatenate(segments)


# --- LFO ---

def lfo_signal(
    params: LFOParams,
    duration_s: float,
    sr: int = 44100,
) -> np.ndarray:
    """Generate an LFO waveform."""
    n = max(1, int(duration_s * sr))
    t = np.arange(n) / sr
    phase = 2 * np.pi * params.frequency_hz * t + params.phase_offset

    if params.waveform == LFOWaveform.SINE:
        wave = np.sin(phase)
    elif params.waveform == LFOWaveform.TRIANGLE:
        wave = 2.0 * np.abs(2.0 * (phase / (2 * np.pi) % 1.0) - 1.0) - 1.0
    elif params.waveform == LFOWaveform.SAW_UP:
        wave = 2.0 * (phase / (2 * np.pi) % 1.0) - 1.0
    elif params.waveform == LFOWaveform.SAW_DOWN:
        wave = 1.0 - 2.0 * (phase / (2 * np.pi) % 1.0)
    elif params.waveform == LFOWaveform.SQUARE:
        wave = np.sign(np.sin(phase))
    elif params.waveform == LFOWaveform.SAMPLE_HOLD:
        period_samples = max(1, int(sr / params.frequency_hz))
        rng = np.random.default_rng(42)
        n_periods = n // period_samples + 2
        values = rng.uniform(-1.0, 1.0, n_periods)
        wave = np.repeat(values, period_samples)[:n]
    else:
        wave = np.sin(phase)

    wave *= params.depth
    if params.unipolar:
        wave = (wave + params.depth) / 2.0

    return wave


# --- Portamento / glide ---

def portamento(
    start_hz: float,
    end_hz: float,
    glide_time_s: float,
    sr: int = 44100,
    exponential: bool = True,
) -> np.ndarray:
    """Generate a frequency glide curve (portamento)."""
    n = max(1, int(glide_time_s * sr))
    if exponential and start_hz > 0 and end_hz > 0:
        return np.exp(np.linspace(np.log(start_hz), np.log(end_hz), n))
    return np.linspace(start_hz, end_hz, n)


def legato_portamento(
    frequencies_hz: List[float],
    note_durations_s: List[float],
    glide_fraction: float = 0.1,
    sr: int = 44100,
) -> np.ndarray:
    """Frequency curve for a legato passage with portamento between notes."""
    segments = []
    for i, (freq, dur) in enumerate(zip(frequencies_hz, note_durations_s)):
        n_total = max(1, int(dur * sr))
        if i > 0 and glide_fraction > 0:
            glide_n = max(1, int(n_total * glide_fraction))
            steady_n = n_total - glide_n
            glide = portamento(frequencies_hz[i - 1], freq, glide_n / sr, sr)
            steady = np.full(steady_n, freq)
            segments.append(np.concatenate([glide, steady]))
        else:
            segments.append(np.full(n_total, freq))
    return np.concatenate(segments) if segments else np.array([440.0])


# --- Modulation matrix ---

@dataclass
class ModulationMatrix:
    routes: List[ModulationRoute] = field(default_factory=list)

    def add_route(self, source: str, destination: str, amount: float = 1.0) -> None:
        self.routes.append(ModulationRoute(source=source, destination=destination, amount=amount))

    def get_modulation(self, destination: str, source_values: Dict[str, float]) -> float:
        """Sum all modulation contributions for a destination parameter."""
        total = 0.0
        for route in self.routes:
            if route.destination == destination and route.source in source_values:
                total += source_values[route.source] * route.amount
        return total

    def apply_modulation(
        self,
        base_values: Dict[str, float],
        source_values: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply all modulation routes to base parameter values."""
        result = dict(base_values)
        for dest in base_values:
            mod = self.get_modulation(dest, source_values)
            result[dest] = base_values[dest] + mod
        return result


def velocity_curve(velocity: int, curve_exp: float = 1.0) -> float:
    """Map MIDI velocity (0-127) to amplitude with adjustable curve."""
    v = np.clip(velocity, 0, 127) / 127.0
    return float(v ** curve_exp)


def key_tracking(note: int, center_note: int = 60, amount: float = 1.0) -> float:
    """Semitone offset from center note, scaled by tracking amount."""
    return (note - center_note) * amount


def aftertouch_to_mod(pressure: int, depth: float = 1.0) -> float:
    """Convert MIDI aftertouch (0-127) to modulation value."""
    return (np.clip(pressure, 0, 127) / 127.0) * depth
