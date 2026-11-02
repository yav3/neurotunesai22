"""Physical modeling synthesis: Karplus-Strong strings, reed model,
bar/membrane percussion, bowed string, parameter estimation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class KarplusStrongParams:
    frequency: float = 440.0
    sr: float = 44100.0
    duration: float = 1.0
    damping: float = 0.996
    blend: float = 0.5
    stretch: float = 0.5


def karplus_strong(params: KarplusStrongParams) -> np.ndarray:
    n_samples = int(params.sr * params.duration)
    delay = int(params.sr / params.frequency)
    if delay < 2:
        delay = 2

    rng = np.random.default_rng(42)
    buf = rng.uniform(-1.0, 1.0, delay).astype(np.float64)
    output = np.zeros(n_samples)

    idx = 0
    for i in range(n_samples):
        output[i] = buf[idx]
        next_idx = (idx + 1) % delay
        if rng.random() < params.blend:
            avg = 0.5 * (buf[idx] + buf[next_idx])
        else:
            avg = buf[idx]
        if rng.random() < params.stretch:
            buf[idx] = params.damping * avg
        else:
            buf[idx] = params.damping * buf[idx]
        idx = next_idx

    return output


def karplus_strong_extended(frequency: float, sr: float = 44100.0,
                            duration: float = 1.0, damping: float = 0.996,
                            pickup_position: float = 0.5) -> np.ndarray:
    n_samples = int(sr * duration)
    delay = int(sr / frequency)
    if delay < 2:
        delay = 2

    pickup_idx = max(1, int(pickup_position * delay))
    rng = np.random.default_rng(0)
    buf = rng.uniform(-1.0, 1.0, delay).astype(np.float64)
    output = np.zeros(n_samples)

    idx = 0
    for i in range(n_samples):
        read_idx = (idx + pickup_idx) % delay
        output[i] = buf[read_idx]
        next_idx = (idx + 1) % delay
        buf[idx] = damping * 0.5 * (buf[idx] + buf[next_idx])
        idx = next_idx

    return output


@dataclass
class ReedModelParams:
    frequency: float = 261.63
    sr: float = 44100.0
    duration: float = 1.0
    blowing_pressure: float = 0.6
    reed_stiffness: float = 0.5
    bore_loss: float = 0.95
    noise_level: float = 0.02


def reed_model(params: ReedModelParams) -> np.ndarray:
    n_samples = int(params.sr * params.duration)
    delay = int(params.sr / (2.0 * params.frequency))
    if delay < 1:
        delay = 1

    bore = np.zeros(delay)
    output = np.zeros(n_samples)
    rng = np.random.default_rng(1)

    idx = 0
    for i in range(n_samples):
        reflection = -params.bore_loss * bore[(idx - 1) % delay]
        pressure_diff = params.blowing_pressure + reflection
        noise = params.noise_level * rng.standard_normal()

        reed_opening = max(0.0, 1.0 - params.reed_stiffness * pressure_diff)
        flow = reed_opening * (params.blowing_pressure - reflection) + noise

        bore[idx] = flow
        output[i] = bore[(idx - delay // 2) % delay]
        idx = (idx + 1) % delay

    peak = np.max(np.abs(output)) + 1e-12
    return output / peak


@dataclass
class ModalParams:
    frequencies: List[float] = field(default_factory=lambda: [440.0, 1100.0, 1760.0])
    amplitudes: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    decay_rates: List[float] = field(default_factory=lambda: [3.0, 5.0, 8.0])
    sr: float = 44100.0
    duration: float = 1.0


def modal_synthesis(params: ModalParams) -> np.ndarray:
    n_samples = int(params.sr * params.duration)
    t = np.arange(n_samples) / params.sr
    output = np.zeros(n_samples)

    for freq, amp, decay in zip(params.frequencies, params.amplitudes, params.decay_rates):
        mode = amp * np.exp(-decay * t) * np.sin(2.0 * np.pi * freq * t)
        output += mode

    peak = np.max(np.abs(output)) + 1e-12
    return output / peak


def bar_frequencies(fundamental: float, n_modes: int = 6) -> List[float]:
    ratios = [1.0, 2.756, 5.404, 8.933, 13.344, 18.637]
    return [fundamental * ratios[i] for i in range(min(n_modes, len(ratios)))]


def membrane_frequencies(fundamental: float, n_modes: int = 6) -> List[float]:
    ratios = [1.0, 1.594, 2.136, 2.296, 2.653, 2.917]
    return [fundamental * ratios[i] for i in range(min(n_modes, len(ratios)))]


def bar_model(fundamental: float = 440.0, sr: float = 44100.0,
              duration: float = 1.0, n_modes: int = 5) -> np.ndarray:
    freqs = bar_frequencies(fundamental, n_modes)
    amps = [1.0 / (i + 1) ** 1.5 for i in range(len(freqs))]
    decays = [2.0 + 3.0 * i for i in range(len(freqs))]
    params = ModalParams(frequencies=freqs, amplitudes=amps,
                         decay_rates=decays, sr=sr, duration=duration)
    return modal_synthesis(params)


def membrane_model(fundamental: float = 200.0, sr: float = 44100.0,
                   duration: float = 0.5, n_modes: int = 6) -> np.ndarray:
    freqs = membrane_frequencies(fundamental, n_modes)
    amps = [1.0 / (i + 1) ** 1.2 for i in range(len(freqs))]
    decays = [5.0 + 4.0 * i for i in range(len(freqs))]
    params = ModalParams(frequencies=freqs, amplitudes=amps,
                         decay_rates=decays, sr=sr, duration=duration)
    return modal_synthesis(params)


@dataclass
class BowedStringParams:
    frequency: float = 440.0
    sr: float = 44100.0
    duration: float = 1.0
    bow_velocity: float = 0.3
    bow_force: float = 0.5
    bow_position: float = 0.12
    damping: float = 0.995


def _friction_curve(v_rel: float, force: float) -> float:
    mu_s = 0.8
    mu_d = 0.3
    v_s = 0.01
    if abs(v_rel) < v_s:
        return mu_s * force * v_rel / v_s
    return mu_d * force * np.sign(v_rel)


def bowed_string(params: BowedStringParams) -> np.ndarray:
    n_samples = int(params.sr * params.duration)
    delay = int(params.sr / params.frequency)
    if delay < 4:
        delay = 4

    bow_idx = max(1, int(params.bow_position * delay))
    nut_delay = np.zeros(bow_idx)
    bridge_delay = np.zeros(delay - bow_idx)

    output = np.zeros(n_samples)
    v_bow = params.bow_velocity
    nut_ptr = 0
    bridge_ptr = 0

    for i in range(n_samples):
        v_nut = nut_delay[(nut_ptr - 1) % len(nut_delay)]
        v_bridge = bridge_delay[(bridge_ptr - 1) % len(bridge_delay)]
        v_string = v_nut + v_bridge

        v_rel = v_bow - v_string
        f_friction = _friction_curve(v_rel, params.bow_force)

        nut_delay[nut_ptr] = params.damping * (v_bridge + f_friction)
        bridge_delay[bridge_ptr] = params.damping * (v_nut + f_friction)

        output[i] = bridge_delay[bridge_ptr]
        nut_ptr = (nut_ptr + 1) % len(nut_delay)
        bridge_ptr = (bridge_ptr + 1) % len(bridge_delay)

    peak = np.max(np.abs(output)) + 1e-12
    return output / peak


@dataclass
class EstimatedParams:
    frequency: float
    decay_rate: float
    n_harmonics: int
    harmonic_amplitudes: List[float]
    attack_time_s: float
    residual_energy: float


def estimate_parameters(signal: np.ndarray, sr: float = 44100.0) -> EstimatedParams:
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    peak_idx = np.argmax(spectrum[1:]) + 1
    f0 = float(freqs[peak_idx])
    if f0 < 20.0:
        f0 = 20.0

    harmonics: List[float] = []
    for h in range(1, 17):
        target = h * f0
        idx = int(round(target * n / sr))
        if idx >= len(spectrum):
            break
        window = max(1, int(0.5 * f0 * n / sr))
        lo = max(0, idx - window)
        hi = min(len(spectrum), idx + window + 1)
        harmonics.append(float(np.max(spectrum[lo:hi])))

    if harmonics:
        h_max = max(harmonics)
        harmonics = [a / (h_max + 1e-12) for a in harmonics]

    analytic = np.abs(signal)
    peak_val = np.max(analytic)
    if peak_val > 0:
        peak_pos = np.argmax(analytic)
        attack_time = float(peak_pos / sr)

        tail = analytic[peak_pos:]
        if len(tail) > 10:
            threshold = peak_val * np.exp(-1)
            below = np.where(tail < threshold)[0]
            if len(below) > 0:
                decay_time = float(below[0] / sr)
                decay_rate = 1.0 / (decay_time + 1e-6)
            else:
                decay_rate = 0.5
        else:
            decay_rate = 1.0
    else:
        attack_time = 0.0
        decay_rate = 1.0

    harmonic_power = sum(a ** 2 for a in harmonics)
    total_power = float(np.sum(spectrum ** 2))
    residual = 1.0 - harmonic_power / (total_power + 1e-12)

    return EstimatedParams(
        frequency=f0,
        decay_rate=decay_rate,
        n_harmonics=len(harmonics),
        harmonic_amplitudes=harmonics,
        attack_time_s=attack_time,
        residual_energy=max(0.0, min(1.0, residual)),
    )


def excitation_resonance_decompose(signal: np.ndarray,
                                   sr: float = 44100.0,
                                   lpc_order: int = 24
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.signal import lfilter

    n = len(signal)
    windowed = signal * np.hamming(n)

    r = np.correlate(windowed, windowed, mode='full')[n - 1:n + lpc_order]
    r[0] += 1e-10

    a = np.zeros(lpc_order + 1)
    a[0] = 1.0
    e = r[0]

    for i in range(1, lpc_order + 1):
        lam = -sum(a[j] * r[i - j] for j in range(1, i)) - r[i]
        lam /= e + 1e-12
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + lam * a[i - j]
        a_new[i] = lam
        a = a_new
        e *= (1.0 - lam * lam)
        if e <= 0:
            break

    excitation = lfilter(a[:lpc_order + 1], [1.0], signal)
    resonance = lfilter([1.0], a[:lpc_order + 1], excitation)

    return excitation, resonance
