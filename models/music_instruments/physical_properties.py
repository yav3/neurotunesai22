"""Physical properties of musical instruments: material resonance,
body modes, radiation patterns, string-air-body coupling.

Based on Fletcher & Rossing (1998) physics of musical instruments
and Chaigne & Kergomard (2016) acoustics of musical instruments.
Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class MaterialProperties:
    density: float           # kg/m^3
    youngs_modulus: float    # Pa
    poisson_ratio: float = 0.3
    loss_factor: float = 0.01


@dataclass
class ResonanceMode:
    frequency: float
    amplitude: float
    q_factor: float
    mode_shape: Optional[np.ndarray] = None


@dataclass
class InstrumentBody:
    modes: List[ResonanceMode] = field(default_factory=list)
    radiation_efficiency: float = 0.5
    volume: float = 0.0
    wall_thickness: float = 0.003


SPRUCE = MaterialProperties(400.0, 10e9, 0.37, 0.008)
MAPLE = MaterialProperties(660.0, 12e9, 0.35, 0.012)
EBONY = MaterialProperties(1100.0, 16e9, 0.30, 0.006)
STEEL = MaterialProperties(7800.0, 200e9, 0.30, 0.002)
NYLON = MaterialProperties(1150.0, 3e9, 0.40, 0.05)
BRASS = MaterialProperties(8500.0, 110e9, 0.34, 0.003)


def string_tension(f0: float, length: float,
                   linear_density: float) -> float:
    return 4.0 * linear_density * length ** 2 * f0 ** 2


def string_wave_speed(tension: float,
                      linear_density: float) -> float:
    return np.sqrt(tension / linear_density)


def body_mode_frequencies(length: float, width: float,
                          thickness: float,
                          material: MaterialProperties,
                          n_modes: int = 10) -> np.ndarray:
    c_plate = np.sqrt(material.youngs_modulus / material.density)
    kappa = thickness / np.sqrt(12.0)
    modes = []
    for m in range(1, n_modes + 1):
        for n in range(1, 3):
            f = (np.pi * kappa * c_plate / 2.0) * np.sqrt(
                (m / length) ** 2 + (n / width) ** 2
            )
            modes.append(f)
    modes.sort()
    return np.array(modes[:n_modes])


def helmholtz_resonance(volume: float, neck_length: float,
                        neck_area: float,
                        c: float = 343.0) -> float:
    effective_length = neck_length + 1.7 * np.sqrt(neck_area / np.pi)
    return c / (2 * np.pi) * np.sqrt(neck_area / (volume * effective_length))


def soundboard_radiation(mode_freqs: np.ndarray,
                         mode_amps: np.ndarray,
                         eval_freqs: np.ndarray,
                         damping: float = 0.02) -> np.ndarray:
    response = np.zeros(len(eval_freqs))
    for i in range(len(mode_freqs)):
        f0 = mode_freqs[i]
        a = mode_amps[i]
        r = eval_freqs / (f0 + 1e-10)
        mag = a / np.sqrt((1 - r ** 2) ** 2 + (2 * damping * r) ** 2)
        response += mag
    return response


def impedance_matching(z_source: float, z_load: float) -> float:
    return 4 * z_source * z_load / (z_source + z_load) ** 2


def coupling_efficiency(z_string: float, z_bridge: float,
                        z_body: float) -> float:
    eta_bridge = impedance_matching(z_string, z_bridge)
    eta_body = impedance_matching(z_bridge, z_body)
    return eta_bridge * eta_body


def instrument_formants(body_modes: List[ResonanceMode],
                        freq_range: Tuple[float, float] = (80, 5000),
                        n_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    freqs = np.linspace(freq_range[0], freq_range[1], n_points)
    spectrum = np.zeros(n_points)
    for mode in body_modes:
        f0 = mode.frequency
        q = mode.q_factor
        bw = f0 / (q + 1e-10)
        gaussian = mode.amplitude * np.exp(-0.5 * ((freqs - f0) / (bw / 2)) ** 2)
        spectrum += gaussian
    return freqs, spectrum


def spectral_envelope_model(fundamental: float,
                            body_modes: List[ResonanceMode],
                            n_harmonics: int = 30) -> np.ndarray:
    harmonics = fundamental * np.arange(1, n_harmonics + 1)
    amplitudes = np.zeros(n_harmonics)
    for i, f in enumerate(harmonics):
        amp = 1.0 / (i + 1)
        for mode in body_modes:
            r = f / (mode.frequency + 1e-10)
            bw = mode.frequency / (mode.q_factor + 1e-10)
            gain = mode.amplitude / np.sqrt(1 + ((f - mode.frequency) / (bw / 2 + 1e-10)) ** 2)
            amp *= (1 + gain)
        amplitudes[i] = amp
    return amplitudes / (np.max(amplitudes) + 1e-10)


def decay_rates(mode_freqs: np.ndarray,
                q_factors: np.ndarray) -> np.ndarray:
    tau = q_factors / (np.pi * mode_freqs + 1e-10)
    return tau


def sympathetic_resonance(exciter_freq: float,
                          string_freqs: np.ndarray,
                          bandwidth: float = 2.0) -> np.ndarray:
    response = np.zeros(len(string_freqs))
    for n in range(1, 9):
        harmonic = exciter_freq * n
        for i, sf in enumerate(string_freqs):
            for m in range(1, 9):
                sf_harm = sf * m
                if abs(harmonic - sf_harm) < bandwidth:
                    response[i] += 1.0 / (n * m)
    return response / (np.max(response) + 1e-10)


def wolf_tone_frequency(body_mode_freq: float,
                        string_freq: float,
                        coupling: float = 0.1) -> float:
    delta = abs(body_mode_freq - string_freq)
    if delta < coupling * string_freq:
        return float(np.sqrt(body_mode_freq * string_freq))
    return 0.0


def radiation_directivity(frequency: float,
                          radius: float,
                          angles: np.ndarray,
                          c: float = 343.0) -> np.ndarray:
    k = 2 * np.pi * frequency / c
    ka = k * radius
    if ka < 0.1:
        return np.ones(len(angles))
    x = ka * np.sin(angles)
    pattern = np.where(np.abs(x) < 1e-10, 1.0,
                       2.0 * np.sin(x) / (x + 1e-20))  # simplified
    return np.abs(pattern) ** 2


def bridge_admittance(body: InstrumentBody,
                      freqs: np.ndarray) -> np.ndarray:
    admittance = np.zeros(len(freqs), dtype=complex)
    for mode in body.modes:
        omega_n = 2 * np.pi * mode.frequency
        omega = 2 * np.pi * freqs
        eta = 1.0 / (mode.q_factor + 1e-10)
        admittance += (1j * omega * mode.amplitude /
                       (omega_n ** 2 - omega ** 2 + 1j * eta * omega_n * omega))
    return admittance
