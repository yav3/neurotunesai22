"""Sympathetic resonance and string coupling models."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class StringState:
    """State of a vibrating string."""
    fundamental_hz: float
    amplitude: float = 1.0
    damping: float = 0.001
    tension_newtons: float = 70.0
    mass_per_length: float = 0.001  # kg/m
    length_m: float = 0.65


def harmonic_series(fundamental: float, n_harmonics: int = 16) -> list[float]:
    """Return the first n harmonics of a fundamental frequency."""
    return [fundamental * (k + 1) for k in range(n_harmonics)]


def harmonic_amplitude(n: int, decay_rate: float = 1.0) -> float:
    """Amplitude of the nth harmonic (1-indexed) under power-law decay: A_n = 1/n^decay."""
    if n < 1:
        return 0.0
    return 1.0 / (n ** decay_rate)


def sympathetic_coupling(driver: StringState, resonator: StringState, tolerance_cents: float = 50.0) -> float:
    """Coupling strength between driver and resonator strings.

    Returns a value in [0, 1] based on how close any driver harmonic
    is to any resonator harmonic (within tolerance_cents).
    """
    driver_harmonics = harmonic_series(driver.fundamental_hz, 12)
    resonator_harmonics = harmonic_series(resonator.fundamental_hz, 12)

    max_coupling = 0.0
    for i, dh in enumerate(driver_harmonics):
        for j, rh in enumerate(resonator_harmonics):
            if dh < 1.0 or rh < 1.0:
                continue
            cents = abs(1200.0 * math.log2(dh / rh))
            if cents <= tolerance_cents:
                amp_d = harmonic_amplitude(i + 1)
                amp_r = harmonic_amplitude(j + 1)
                coupling = amp_d * amp_r * (1.0 - cents / tolerance_cents)
                max_coupling = max(max_coupling, coupling)
    return max_coupling


def resonance_spectrum(
    strings: list[StringState],
    excited_index: int,
    n_harmonics: int = 12,
) -> dict[int, float]:
    """Compute sympathetic response amplitudes for all strings when one is excited.

    Returns {string_index: response_amplitude} for each non-excited string.
    """
    driver = strings[excited_index]
    result: dict[int, float] = {}
    for i, s in enumerate(strings):
        if i == excited_index:
            continue
        c = sympathetic_coupling(driver, s)
        result[i] = c * driver.amplitude
    return result


def total_sympathetic_energy(strings: list[StringState], excited_index: int) -> float:
    """Total sympathetic energy transferred from the excited string to all others."""
    spectrum = resonance_spectrum(strings, excited_index)
    return sum(a ** 2 for a in spectrum.values())


def decay_time(string: StringState, threshold: float = 0.001) -> float:
    """Time in seconds for amplitude to decay below threshold (exponential model)."""
    if string.damping <= 0 or string.amplitude <= threshold:
        return 0.0
    return -math.log(threshold / string.amplitude) / string.damping


def inharmonicity_coefficient(string: StringState) -> float:
    """Inharmonicity coefficient B for a stiff string.

    B = (pi^2 * E * d^4) / (64 * T * L^2) simplified to a function of
    tension and mass_per_length (assuming steel with E*A/rho typical values).
    """
    if string.tension_newtons <= 0 or string.length_m <= 0:
        return 0.0
    stiffness_proxy = string.mass_per_length * 0.01
    b = (math.pi ** 2 * stiffness_proxy) / (
        64.0 * string.tension_newtons * string.length_m ** 2
    )
    return b


def stretched_partial(fundamental: float, n: int, B: float) -> float:
    """Frequency of the nth partial with inharmonicity coefficient B.

    f_n = n * f_1 * sqrt(1 + B * n^2)
    """
    return n * fundamental * math.sqrt(1.0 + B * n * n)


def beating_frequency(f1: float, f2: float) -> float:
    """Beat frequency between two close frequencies."""
    return abs(f1 - f2)


@dataclass
class ResonanceAnalysis:
    """Full resonance analysis for a set of strings."""
    string_count: int
    excited_index: int
    sympathetic_responses: dict[int, float]
    total_energy: float
    strongest_responder: int | None
    strongest_amplitude: float


def analyze_resonance(strings: list[StringState], excited_index: int) -> ResonanceAnalysis:
    """Full resonance analysis when exciting one string in a set."""
    responses = resonance_spectrum(strings, excited_index)
    energy = sum(a ** 2 for a in responses.values())

    strongest_idx = None
    strongest_amp = 0.0
    for idx, amp in responses.items():
        if amp > strongest_amp:
            strongest_amp = amp
            strongest_idx = idx

    return ResonanceAnalysis(
        string_count=len(strings),
        excited_index=excited_index,
        sympathetic_responses=responses,
        total_energy=energy,
        strongest_responder=strongest_idx,
        strongest_amplitude=strongest_amp,
    )


def wolf_tone_risk(string: StringState, body_resonance_hz: float, tolerance_cents: float = 30.0) -> float:
    """Risk factor [0,1] for wolf tone: how close any harmonic is to a body resonance."""
    harmonics = harmonic_series(string.fundamental_hz, 8)
    min_cents = float("inf")
    for h in harmonics:
        if h < 1.0:
            continue
        cents = abs(1200.0 * math.log2(h / body_resonance_hz))
        min_cents = min(min_cents, cents)
    if min_cents >= tolerance_cents:
        return 0.0
    return 1.0 - min_cents / tolerance_cents
