"""Harmonic analysis: partial extraction, harmonic-to-noise ratio, formant estimation."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PartialSet:
    """A set of spectral partials (harmonics or inharmonic components)."""
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    f0: float


@dataclass
class HarmonicAnalysisResult:
    """Full harmonic analysis of a single frame."""
    partials: PartialSet
    hnr_db: float
    harmonic_energy: float
    noise_energy: float
    n_harmonics: int


def extract_partials(
    magnitude: np.ndarray,
    phase: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    n_harmonics: int = 20,
    tolerance_cents: float = 50.0,
) -> PartialSet:
    """Extract harmonic partials from a magnitude spectrum.

    For each expected harmonic (k*f0), find the nearest spectral peak
    within tolerance_cents.
    """
    partial_freqs = []
    partial_amps = []
    partial_phases = []

    for k in range(1, n_harmonics + 1):
        expected = k * f0
        if expected <= 0:
            continue
        cents_range = tolerance_cents / 1200.0
        lo = expected * 2.0 ** (-cents_range)
        hi = expected * 2.0 ** (cents_range)

        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            continue

        region_mag = magnitude[mask]
        region_freq = freqs[mask]
        region_phase = phase[mask]

        peak_idx = np.argmax(region_mag)
        partial_freqs.append(float(region_freq[peak_idx]))
        partial_amps.append(float(region_mag[peak_idx]))
        partial_phases.append(float(region_phase[peak_idx]))

    return PartialSet(
        frequencies=np.array(partial_freqs),
        amplitudes=np.array(partial_amps),
        phases=np.array(partial_phases),
        f0=f0,
    )


def harmonic_to_noise_ratio(
    magnitude: np.ndarray,
    partials: PartialSet,
    freqs: np.ndarray,
    bandwidth_hz: float = 20.0,
) -> float:
    """Compute harmonic-to-noise ratio in dB.

    Harmonic energy is summed around each partial; the rest is noise.
    """
    total_energy = float(np.sum(magnitude ** 2))
    if total_energy < 1e-20:
        return 0.0

    harmonic_mask = np.zeros(len(freqs), dtype=bool)
    for f in partials.frequencies:
        harmonic_mask |= (np.abs(freqs - f) <= bandwidth_hz)

    harmonic_energy = float(np.sum(magnitude[harmonic_mask] ** 2))
    noise_energy = total_energy - harmonic_energy

    if noise_energy < 1e-20:
        return 60.0
    return float(10.0 * np.log10(harmonic_energy / noise_energy))


def spectral_envelope(
    magnitude: np.ndarray,
    order: int = 30,
) -> np.ndarray:
    """Estimate spectral envelope via cepstral smoothing (liftering).

    Low-quefrency cepstral coefficients define the spectral envelope.
    """
    log_mag = np.log(np.maximum(magnitude, 1e-20))
    cepstrum = np.fft.irfft(log_mag)
    liftered = np.zeros_like(cepstrum)
    liftered[:order] = cepstrum[:order]
    if len(cepstrum) > order:
        liftered[-order + 1 :] = cepstrum[-order + 1 :]
    envelope = np.fft.rfft(liftered).real
    return np.exp(envelope[: len(magnitude)])


def estimate_formants(
    envelope: np.ndarray,
    freqs: np.ndarray,
    n_formants: int = 4,
    min_freq: float = 100.0,
    max_freq: float = 5000.0,
) -> list[tuple[float, float]]:
    """Estimate formant frequencies and amplitudes from a spectral envelope.

    Returns list of (frequency, amplitude) tuples.
    """
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    env_region = envelope[mask]
    freq_region = freqs[mask]

    if len(env_region) < 3:
        return []

    peaks = []
    for i in range(1, len(env_region) - 1):
        if env_region[i] > env_region[i - 1] and env_region[i] > env_region[i + 1]:
            peaks.append((float(freq_region[i]), float(env_region[i])))

    peaks.sort(key=lambda x: -x[1])
    formants = peaks[:n_formants]
    formants.sort(key=lambda x: x[0])
    return formants


def harmonic_deviation(partials: PartialSet) -> float:
    """Mean log-amplitude deviation from a smooth spectral envelope.

    Measures how much individual partials deviate from the expected
    roll-off, indicating timbral irregularity.
    """
    if len(partials.amplitudes) < 3:
        return 0.0
    log_amps = np.log(np.maximum(partials.amplitudes, 1e-20))
    x = np.arange(len(log_amps), dtype=float)
    coeffs = np.polyfit(x, log_amps, 1)
    trend = np.polyval(coeffs, x)
    deviation = np.mean(np.abs(log_amps - trend))
    return float(deviation)


def analyse_harmonics(
    magnitude: np.ndarray,
    phase: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    n_harmonics: int = 20,
) -> HarmonicAnalysisResult:
    """Full harmonic analysis of a single spectral frame."""
    partials = extract_partials(magnitude, phase, freqs, f0, n_harmonics)
    hnr = harmonic_to_noise_ratio(magnitude, partials, freqs)

    total = float(np.sum(magnitude ** 2))
    harmonic_e = float(np.sum(partials.amplitudes ** 2))
    noise_e = max(0.0, total - harmonic_e)

    return HarmonicAnalysisResult(
        partials=partials,
        hnr_db=hnr,
        harmonic_energy=harmonic_e,
        noise_energy=noise_e,
        n_harmonics=len(partials.frequencies),
    )
