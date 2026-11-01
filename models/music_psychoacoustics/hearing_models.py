"""Advanced hearing models: excitation patterns, auditory filters, specific loudness.

Implements Patterson/Moore (1986) auditory filterbank (ERB gammatone),
Glasberg & Moore (2002) specific loudness model, and Terhardt (1979)
virtual pitch extraction. Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence


@dataclass
class AuditoryFilter:
    centre_freq: float
    erb: float
    gain: float
    bandwidth_3db: float


@dataclass
class ExcitationPattern:
    frequencies: np.ndarray
    excitation_db: np.ndarray
    specific_loudness: np.ndarray
    total_loudness_sone: float


@dataclass
class VirtualPitch:
    frequency: float
    salience: float
    subharmonic_support: int
    harmonics_matched: List[int]


def erb_rate(f: float) -> float:
    return 21.4 * np.log10(0.00437 * f + 1.0)


def erb_rate_inverse(e: float) -> float:
    return (10.0 ** (e / 21.4) - 1.0) / 0.00437


def erb_bandwidth(f: float) -> float:
    return 24.7 * (0.00437 * f + 1.0)


def erb_filterbank_centres(low: float = 50.0, high: float = 16000.0,
                           n_filters: int = 64) -> np.ndarray:
    e_low = erb_rate(low)
    e_high = erb_rate(high)
    erb_points = np.linspace(e_low, e_high, n_filters)
    return np.array([erb_rate_inverse(e) for e in erb_points])


def gammatone_impulse_response(fc: float, sr: float = 44100.0,
                                duration: float = 0.05,
                                order: int = 4) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / sr)
    b = 1.019 * 2 * np.pi * erb_bandwidth(fc)
    env = t ** (order - 1) * np.exp(-b * t)
    carrier = np.cos(2 * np.pi * fc * t)
    ir = env * carrier
    norm = np.sqrt(np.sum(ir ** 2))
    if norm > 1e-12:
        ir /= norm
    return ir


def gammatone_filterbank(signal: np.ndarray, sr: float = 44100.0,
                          n_filters: int = 64,
                          low: float = 50.0, high: float = 16000.0
                          ) -> np.ndarray:
    centres = erb_filterbank_centres(low, high, n_filters)
    n = len(signal)
    output = np.zeros((n_filters, n))
    for i, fc in enumerate(centres):
        ir = gammatone_impulse_response(fc, sr)
        filtered = np.convolve(signal, ir, mode='full')[:n]
        output[i] = filtered
    return output


def excitation_pattern_from_spectrum(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    db_ref: float = 1e-5,
) -> ExcitationPattern:
    n_filters = 64
    centres = erb_filterbank_centres(50.0, min(frequencies[-1], 16000.0), n_filters)
    excitation = np.zeros(n_filters)
    for i, fc in enumerate(centres):
        bw = erb_bandwidth(fc)
        weights = np.exp(-0.5 * ((frequencies - fc) / (bw * 0.5)) ** 2)
        excitation[i] = np.sum(weights * magnitude ** 2)
    excitation_db = 10.0 * np.log10(excitation / (db_ref ** 2) + 1e-30)
    spec_loud = _specific_loudness_zwicker(excitation_db)
    total = float(np.sum(spec_loud) * (erb_rate(centres[-1]) - erb_rate(centres[0])) / n_filters)
    return ExcitationPattern(centres, excitation_db, spec_loud, max(total, 0.0))


def _specific_loudness_zwicker(excitation_db: np.ndarray) -> np.ndarray:
    threshold_quiet = 3.64 * (1000.0 / 1000.0) ** -0.8
    e_tq = 10.0 ** (threshold_quiet / 10.0)
    e = 10.0 ** (excitation_db / 10.0)
    ratio = e / (e_tq + 1e-30)
    ratio = np.maximum(ratio, 1e-10)
    spec_loud = 0.08 * (e_tq + 1e-10) ** 0.23 * ((ratio) ** 0.23 - 1.0)
    return np.maximum(spec_loud, 0.0)


def loudness_from_spectrum(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
) -> float:
    ep = excitation_pattern_from_spectrum(magnitude, frequencies)
    return ep.total_loudness_sone


def sone_to_phon(sone: float) -> float:
    if sone <= 0:
        return 0.0
    if sone >= 1.0:
        return 40.0 + 10.0 * np.log2(sone)
    return 40.0 * (sone + 0.0005) ** 0.35


def phon_to_sone(phon: float) -> float:
    if phon <= 0:
        return 0.0
    if phon >= 40:
        return 2.0 ** ((phon - 40.0) / 10.0)
    return (phon / 40.0) ** (1.0 / 0.35) - 0.0005


def a_weighting(frequencies: np.ndarray) -> np.ndarray:
    f2 = frequencies ** 2
    num = 12194.0 ** 2 * f2 ** 2
    denom = ((f2 + 20.6 ** 2) *
             np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) *
             (f2 + 12194.0 ** 2))
    denom = np.maximum(denom, 1e-30)
    ra = num / denom
    ra_ref = 12194.0 ** 2 * 1000.0 ** 4 / (
        (1000.0 ** 2 + 20.6 ** 2) *
        np.sqrt((1000.0 ** 2 + 107.7 ** 2) * (1000.0 ** 2 + 737.9 ** 2)) *
        (1000.0 ** 2 + 12194.0 ** 2))
    ra = ra / (ra_ref + 1e-30)
    return 20.0 * np.log10(np.maximum(ra, 1e-30))


def c_weighting(frequencies: np.ndarray) -> np.ndarray:
    f2 = frequencies ** 2
    num = 12194.0 ** 2 * f2
    denom = (f2 + 20.6 ** 2) * (f2 + 12194.0 ** 2)
    denom = np.maximum(denom, 1e-30)
    rc = num / denom
    rc_ref = 12194.0 ** 2 * 1000.0 ** 2 / (
        (1000.0 ** 2 + 20.6 ** 2) * (1000.0 ** 2 + 12194.0 ** 2))
    rc = rc / (rc_ref + 1e-30)
    return 20.0 * np.log10(np.maximum(rc, 1e-30))


def terhardt_virtual_pitch(
    partials: np.ndarray,
    amplitudes: np.ndarray,
    tolerance_cents: float = 50.0,
    max_harmonic: int = 8,
) -> List[VirtualPitch]:
    if len(partials) < 2:
        return []
    candidates = {}
    for i, f in enumerate(partials):
        for h in range(1, max_harmonic + 1):
            f0_candidate = f / h
            if f0_candidate < 30.0:
                continue
            key = round(1200.0 * np.log2(f0_candidate / 30.0))
            if key not in candidates:
                candidates[key] = {"f0": f0_candidate, "support": 0,
                                   "salience": 0.0, "harmonics": []}
            ratio = f / candidates[key]["f0"]
            nearest_h = round(ratio)
            if nearest_h < 1 or nearest_h > max_harmonic:
                continue
            deviation_cents = abs(1200.0 * np.log2(ratio / nearest_h + 1e-30))
            if deviation_cents < tolerance_cents:
                candidates[key]["support"] += 1
                weight = amplitudes[i] / (nearest_h ** 0.5)
                candidates[key]["salience"] += weight
                if nearest_h not in candidates[key]["harmonics"]:
                    candidates[key]["harmonics"].append(nearest_h)

    results = []
    for c in candidates.values():
        if c["support"] >= 2:
            results.append(VirtualPitch(
                frequency=c["f0"],
                salience=c["salience"],
                subharmonic_support=c["support"],
                harmonics_matched=sorted(c["harmonics"]),
            ))
    results.sort(key=lambda v: v.salience, reverse=True)
    return results[:10]


def missing_fundamental(partials: np.ndarray) -> float:
    if len(partials) < 2:
        return 0.0
    diffs = []
    for i in range(len(partials)):
        for j in range(i + 1, min(i + 4, len(partials))):
            diffs.append(partials[j] - partials[i])
    if not diffs:
        return 0.0
    from collections import Counter
    rounded = [round(d, -1) for d in diffs]
    counts = Counter(rounded)
    gcd_candidate = counts.most_common(1)[0][0]
    return float(gcd_candidate) if gcd_candidate > 20 else 0.0


def masking_threshold(
    signal_spectrum_db: np.ndarray,
    frequencies: np.ndarray,
    spreading_slope_lower: float = 27.0,
    spreading_slope_upper: float = -12.0,
) -> np.ndarray:
    n = len(frequencies)
    bark = 13.0 * np.arctan(0.00076 * frequencies) + 3.5 * np.arctan((frequencies / 7500.0) ** 2)
    mask = np.full(n, -100.0)
    for i in range(n):
        if signal_spectrum_db[i] < -80:
            continue
        for j in range(n):
            db = bark[j] - bark[i]
            if db < 0:
                spread = signal_spectrum_db[i] + spreading_slope_lower * db
            else:
                spread = signal_spectrum_db[i] + spreading_slope_upper * db
            mask[j] = max(mask[j], spread)
    return mask


def auditory_roughness(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
) -> float:
    n = len(frequencies)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            f_diff = abs(frequencies[i] - frequencies[j])
            f_avg = (frequencies[i] + frequencies[j]) / 2.0
            s = 0.24 * f_avg / (0.021 * f_avg + 19)
            x = f_diff / max(s, 1e-10)
            d = max(np.exp(-3.5 * x) - np.exp(-5.75 * x), 0.0)
            total += amplitudes[i] * amplitudes[j] * d
    return float(total)


def perceptual_spectral_distance(
    spec_a: np.ndarray,
    spec_b: np.ndarray,
    frequencies: np.ndarray,
) -> float:
    weights = 1.0 / (erb_bandwidth(frequencies) + 1e-10)
    weights /= weights.sum()
    db_a = 20.0 * np.log10(np.maximum(spec_a, 1e-10))
    db_b = 20.0 * np.log10(np.maximum(spec_b, 1e-10))
    diff = (db_a - db_b) ** 2
    return float(np.sqrt(np.sum(weights * diff)))
