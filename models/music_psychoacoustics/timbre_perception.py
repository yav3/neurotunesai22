"""Timbre perception: perceptual timbre descriptors and distance.

Models timbre as a multidimensional perceptual space with descriptors
based on spectral, temporal, and harmonic features. Implements odd/even
harmonic ratio, inharmonicity, warmth, and roughness measures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TimbreDescriptors:
    brightness: float = 0.0
    warmth: float = 0.0
    roughness: float = 0.0
    odd_even: float = 1.0
    inharmonicity_val: float = 0.0
    attack_sharpness: float = 0.0
    spectral_centroid: float = 0.0
    spectral_flux: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.brightness, self.warmth, self.roughness,
            self.odd_even, self.inharmonicity_val, self.attack_sharpness,
            self.spectral_centroid, self.spectral_flux,
        ])


def odd_even_ratio(spectrum: np.ndarray, f0_bin: int) -> float:
    if f0_bin < 1 or len(spectrum) < 2 * f0_bin:
        return 1.0
    odd_sum = 0.0
    even_sum = 0.0
    for h in range(1, len(spectrum) // f0_bin + 1):
        idx = h * f0_bin
        if idx >= len(spectrum):
            break
        power = abs(spectrum[idx]) ** 2
        if h % 2 == 1:
            odd_sum += power
        else:
            even_sum += power
    if even_sum < 1e-12:
        return float("inf") if odd_sum > 0 else 1.0
    return float(odd_sum / even_sum)


def inharmonicity(partials_hz: np.ndarray, f0: float) -> float:
    if f0 <= 0 or len(partials_hz) < 2:
        return 0.0
    deviations = []
    for i, freq in enumerate(partials_hz):
        expected = f0 * (i + 1)
        if expected > 0:
            dev = abs(freq - expected) / expected
            deviations.append(dev)
    if not deviations:
        return 0.0
    return float(np.mean(deviations))


def warmth(spectrum: np.ndarray, freqs: np.ndarray, cutoff: float = 500.0) -> float:
    if len(spectrum) == 0 or len(freqs) == 0:
        return 0.0
    power = np.abs(spectrum) ** 2
    total = np.sum(power)
    if total < 1e-12:
        return 0.0
    low_mask = freqs <= cutoff
    low_power = np.sum(power[low_mask])
    return float(low_power / total)


def roughness_dissonance(f1: float, f2: float, a1: float = 1.0, a2: float = 1.0) -> float:
    s = abs(f2 - f1)
    if s < 1e-6:
        return 0.0
    f_min = min(f1, f2)
    if f_min < 1e-6:
        return 0.0
    s_star = 0.24 / (0.021 * f_min + 19)
    x = s / (f_min * s_star) if f_min * s_star > 0 else 0
    d = np.exp(-3.5 * x) - np.exp(-5.75 * x)
    return float(a1 * a2 * max(d, 0))


def total_roughness(freqs: np.ndarray, amplitudes: np.ndarray) -> float:
    n = len(freqs)
    if n < 2:
        return 0.0
    r = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r += roughness_dissonance(freqs[i], freqs[j], amplitudes[i], amplitudes[j])
    return float(r)


def analyze_timbre(spectrum: np.ndarray, freqs: np.ndarray,
                   f0: float = 0.0, sr: int = 22050) -> TimbreDescriptors:
    power = np.abs(spectrum) ** 2
    total = np.sum(power)

    if total < 1e-12:
        return TimbreDescriptors()

    centroid = float(np.sum(freqs * power) / total) if len(freqs) == len(power) else 0.0
    bright = float(np.sum(power[freqs > 3000]) / total) if np.any(freqs > 3000) else 0.0
    warm = warmth(spectrum, freqs)

    f0_bin = int(f0 * len(spectrum) / sr) if f0 > 0 and sr > 0 else 0
    oe = odd_even_ratio(spectrum, f0_bin) if f0_bin > 0 else 1.0

    return TimbreDescriptors(
        brightness=bright,
        warmth=warm,
        roughness=0.0,
        odd_even=min(oe, 100.0),
        inharmonicity_val=0.0,
        attack_sharpness=0.0,
        spectral_centroid=centroid,
        spectral_flux=0.0,
    )


def timbre_distance(a: TimbreDescriptors, b: TimbreDescriptors) -> float:
    va = a.to_vector()
    vb = b.to_vector()
    return float(np.linalg.norm(va - vb))
