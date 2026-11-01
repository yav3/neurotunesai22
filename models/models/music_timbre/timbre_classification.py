"""Timbre classification: feature extraction, rule-based classifier, instrument profiles.

Maps spectral features to perceptual timbre classes (bright/dark/warm/harsh etc.),
provides canonical instrument profiles, and supports morphing paths in feature space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class TimbreClass(Enum):
    BRIGHT = "bright"
    DARK = "dark"
    WARM = "warm"
    HARSH = "harsh"
    THIN = "thin"
    FULL = "full"
    NASAL = "nasal"
    BREATHY = "breathy"


@dataclass(frozen=True)
class TimbreFeatureVector:
    spectral_centroid: float
    spectral_spread: float
    spectral_flatness: float
    attack_time_ms: float
    decay_rate: float
    harmonic_ratio: float
    noise_ratio: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.spectral_centroid, self.spectral_spread,
            self.spectral_flatness, self.attack_time_ms,
            self.decay_rate, self.harmonic_ratio, self.noise_ratio,
        ])


def extract_timbre_features(
    spectrum: np.ndarray, sr: float,
) -> TimbreFeatureVector:
    """Compute TimbreFeatureVector from a magnitude spectrum."""
    n = len(spectrum)
    freqs = np.linspace(0, sr / 2, n)
    power = spectrum ** 2
    total = np.sum(power) + 1e-12
    centroid = float(np.sum(freqs * power) / total)
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total))
    geo_mean = np.exp(np.mean(np.log(power + 1e-12)))
    arith_mean = np.mean(power) + 1e-12
    flatness = float(geo_mean / arith_mean)
    peak_idx = np.argmax(spectrum)
    if peak_idx < n - 1:
        after_peak = spectrum[peak_idx:]
        half_idx = np.argmax(after_peak < spectrum[peak_idx] * 0.5)
        decay = float(half_idx / max(n, 1))
    else:
        decay = 0.5
    f0_idx = max(1, np.argmax(spectrum[:n // 4]))
    harmonics_power = 0.0
    for h in range(1, 9):
        hidx = f0_idx * h
        if hidx < n:
            lo = max(0, hidx - 2)
            hi = min(n, hidx + 3)
            harmonics_power += float(np.sum(power[lo:hi]))
    harm_ratio = min(1.0, harmonics_power / float(total))
    noise_ratio = max(0.0, 1.0 - harm_ratio)
    return TimbreFeatureVector(
        spectral_centroid=centroid,
        spectral_spread=spread,
        spectral_flatness=flatness,
        attack_time_ms=10.0,
        decay_rate=decay,
        harmonic_ratio=harm_ratio,
        noise_ratio=noise_ratio,
    )


def classify_timbre(features: TimbreFeatureVector) -> TimbreClass:
    """Rule-based timbre classification from feature vector."""
    if features.spectral_centroid > 3000 and features.harmonic_ratio > 0.7:
        return TimbreClass.BRIGHT
    if features.spectral_centroid < 800 and features.spectral_flatness < 0.3:
        return TimbreClass.DARK
    if features.spectral_centroid < 1500 and features.harmonic_ratio > 0.8:
        return TimbreClass.WARM
    if features.spectral_flatness > 0.6 and features.noise_ratio > 0.4:
        return TimbreClass.HARSH
    if features.spectral_spread < 500 and features.harmonic_ratio > 0.6:
        return TimbreClass.THIN
    if features.spectral_spread > 2000 and features.harmonic_ratio > 0.5:
        return TimbreClass.FULL
    if 1000 < features.spectral_centroid < 2500 and features.spectral_flatness < 0.2:
        return TimbreClass.NASAL
    if features.noise_ratio > 0.5:
        return TimbreClass.BREATHY
    return TimbreClass.WARM


FEATURE_WEIGHTS = np.array([1.0, 0.8, 0.6, 0.3, 0.5, 1.0, 0.7])


def timbre_distance(a: TimbreFeatureVector, b: TimbreFeatureVector) -> float:
    """Weighted Euclidean distance in timbre feature space."""
    va = a.to_array()
    vb = b.to_array()
    normed_a = va / (np.abs(va).max() + 1e-12)
    normed_b = vb / (np.abs(vb).max() + 1e-12)
    diff = (normed_a - normed_b) * FEATURE_WEIGHTS
    return float(np.sqrt(np.sum(diff ** 2)))


INSTRUMENT_PROFILES: dict[str, TimbreFeatureVector] = {
    "piano": TimbreFeatureVector(1800.0, 1200.0, 0.15, 5.0, 0.3, 0.85, 0.15),
    "violin": TimbreFeatureVector(2200.0, 1400.0, 0.10, 30.0, 0.1, 0.90, 0.10),
    "flute": TimbreFeatureVector(2800.0, 800.0, 0.25, 15.0, 0.2, 0.70, 0.30),
    "trumpet": TimbreFeatureVector(2600.0, 1600.0, 0.08, 8.0, 0.15, 0.92, 0.08),
    "guitar": TimbreFeatureVector(1600.0, 1100.0, 0.20, 3.0, 0.35, 0.80, 0.20),
    "clarinet": TimbreFeatureVector(1500.0, 900.0, 0.12, 20.0, 0.12, 0.88, 0.12),
    "cello": TimbreFeatureVector(1000.0, 800.0, 0.10, 40.0, 0.08, 0.92, 0.08),
    "oboe": TimbreFeatureVector(1900.0, 700.0, 0.09, 25.0, 0.10, 0.93, 0.07),
    "saxophone": TimbreFeatureVector(2000.0, 1300.0, 0.18, 12.0, 0.14, 0.78, 0.22),
    "organ": TimbreFeatureVector(1400.0, 1500.0, 0.05, 50.0, 0.02, 0.95, 0.05),
}


def instrument_timbre_profile(name: str) -> TimbreFeatureVector | None:
    """Return canonical timbre profile for a named instrument."""
    return INSTRUMENT_PROFILES.get(name.lower())


def timbre_morphing_path(
    a: TimbreFeatureVector,
    b: TimbreFeatureVector,
    n_steps: int = 10,
) -> list[TimbreFeatureVector]:
    """Linear interpolation path between two timbre feature vectors."""
    va = a.to_array()
    vb = b.to_array()
    path = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        v = va * (1.0 - t) + vb * t
        path.append(TimbreFeatureVector(
            spectral_centroid=float(v[0]),
            spectral_spread=float(v[1]),
            spectral_flatness=float(v[2]),
            attack_time_ms=float(v[3]),
            decay_rate=float(v[4]),
            harmonic_ratio=float(v[5]),
            noise_ratio=float(v[6]),
        ))
    return path


def nearest_instrument(features: TimbreFeatureVector) -> tuple[str, float]:
    """Find the closest canonical instrument profile."""
    best_name = "unknown"
    best_dist = float("inf")
    for name, profile in INSTRUMENT_PROFILES.items():
        d = timbre_distance(features, profile)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name, best_dist
