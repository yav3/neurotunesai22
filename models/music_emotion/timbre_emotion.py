"""Timbre-emotion mapping: spectral texture to affective dimensions.

Maps timbral descriptors (spectral centroid, flux, roughness, brightness,
warmth, attack sharpness) onto valence-arousal-dominance using perceptual
weighting. Based on McAdams et al. (1995) timbre space, Hailstone et al.
(2009) timbre-affect associations, and Eerola et al. (2012) feature-emotion
correlations. Pure numpy/scipy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence
from enum import Enum


class TimbreQuality(Enum):
    BRIGHT = "bright"
    DARK = "dark"
    WARM = "warm"
    HARSH = "harsh"
    SMOOTH = "smooth"
    ROUGH = "rough"
    THIN = "thin"
    FULL = "full"
    NASAL = "nasal"
    BREATHY = "breathy"


@dataclass
class TimbreDescriptor:
    spectral_centroid: float       # Hz — brightness axis
    spectral_spread: float         # Hz — bandwidth
    spectral_flux: float           # rate of spectral change
    spectral_rolloff: float        # Hz — high-frequency energy bound
    spectral_flatness: float       # 0=tonal, 1=noise
    roughness: float               # sensory dissonance estimate
    brightness: float              # normalised centroid, 0–1
    warmth: float                  # low-frequency energy ratio, 0–1
    attack_time: float             # seconds
    zero_crossing_rate: float      # per second


@dataclass
class TimbreEmotionMapping:
    valence: float
    arousal: float
    dominance: float
    qualities: List[TimbreQuality]
    confidence: float


EEROLA_WEIGHTS_VALENCE = np.array([
    -0.18,   # spectral_centroid (brighter → slightly less pleasant)
     0.05,   # spectral_spread
    -0.12,   # spectral_flux (changing timbre → less pleasant)
    -0.10,   # spectral_rolloff
    -0.08,   # spectral_flatness (noisier → less pleasant)
    -0.35,   # roughness (strong negative)
     0.15,   # brightness (mild positive after centroid control)
     0.28,   # warmth (strong positive)
    -0.22,   # attack_time (slower attack → calmer, slightly positive)
    -0.06,   # zero_crossing_rate
])

EEROLA_WEIGHTS_AROUSAL = np.array([
     0.30,   # spectral_centroid
     0.12,   # spectral_spread
     0.25,   # spectral_flux
     0.18,   # spectral_rolloff
     0.10,   # spectral_flatness
     0.22,   # roughness
     0.28,   # brightness
    -0.15,   # warmth
    -0.35,   # attack_time (fast attack → high arousal)
     0.15,   # zero_crossing_rate
])

EEROLA_WEIGHTS_DOMINANCE = np.array([
     0.20,   # spectral_centroid
     0.15,   # spectral_spread
     0.10,   # spectral_flux
     0.12,   # spectral_rolloff
     0.05,   # spectral_flatness
     0.18,   # roughness (rough → dominant)
     0.22,   # brightness
    -0.08,   # warmth
    -0.25,   # attack_time (fast attack → dominant)
     0.08,   # zero_crossing_rate
])


def descriptor_to_vector(desc: TimbreDescriptor) -> np.ndarray:
    return np.array([
        desc.spectral_centroid,
        desc.spectral_spread,
        desc.spectral_flux,
        desc.spectral_rolloff,
        desc.spectral_flatness,
        desc.roughness,
        desc.brightness,
        desc.warmth,
        desc.attack_time,
        desc.zero_crossing_rate,
    ])


def normalise_descriptor(vec: np.ndarray,
                          means: Optional[np.ndarray] = None,
                          stds: Optional[np.ndarray] = None) -> np.ndarray:
    if means is None:
        means = np.array([2000, 1500, 0.1, 4000, 0.3, 0.3, 0.5, 0.5, 0.05, 50])
    if stds is None:
        stds = np.array([1500, 1000, 0.08, 3000, 0.25, 0.25, 0.3, 0.3, 0.04, 40])
    stds = np.where(stds < 1e-10, 1.0, stds)
    return (vec - means) / stds


def timbre_to_valence(desc: TimbreDescriptor) -> float:
    vec = normalise_descriptor(descriptor_to_vector(desc))
    return float(np.clip(np.dot(EEROLA_WEIGHTS_VALENCE, vec), -1, 1))


def timbre_to_arousal(desc: TimbreDescriptor) -> float:
    vec = normalise_descriptor(descriptor_to_vector(desc))
    return float(np.clip(np.dot(EEROLA_WEIGHTS_AROUSAL, vec), -1, 1))


def timbre_to_dominance(desc: TimbreDescriptor) -> float:
    vec = normalise_descriptor(descriptor_to_vector(desc))
    return float(np.clip(np.dot(EEROLA_WEIGHTS_DOMINANCE, vec), -1, 1))


def classify_timbre_qualities(desc: TimbreDescriptor) -> List[TimbreQuality]:
    qualities = []
    if desc.brightness > 0.6:
        qualities.append(TimbreQuality.BRIGHT)
    elif desc.brightness < 0.3:
        qualities.append(TimbreQuality.DARK)
    if desc.warmth > 0.6:
        qualities.append(TimbreQuality.WARM)
    if desc.roughness > 0.5:
        qualities.append(TimbreQuality.HARSH)
    elif desc.roughness < 0.15:
        qualities.append(TimbreQuality.SMOOTH)
    if desc.spectral_spread < 800:
        qualities.append(TimbreQuality.THIN)
    elif desc.spectral_spread > 2500:
        qualities.append(TimbreQuality.FULL)
    if desc.spectral_flatness > 0.6:
        qualities.append(TimbreQuality.BREATHY)
    if desc.spectral_centroid > 1000 and desc.spectral_spread < 1200:
        qualities.append(TimbreQuality.NASAL)
    return qualities if qualities else [TimbreQuality.SMOOTH]


def timbre_emotion_map(desc: TimbreDescriptor) -> TimbreEmotionMapping:
    v = timbre_to_valence(desc)
    a = timbre_to_arousal(desc)
    d = timbre_to_dominance(desc)
    q = classify_timbre_qualities(desc)
    vec = normalise_descriptor(descriptor_to_vector(desc))
    mag = float(np.linalg.norm(vec))
    conf = float(np.clip(1.0 - np.exp(-0.3 * mag), 0.1, 0.95))
    return TimbreEmotionMapping(v, a, d, q, conf)


def timbre_emotion_trajectory(
    descriptors: Sequence[TimbreDescriptor],
    smooth_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(descriptors)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    vals = np.array([timbre_to_valence(d) for d in descriptors])
    aros = np.array([timbre_to_arousal(d) for d in descriptors])
    doms = np.array([timbre_to_dominance(d) for d in descriptors])
    if n >= smooth_window and smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        vals = np.convolve(vals, kernel, mode='same')
        aros = np.convolve(aros, kernel, mode='same')
        doms = np.convolve(doms, kernel, mode='same')
    return vals, aros, doms


def timbre_distance(a: TimbreDescriptor, b: TimbreDescriptor) -> float:
    va = normalise_descriptor(descriptor_to_vector(a))
    vb = normalise_descriptor(descriptor_to_vector(b))
    return float(np.linalg.norm(va - vb))


def pairwise_timbre_distance(descriptors: List[TimbreDescriptor]) -> np.ndarray:
    n = len(descriptors)
    vecs = np.array([normalise_descriptor(descriptor_to_vector(d)) for d in descriptors])
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(vecs[i] - vecs[j]))
            mat[i, j] = mat[j, i] = d
    return mat


def emotion_centroid_from_timbres(
    descriptors: List[TimbreDescriptor],
) -> Tuple[float, float, float]:
    if not descriptors:
        return 0.0, 0.0, 0.0
    vals = [timbre_to_valence(d) for d in descriptors]
    aros = [timbre_to_arousal(d) for d in descriptors]
    doms = [timbre_to_dominance(d) for d in descriptors]
    return float(np.mean(vals)), float(np.mean(aros)), float(np.mean(doms))


def timbre_emotion_variance(
    descriptors: List[TimbreDescriptor],
) -> Tuple[float, float, float]:
    if len(descriptors) < 2:
        return 0.0, 0.0, 0.0
    vals = [timbre_to_valence(d) for d in descriptors]
    aros = [timbre_to_arousal(d) for d in descriptors]
    doms = [timbre_to_dominance(d) for d in descriptors]
    return float(np.var(vals)), float(np.var(aros)), float(np.var(doms))


def spectral_centroid_from_magnitude(
    magnitude_spectrum: np.ndarray,
    frequencies: np.ndarray,
) -> float:
    total = magnitude_spectrum.sum()
    if total < 1e-10:
        return 0.0
    return float(np.dot(frequencies, magnitude_spectrum) / total)


def spectral_spread_from_magnitude(
    magnitude_spectrum: np.ndarray,
    frequencies: np.ndarray,
) -> float:
    centroid = spectral_centroid_from_magnitude(magnitude_spectrum, frequencies)
    total = magnitude_spectrum.sum()
    if total < 1e-10:
        return 0.0
    return float(np.sqrt(np.dot((frequencies - centroid) ** 2, magnitude_spectrum) / total))


def spectral_flux_from_frames(
    current: np.ndarray,
    previous: np.ndarray,
) -> float:
    diff = current - previous
    return float(np.sqrt(np.sum(diff[diff > 0] ** 2)))


def roughness_from_partials(
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
            s = 0.24 * f_avg / (0.021 * f_avg + 19)  # critical bandwidth (Plomp & Levelt)
            x = f_diff / max(s, 1e-10)
            d = np.exp(-3.5 * x) - np.exp(-5.75 * x)
            total += amplitudes[i] * amplitudes[j] * max(d, 0.0)
    return float(total)


def descriptor_from_spectrum(
    magnitude_spectrum: np.ndarray,
    frequencies: np.ndarray,
    previous_spectrum: Optional[np.ndarray] = None,
    attack_time: float = 0.05,
    sample_rate: float = 44100.0,
) -> TimbreDescriptor:
    centroid = spectral_centroid_from_magnitude(magnitude_spectrum, frequencies)
    spread = spectral_spread_from_magnitude(magnitude_spectrum, frequencies)
    flux = 0.0
    if previous_spectrum is not None:
        flux = spectral_flux_from_frames(magnitude_spectrum, previous_spectrum)
    total = magnitude_spectrum.sum()
    rolloff_threshold = 0.85 * total
    cumsum = np.cumsum(magnitude_spectrum)
    rolloff_idx = np.searchsorted(cumsum, rolloff_threshold)
    rolloff = frequencies[min(rolloff_idx, len(frequencies) - 1)]
    geo_mean = np.exp(np.mean(np.log(magnitude_spectrum + 1e-10)))
    arith_mean = np.mean(magnitude_spectrum)
    flatness = geo_mean / max(arith_mean, 1e-10)
    nyquist = sample_rate / 2.0
    brightness = centroid / max(nyquist, 1.0)
    low_mask = frequencies < 500
    warmth = magnitude_spectrum[low_mask].sum() / max(total, 1e-10) if low_mask.any() else 0.0
    zcr = 0.0
    rough = roughness_from_partials(
        frequencies[:min(20, len(frequencies))],
        magnitude_spectrum[:min(20, len(magnitude_spectrum))],
    )
    return TimbreDescriptor(
        spectral_centroid=centroid,
        spectral_spread=spread,
        spectral_flux=flux,
        spectral_rolloff=float(rolloff),
        spectral_flatness=float(np.clip(flatness, 0, 1)),
        roughness=float(np.clip(rough, 0, 1)),
        brightness=float(np.clip(brightness, 0, 1)),
        warmth=float(np.clip(warmth, 0, 1)),
        attack_time=attack_time,
        zero_crossing_rate=zcr,
    )
