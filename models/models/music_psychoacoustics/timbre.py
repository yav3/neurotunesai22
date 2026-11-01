"""Timbre perception: multidimensional timbre space, spectral and temporal descriptors."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class TimbreDescriptors:
    """Multidimensional timbre descriptor set."""
    spectral_centroid: float
    spectral_spread: float
    spectral_flux: float
    spectral_rolloff: float
    spectral_flatness: float
    brightness: float
    attack_time: float
    log_attack_time: float
    temporal_centroid: float
    zero_crossing_rate: float


def spectral_centroid(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    total = np.sum(magnitude)
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * magnitude) / total)


def spectral_spread(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    sc = spectral_centroid(magnitude, freqs)
    total = np.sum(magnitude)
    if total < 1e-10:
        return 0.0
    return float(np.sqrt(np.sum(magnitude * (freqs - sc) ** 2) / total))


def spectral_flux(mag_current: np.ndarray, mag_previous: np.ndarray) -> float:
    diff = mag_current - mag_previous
    return float(np.sqrt(np.sum(diff ** 2)))


def spectral_rolloff(magnitude: np.ndarray, freqs: np.ndarray, percentile: float = 0.85) -> float:
    total = np.sum(magnitude)
    if total < 1e-10:
        return 0.0
    cumsum = np.cumsum(magnitude)
    idx = np.searchsorted(cumsum, percentile * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def spectral_flatness(magnitude: np.ndarray) -> float:
    mag = np.maximum(magnitude, 1e-10)
    geo_mean = np.exp(np.mean(np.log(mag)))
    arith_mean = np.mean(mag)
    if arith_mean < 1e-10:
        return 0.0
    return float(geo_mean / arith_mean)


def brightness(magnitude: np.ndarray, freqs: np.ndarray, cutoff: float = 1500.0) -> float:
    total = np.sum(magnitude)
    if total < 1e-10:
        return 0.0
    high = np.sum(magnitude[freqs >= cutoff])
    return float(high / total)


def attack_time(envelope: np.ndarray, sr: float = 22050.0, threshold: float = 0.9) -> float:
    if len(envelope) == 0:
        return 0.0
    peak = np.max(envelope)
    if peak < 1e-10:
        return 0.0
    norm = envelope / peak
    onset_idx = 0
    for i in range(len(norm)):
        if norm[i] > 0.01:
            onset_idx = i
            break
    peak_idx = int(np.argmax(norm >= threshold))
    if peak_idx <= onset_idx:
        peak_idx = int(np.argmax(envelope))
    return float((peak_idx - onset_idx) / sr)


def log_attack_time(envelope: np.ndarray, sr: float = 22050.0) -> float:
    at = attack_time(envelope, sr)
    if at <= 0:
        return -5.0
    return float(np.log10(at))


def temporal_centroid(envelope: np.ndarray, sr: float = 22050.0) -> float:
    total = np.sum(envelope)
    if total < 1e-10:
        return 0.0
    times = np.arange(len(envelope)) / sr
    return float(np.sum(times * envelope) / total)


def zero_crossing_rate(signal: np.ndarray) -> float:
    if len(signal) < 2:
        return 0.0
    signs = np.sign(signal)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (len(signal) - 1))


def compute_timbre_descriptors(
    signal: np.ndarray,
    sr: float = 22050.0,
    n_fft: int = 2048,
) -> TimbreDescriptors:
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))

    window = np.hanning(n_fft)
    mid = len(signal) // 2
    start = max(0, mid - n_fft // 2)
    frame = signal[start:start + n_fft] * window
    mag = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # previous frame for flux
    prev_start = max(0, start - n_fft // 4)
    prev_frame = signal[prev_start:prev_start + n_fft]
    if len(prev_frame) < n_fft:
        prev_frame = np.pad(prev_frame, (0, n_fft - len(prev_frame)))
    prev_mag = np.abs(np.fft.rfft(prev_frame * window))

    env = np.abs(signal)

    return TimbreDescriptors(
        spectral_centroid=spectral_centroid(mag, freqs),
        spectral_spread=spectral_spread(mag, freqs),
        spectral_flux=spectral_flux(mag, prev_mag),
        spectral_rolloff=spectral_rolloff(mag, freqs),
        spectral_flatness=spectral_flatness(mag),
        brightness=brightness(mag, freqs),
        attack_time=attack_time(env, sr),
        log_attack_time=log_attack_time(env, sr),
        temporal_centroid=temporal_centroid(env, sr),
        zero_crossing_rate=zero_crossing_rate(signal),
    )


class TimbreSpace:
    """Map sounds into a multidimensional timbre space."""

    def __init__(self, n_dimensions: int = 3) -> None:
        self.n_dimensions = n_dimensions
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._components: np.ndarray | None = None

    def _descriptors_to_vector(self, d: TimbreDescriptors) -> np.ndarray:
        return np.array([
            d.spectral_centroid,
            d.spectral_spread,
            d.spectral_flux,
            d.spectral_rolloff,
            d.spectral_flatness,
            d.brightness,
            d.log_attack_time,
            d.temporal_centroid,
            d.zero_crossing_rate,
        ])

    def fit(self, descriptors: list[TimbreDescriptors]) -> None:
        if len(descriptors) < 2:
            return
        X = np.array([self._descriptors_to_vector(d) for d in descriptors])
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std = np.maximum(self._std, 1e-10)
        Z = (X - self._mean) / self._std

        cov = Z.T @ Z / (len(Z) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(-eigvals)
        k = min(self.n_dimensions, len(order))
        self._components = eigvecs[:, order[:k]]

    def transform(self, d: TimbreDescriptors) -> np.ndarray:
        v = self._descriptors_to_vector(d)
        if self._mean is None or self._components is None:
            return v[:self.n_dimensions]
        z = (v - self._mean) / self._std
        return z @ self._components

    def distance(self, a: TimbreDescriptors, b: TimbreDescriptors) -> float:
        pa = self.transform(a)
        pb = self.transform(b)
        return float(np.linalg.norm(pa - pb))
