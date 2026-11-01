"""Onset detection: spectral flux, complex domain, high-frequency content.

Implements multiple onset detection functions (ODF), adaptive thresholding,
and onset selection for beat tracking and rhythm analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class OnsetMethod(Enum):
    SPECTRAL_FLUX = "spectral_flux"
    HIGH_FREQUENCY_CONTENT = "hfc"
    COMPLEX_DOMAIN = "complex_domain"
    ENERGY = "energy"
    PHASE_DEVIATION = "phase_deviation"


@dataclass(frozen=True)
class Onset:
    time_s: float
    strength: float
    method: OnsetMethod


def _stft_frames(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft)
    n_frames = max(0, (len(signal) - n_fft) // hop + 1)
    frames = np.empty((n_frames, n_fft // 2 + 1), dtype=complex)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + n_fft] * window
        frames[i] = np.fft.rfft(frame)
    return frames


def spectral_flux(signal: np.ndarray, sr: float = 44100.0,
                  n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    frames = _stft_frames(signal, n_fft, hop)
    if len(frames) < 2:
        return np.array([])
    mag = np.abs(frames)
    diff = mag[1:] - mag[:-1]
    flux = np.sum(np.maximum(diff, 0.0), axis=1)
    return flux


def high_frequency_content(signal: np.ndarray, sr: float = 44100.0,
                           n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    frames = _stft_frames(signal, n_fft, hop)
    if len(frames) == 0:
        return np.array([])
    mag = np.abs(frames)
    n_bins = mag.shape[1]
    weights = np.arange(n_bins, dtype=float)
    hfc = np.sum(mag ** 2 * weights[np.newaxis, :], axis=1)
    return hfc


def complex_domain_odf(signal: np.ndarray, sr: float = 44100.0,
                       n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    frames = _stft_frames(signal, n_fft, hop)
    if len(frames) < 3:
        return np.array([])
    mag = np.abs(frames)
    phase = np.angle(frames)
    odf = np.zeros(len(frames) - 2)
    for i in range(2, len(frames)):
        predicted_phase = 2.0 * phase[i - 1] - phase[i - 2]
        predicted = mag[i - 1] * np.exp(1j * predicted_phase)
        actual = frames[i]
        odf[i - 2] = np.sum(np.abs(actual - predicted))
    return odf


def energy_odf(signal: np.ndarray, sr: float = 44100.0,
               n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    n_frames = max(0, (len(signal) - n_fft) // hop + 1)
    if n_frames < 2:
        return np.array([])
    energies = np.empty(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + n_fft]
        energies[i] = np.sum(frame ** 2)
    diff = np.maximum(energies[1:] - energies[:-1], 0.0)
    return diff


def phase_deviation_odf(signal: np.ndarray, sr: float = 44100.0,
                        n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    frames = _stft_frames(signal, n_fft, hop)
    if len(frames) < 3:
        return np.array([])
    phase = np.angle(frames)
    odf = np.zeros(len(frames) - 2)
    for i in range(2, len(frames)):
        expected = 2.0 * phase[i - 1] - phase[i - 2]
        deviation = phase[i] - expected
        deviation = np.arctan2(np.sin(deviation), np.cos(deviation))
        odf[i - 2] = np.mean(np.abs(deviation))
    return odf


def adaptive_threshold(odf: np.ndarray, window: int = 5,
                       multiplier: float = 1.5, offset: float = 0.0) -> np.ndarray:
    if len(odf) == 0:
        return np.array([])
    half = window // 2
    threshold = np.empty(len(odf))
    for i in range(len(odf)):
        lo = max(0, i - half)
        hi = min(len(odf), i + half + 1)
        local_median = np.median(odf[lo:hi])
        threshold[i] = multiplier * local_median + offset
    return threshold


def pick_onsets(odf: np.ndarray, sr: float = 44100.0, hop: int = 512,
                threshold_multiplier: float = 1.5,
                min_interval_s: float = 0.05) -> list[Onset]:
    if len(odf) == 0:
        return []
    thresh = adaptive_threshold(odf, multiplier=threshold_multiplier)
    min_interval_frames = int(min_interval_s * sr / hop)
    onsets: list[Onset] = []
    last_onset_frame = -min_interval_frames - 1
    for i in range(1, len(odf) - 1):
        if odf[i] > thresh[i] and odf[i] >= odf[i - 1] and odf[i] >= odf[i + 1]:
            if i - last_onset_frame >= min_interval_frames:
                time_s = (i + 1) * hop / sr
                strength = float(odf[i])
                onsets.append(Onset(time_s, strength, OnsetMethod.SPECTRAL_FLUX))
                last_onset_frame = i
    return onsets


def compute_odf(signal: np.ndarray, method: OnsetMethod,
                sr: float = 44100.0, n_fft: int = 2048,
                hop: int = 512) -> np.ndarray:
    if method == OnsetMethod.SPECTRAL_FLUX:
        return spectral_flux(signal, sr, n_fft, hop)
    if method == OnsetMethod.HIGH_FREQUENCY_CONTENT:
        return high_frequency_content(signal, sr, n_fft, hop)
    if method == OnsetMethod.COMPLEX_DOMAIN:
        return complex_domain_odf(signal, sr, n_fft, hop)
    if method == OnsetMethod.ENERGY:
        return energy_odf(signal, sr, n_fft, hop)
    if method == OnsetMethod.PHASE_DEVIATION:
        return phase_deviation_odf(signal, sr, n_fft, hop)
    return spectral_flux(signal, sr, n_fft, hop)


@dataclass
class OnsetDetector:
    method: OnsetMethod = OnsetMethod.SPECTRAL_FLUX
    n_fft: int = 2048
    hop: int = 512
    threshold_multiplier: float = 1.5
    min_interval_s: float = 0.05

    def detect(self, signal: np.ndarray, sr: float = 44100.0) -> list[Onset]:
        odf = compute_odf(signal, self.method, sr, self.n_fft, self.hop)
        onsets = pick_onsets(odf, sr, self.hop, self.threshold_multiplier,
                            self.min_interval_s)
        for i, o in enumerate(onsets):
            onsets[i] = Onset(o.time_s, o.strength, self.method)
        return onsets

    def odf(self, signal: np.ndarray, sr: float = 44100.0) -> np.ndarray:
        return compute_odf(signal, self.method, sr, self.n_fft, self.hop)
