"""Onset detection: identify note and event boundaries in audio.

Multiple onset detection functions (spectral flux, high-frequency content,
complex domain, superflux) plus adaptive thresholding, onset strength
curves, and beat-phase onset quantisation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class OnsetResult:
    times_s: np.ndarray
    strengths: np.ndarray
    method: str


def spectral_flux(magnitude: np.ndarray, rectify: bool = True) -> np.ndarray:
    diff = np.diff(magnitude, axis=1)
    if rectify:
        diff = np.maximum(diff, 0.0)
    return np.sum(diff, axis=0)


def high_frequency_content(magnitude: np.ndarray, sr: int,
                           n_fft: int) -> np.ndarray:
    n_bins, n_frames = magnitude.shape
    freqs = np.linspace(0, sr / 2, n_bins)
    weights = freqs ** 2
    return np.array([np.sum(weights * magnitude[:, t] ** 2)
                     for t in range(n_frames)])


def complex_domain_odf(stft_complex: np.ndarray) -> np.ndarray:
    n_bins, n_frames = stft_complex.shape
    odf = np.zeros(n_frames)
    for t in range(2, n_frames):
        predicted = 2 * stft_complex[:, t - 1] - stft_complex[:, t - 2]
        deviation = np.abs(stft_complex[:, t] - predicted)
        odf[t] = np.sum(deviation)
    return odf


def superflux(magnitude: np.ndarray, lag: int = 2,
              max_filter_size: int = 3) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d
    mag_max = maximum_filter1d(magnitude, size=max_filter_size, axis=0)
    n_frames = magnitude.shape[1]
    odf = np.zeros(n_frames)
    for t in range(lag, n_frames):
        diff = mag_max[:, t] - mag_max[:, t - lag]
        odf[t] = np.sum(np.maximum(diff, 0.0))
    return odf


def adaptive_threshold(odf: np.ndarray, window: int = 10,
                       delta: float = 0.1) -> np.ndarray:
    n = len(odf)
    threshold = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        threshold[i] = np.median(odf[lo:hi]) + delta * np.std(odf[lo:hi])
    return threshold


def pick_onsets(odf: np.ndarray, threshold: np.ndarray,
                hop_s: float, min_gap_s: float = 0.03) -> np.ndarray:
    above = odf > threshold
    peaks = []
    last = -min_gap_s / hop_s - 1
    for i in range(1, len(odf) - 1):
        if above[i] and odf[i] >= odf[i - 1] and odf[i] >= odf[i + 1]:
            if i - last >= min_gap_s / hop_s:
                peaks.append(i)
                last = i
    return np.array(peaks) * hop_s


def onset_strength_curve(magnitude: np.ndarray, sr: int,
                         hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
    flux = spectral_flux(magnitude)
    times = np.arange(len(flux)) * hop_length / sr
    flux_norm = flux / (np.max(flux) + 1e-10)
    return times, flux_norm


def onset_backtrack(onsets_s: np.ndarray, energy: np.ndarray,
                    hop_s: float) -> np.ndarray:
    corrected = np.copy(onsets_s)
    for i, t in enumerate(onsets_s):
        idx = int(t / hop_s)
        search_start = max(0, idx - 5)
        segment = energy[search_start:idx + 1]
        if len(segment) > 0:
            corrected[i] = (search_start + np.argmin(segment)) * hop_s
    return corrected


def quantise_to_grid(onsets_s: np.ndarray, bpm: float,
                     subdivisions: int = 4) -> np.ndarray:
    beat_dur = 60.0 / bpm
    grid_dur = beat_dur / subdivisions
    return np.round(onsets_s / grid_dur) * grid_dur


def onset_density(onsets_s: np.ndarray, window_s: float = 1.0,
                  total_dur_s: float = 0.0) -> np.ndarray:
    if len(onsets_s) == 0:
        return np.array([0.0])
    dur = max(total_dur_s, onsets_s[-1])
    n_windows = max(1, int(dur / window_s))
    density = np.zeros(n_windows)
    for t in onsets_s:
        w = min(int(t / window_s), n_windows - 1)
        density[w] += 1
    return density / window_s
