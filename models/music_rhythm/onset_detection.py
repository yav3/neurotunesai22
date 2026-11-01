"""Onset detection and IOI analysis: spectral flux, HFC, complex-domain ODF,
peak picking with adaptive threshold, inter-onset interval statistics.

Provides the rhythmic onset layer between raw spectrograms and higher-level
beat/tempo analysis.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class SpectralFluxResult:
    """Result of spectral flux computation."""
    times: np.ndarray
    flux: np.ndarray
    peaks: np.ndarray


def spectral_flux(stft_mag: np.ndarray, hop_length: int = 512,
                  sr: int = 22050) -> SpectralFluxResult:
    """Half-wave rectified spectral flux from magnitude STFT.

    Parameters
    ----------
    stft_mag : (n_freq, n_frames) magnitude spectrogram
    hop_length : samples between frames
    sr : sample rate
    """
    n_frames = stft_mag.shape[1]
    flux = np.zeros(n_frames)
    for t in range(1, n_frames):
        diff = stft_mag[:, t] - stft_mag[:, t - 1]
        flux[t] = float(np.sum(np.maximum(diff, 0.0)))
    times = np.arange(n_frames) * hop_length / sr
    peaks = pick_peaks(flux)
    return SpectralFluxResult(times=times, flux=flux, peaks=peaks)


def high_frequency_content(stft_mag: np.ndarray, hop_length: int = 512,
                           sr: int = 22050) -> np.ndarray:
    """HFC onset detection function: sum(|X(k,t)|^2 * k)."""
    n_freq, n_frames = stft_mag.shape
    k = np.arange(n_freq).reshape(-1, 1)
    hfc = np.sum(stft_mag ** 2 * k, axis=0)
    return hfc


def complex_domain(stft: np.ndarray, hop_length: int = 512,
                   sr: int = 22050) -> np.ndarray:
    """Complex-domain onset detection: deviation from expected magnitude+phase.

    Parameters
    ----------
    stft : complex STFT (n_freq, n_frames)
    """
    n_freq, n_frames = stft.shape
    odf = np.zeros(n_frames)
    mag = np.abs(stft)
    phase = np.angle(stft)
    for t in range(2, n_frames):
        expected_phase = 2 * phase[:, t - 1] - phase[:, t - 2]
        expected = mag[:, t - 1] * np.exp(1j * expected_phase)
        deviation = np.abs(stft[:, t] - expected)
        odf[t] = float(np.sum(deviation))
    return odf


def adaptive_threshold(odf: np.ndarray, window: int = 10,
                       delta: float = 0.1) -> np.ndarray:
    """Median + delta adaptive threshold curve."""
    n = len(odf)
    threshold = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        threshold[i] = np.median(odf[lo:hi]) + delta
    return threshold


def pick_peaks(odf: np.ndarray, threshold: float = 0.5,
               pre_max: int = 3, post_max: int = 3,
               pre_avg: int = 3, post_avg: int = 3,
               delta: float = 0.0) -> np.ndarray:
    """Peak picking with local maximum and threshold constraints.

    Returns array of peak frame indices.
    """
    n = len(odf)
    peaks = []
    for i in range(pre_max, n - post_max):
        lo_max = max(0, i - pre_max)
        hi_max = min(n, i + post_max + 1)
        if odf[i] < np.max(odf[lo_max:hi_max]):
            continue
        lo_avg = max(0, i - pre_avg)
        hi_avg = min(n, i + post_avg + 1)
        local_avg = np.mean(odf[lo_avg:hi_avg])
        if odf[i] >= local_avg + delta and odf[i] >= threshold:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def onset_strength(stft_mag: np.ndarray, hop_length: int = 512,
                   sr: int = 22050, n_mels: int = 128) -> np.ndarray:
    """Mel-weighted spectral flux (approximate librosa.onset.onset_strength).

    Applies triangular mel filterbank before computing spectral flux.
    """
    n_freq, n_frames = stft_mag.shape
    mel_freqs = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), n_mels + 2)
    mel_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
    fft_freqs = np.linspace(0, sr / 2, n_freq)
    filterbank = np.zeros((n_mels, n_freq))
    for m in range(n_mels):
        lo, center, hi = mel_freqs[m], mel_freqs[m + 1], mel_freqs[m + 2]
        for k in range(n_freq):
            if lo <= fft_freqs[k] <= center and center > lo:
                filterbank[m, k] = (fft_freqs[k] - lo) / (center - lo)
            elif center < fft_freqs[k] <= hi and hi > center:
                filterbank[m, k] = (hi - fft_freqs[k]) / (hi - center)
    mel_spec = filterbank @ stft_mag
    strength = np.zeros(n_frames)
    for t in range(1, n_frames):
        diff = mel_spec[:, t] - mel_spec[:, t - 1]
        strength[t] = float(np.sum(np.maximum(diff, 0.0)))
    return strength


def inter_onset_intervals(onsets: np.ndarray) -> np.ndarray:
    """Compute inter-onset intervals from onset times."""
    if len(onsets) < 2:
        return np.array([])
    return np.diff(onsets)


def ioi_histogram(iois: np.ndarray,
                  n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of inter-onset intervals."""
    if len(iois) == 0:
        return np.zeros(n_bins), np.zeros(n_bins + 1)
    counts, edges = np.histogram(iois, bins=n_bins)
    return counts.astype(float), edges


def ioi_regularity(iois: np.ndarray) -> float:
    """Coefficient of variation of IOIs (lower = more regular)."""
    if len(iois) < 2:
        return 0.0
    mean = np.mean(iois)
    if mean < 1e-15:
        return 0.0
    return float(np.std(iois) / mean)


def tempo_from_onsets(onsets: np.ndarray, sr: int = 1) -> float:
    """Estimate BPM from median IOI.

    If onsets are in seconds, pass sr=1.
    If onsets are in samples, pass the sample rate.
    """
    iois = inter_onset_intervals(onsets)
    if len(iois) == 0:
        return 0.0
    median_ioi = float(np.median(iois))
    if sr > 1:
        median_ioi /= sr
    if median_ioi < 1e-10:
        return 0.0
    return 60.0 / median_ioi


def onset_density(onsets: np.ndarray, duration: float) -> float:
    """Onsets per second."""
    if duration <= 0:
        return 0.0
    return len(onsets) / duration


def onset_clustering(onsets: np.ndarray, n_clusters: int = 4,
                     window: int = 5, seed: int = 42) -> np.ndarray:
    """Cluster onsets by local IOI pattern using k-means.

    Each onset is represented by a feature vector of its surrounding IOIs.
    Returns cluster labels per onset.
    """
    iois = inter_onset_intervals(onsets)
    n = len(onsets)
    if n < n_clusters or len(iois) < window:
        return np.zeros(n, dtype=int)
    features = []
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(len(iois), lo + window)
        lo = max(0, hi - window)
        feat = np.zeros(window)
        chunk = iois[lo:hi]
        feat[:len(chunk)] = chunk
        features.append(feat)
    X = np.array(features)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    X = X / norms
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_clusters, n), replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(n_clusters):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
    return labels


def onset_synchrony(onsets_a: np.ndarray, onsets_b: np.ndarray,
                    tolerance: float = 0.05) -> float:
    """Fraction of onsets in A that have a match in B within tolerance."""
    if len(onsets_a) == 0:
        return 0.0
    matches = 0
    for t in onsets_a:
        if np.any(np.abs(onsets_b - t) <= tolerance):
            matches += 1
    return matches / len(onsets_a)
