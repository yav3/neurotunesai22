"""Acoustic identification: content-based audio retrieval and identification.

Higher-level identification pipeline combining multiple fingerprinting
strategies with scoring, ranking, and confidence estimation.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class IdentificationResult:
    """Result of an acoustic identification query."""
    track_id: str
    confidence: float  # 0..1
    offset_seconds: float
    n_matching_landmarks: int
    time_coherence: float


@dataclass
class AcousticProfile:
    """Compact acoustic profile for fast pre-filtering."""
    track_id: str
    duration: float
    spectral_centroid_mean: float
    spectral_centroid_std: float
    rms_mean: float
    chroma_histogram: np.ndarray  # 12-bin normalized
    tempo_estimate: float


def compute_acoustic_profile(signal: np.ndarray, sr: float = 22050,
                             track_id: str = "") -> AcousticProfile:
    """Compute a compact acoustic profile for pre-filtering."""
    frame_size = 2048
    hop = 512
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    window = np.hanning(frame_size)
    centroids = []
    rms_vals = []
    chroma_hist = np.zeros(12)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        windowed = frame * window
        mag = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)
        total_mag = np.sum(mag) + 1e-10
        centroids.append(np.sum(freqs * mag) / total_mag)
        rms_vals.append(np.sqrt(np.mean(windowed ** 2)))
        for k in range(1, len(freqs)):
            if freqs[k] > 0:
                pc = int(round(12 * np.log2(freqs[k] / 440.0))) % 12
                chroma_hist[pc] += mag[k] ** 2
    chroma_hist /= (chroma_hist.sum() + 1e-10)
    centroids = np.array(centroids)
    acf = np.correlate(signal[:sr * 4] if len(signal) > sr * 4 else signal,
                       signal[:sr * 4] if len(signal) > sr * 4 else signal,
                       mode='full')
    mid = len(acf) // 2
    acf = acf[mid:]
    min_lag = int(sr / 200)  # 200 BPM
    max_lag = int(sr / 40)   # 40 BPM
    if max_lag < len(acf) and min_lag < max_lag:
        search_region = acf[min_lag:max_lag]
        best_lag = np.argmax(search_region) + min_lag
        tempo = 60.0 * sr / best_lag
    else:
        tempo = 120.0
    return AcousticProfile(
        track_id=track_id,
        duration=len(signal) / sr,
        spectral_centroid_mean=float(np.mean(centroids)),
        spectral_centroid_std=float(np.std(centroids)),
        rms_mean=float(np.mean(rms_vals)),
        chroma_histogram=chroma_hist,
        tempo_estimate=tempo,
    )


def profile_distance(a: AcousticProfile, b: AcousticProfile) -> float:
    """Distance between two acoustic profiles for pre-filtering.

    Weighted combination of centroid distance, chroma distance,
    tempo distance, and energy distance.
    """
    centroid_dist = abs(a.spectral_centroid_mean - b.spectral_centroid_mean)
    centroid_norm = centroid_dist / (max(a.spectral_centroid_mean,
                                        b.spectral_centroid_mean, 1.0))
    chroma_dist = 1.0 - np.dot(a.chroma_histogram, b.chroma_histogram) / (
        np.linalg.norm(a.chroma_histogram) * np.linalg.norm(b.chroma_histogram) + 1e-10)
    tempo_ratio = min(a.tempo_estimate, b.tempo_estimate) / (
        max(a.tempo_estimate, b.tempo_estimate) + 1e-10)
    tempo_dist = 1.0 - tempo_ratio
    rms_dist = abs(a.rms_mean - b.rms_mean) / (max(a.rms_mean, b.rms_mean, 1e-10))
    return 0.3 * centroid_norm + 0.35 * chroma_dist + 0.2 * tempo_dist + 0.15 * rms_dist


@dataclass
class IdentificationIndex:
    """Index for acoustic identification queries."""
    profiles: List[AcousticProfile] = field(default_factory=list)
    _landmarks: Dict[str, List[Tuple[int, int, float]]] = field(
        default_factory=dict, repr=False)

    def add_track(self, profile: AcousticProfile,
                  landmarks: Optional[List[Tuple[int, int, float]]] = None) -> None:
        self.profiles.append(profile)
        if landmarks:
            self._landmarks[profile.track_id] = landmarks

    def pre_filter(self, query_profile: AcousticProfile,
                   n_candidates: int = 10) -> List[AcousticProfile]:
        distances = [(profile_distance(query_profile, p), p)
                     for p in self.profiles]
        distances.sort(key=lambda x: x[0])
        return [p for _, p in distances[:n_candidates]]


def time_coherence_check(offsets: np.ndarray, tolerance: float = 0.5) -> float:
    """Check temporal coherence of landmark matches.

    For genuine matches, time offsets should cluster around a single value.
    Returns fraction of matches within tolerance of the mode offset.
    """
    if len(offsets) < 2:
        return 0.0
    hist_bins = int(max(10, len(offsets) / 2))
    counts, edges = np.histogram(offsets, bins=hist_bins)
    mode_idx = np.argmax(counts)
    mode_center = (edges[mode_idx] + edges[mode_idx + 1]) / 2
    coherent = np.sum(np.abs(offsets - mode_center) < tolerance)
    return float(coherent / len(offsets))


def confidence_score(n_matches: int, time_coherence: float,
                     query_duration: float) -> float:
    """Compute identification confidence from match statistics.

    Combines landmark match count, temporal coherence, and
    match density (matches per second of query).
    """
    density = n_matches / (query_duration + 1e-10)
    density_score = min(1.0, density / 5.0)
    match_score = min(1.0, n_matches / 20.0)
    return 0.4 * match_score + 0.4 * time_coherence + 0.2 * density_score


def spectral_contrast_features(signal: np.ndarray, sr: float = 22050,
                               n_bands: int = 7,
                               frame_size: int = 2048,
                               hop: int = 512) -> np.ndarray:
    """Spectral contrast: difference between peaks and valleys per band.

    Useful as a fingerprinting feature — captures timbral characteristics
    that are robust to noise and compression.
    """
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    window = np.hanning(frame_size)
    n_bins = frame_size // 2 + 1
    band_edges = np.logspace(np.log10(20), np.log10(sr / 2), n_bands + 1)
    freqs = np.linspace(0, sr / 2, n_bins)
    contrast = np.zeros((n_frames, n_bands))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        mag = np.abs(np.fft.rfft(frame * window))
        for b in range(n_bands):
            lo = np.searchsorted(freqs, band_edges[b])
            hi = np.searchsorted(freqs, band_edges[b + 1])
            if lo >= hi:
                continue
            band_mag = np.sort(mag[lo:hi])
            n_band = len(band_mag)
            alpha = max(1, int(0.2 * n_band))
            peak = np.mean(band_mag[-alpha:])
            valley = np.mean(band_mag[:alpha])
            contrast[i, b] = np.log(peak + 1e-10) - np.log(valley + 1e-10)
    return contrast
