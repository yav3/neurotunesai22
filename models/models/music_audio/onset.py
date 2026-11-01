"""Onset detection: spectral flux, complex domain, HFC, adaptive thresholding, beat-aligned onsets."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class OnsetMethod(str, Enum):
    SPECTRAL_FLUX = "spectral_flux"
    HIGH_FREQUENCY_CONTENT = "hfc"
    COMPLEX_DOMAIN = "complex_domain"
    PHASE_DEVIATION = "phase_deviation"
    ENERGY = "energy"


@dataclass(frozen=True)
class Onset:
    frame: int
    time: float
    strength: float
    method: OnsetMethod


def _stft_magnitude(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    n_frames = 1 + (len(signal) - frame_size) // hop_size
    if n_frames <= 0:
        return np.zeros((0, frame_size // 2 + 1))
    window = np.hanning(frame_size)
    n_bins = frame_size // 2 + 1
    mag = np.zeros((n_frames, n_bins))
    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start:start + frame_size] * window
        spectrum = np.fft.rfft(frame)
        mag[i] = np.abs(spectrum)
    return mag


def _stft_complex(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    n_frames = 1 + (len(signal) - frame_size) // hop_size
    if n_frames <= 0:
        return np.zeros((0, frame_size // 2 + 1), dtype=complex)
    window = np.hanning(frame_size)
    n_bins = frame_size // 2 + 1
    spectra = np.zeros((n_frames, n_bins), dtype=complex)
    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start:start + frame_size] * window
        spectra[i] = np.fft.rfft(frame)
    return spectra


def spectral_flux(signal: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> np.ndarray:
    mag = _stft_magnitude(signal, frame_size, hop_size)
    if mag.shape[0] < 2:
        return np.zeros(max(mag.shape[0], 0))
    diff = np.diff(mag, axis=0)
    flux = np.sum(np.maximum(diff, 0) ** 2, axis=1)
    return np.concatenate([[0.0], flux])


def high_frequency_content(signal: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> np.ndarray:
    mag = _stft_magnitude(signal, frame_size, hop_size)
    if mag.shape[0] == 0:
        return np.array([])
    n_bins = mag.shape[1]
    weights = np.arange(1, n_bins + 1, dtype=np.float64)
    hfc = np.sum(mag ** 2 * weights[None, :], axis=1)
    return hfc


def complex_domain_odf(signal: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> np.ndarray:
    spectra = _stft_complex(signal, frame_size, hop_size)
    n = spectra.shape[0]
    if n < 3:
        return np.zeros(max(n, 0))
    mag = np.abs(spectra)
    phase = np.angle(spectra)
    odf = np.zeros(n)
    for i in range(2, n):
        predicted_phase = 2 * phase[i - 1] - phase[i - 2]
        predicted = mag[i - 1] * np.exp(1j * predicted_phase)
        deviation = np.abs(spectra[i] - predicted)
        odf[i] = np.sum(deviation)
    return odf


def phase_deviation_odf(signal: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> np.ndarray:
    spectra = _stft_complex(signal, frame_size, hop_size)
    n = spectra.shape[0]
    if n < 3:
        return np.zeros(max(n, 0))
    phase = np.angle(spectra)
    odf = np.zeros(n)
    for i in range(2, n):
        second_diff = phase[i] - 2 * phase[i - 1] + phase[i - 2]
        wrapped = np.abs(np.arctan2(np.sin(second_diff), np.cos(second_diff)))
        odf[i] = np.mean(wrapped)
    return odf


def energy_odf(signal: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> np.ndarray:
    n_frames = 1 + (len(signal) - frame_size) // hop_size
    if n_frames <= 0:
        return np.array([])
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start:start + frame_size]
        energy[i] = np.sum(frame ** 2)
    if n_frames < 2:
        return energy
    diff = np.diff(energy)
    return np.concatenate([[0.0], np.maximum(diff, 0)])


def adaptive_threshold(odf: np.ndarray, window: int = 7, delta: float = 0.0, lambda_: float = 1.0) -> np.ndarray:
    n = len(odf)
    if n == 0:
        return np.array([])
    threshold = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local_median = np.median(odf[lo:hi])
        local_mean = np.mean(odf[lo:hi])
        threshold[i] = delta + lambda_ * max(local_median, local_mean)
    return threshold


def pick_peaks(
    odf: np.ndarray,
    threshold: np.ndarray | None = None,
    min_distance: int = 3,
) -> list[int]:
    if threshold is None:
        threshold = adaptive_threshold(odf)
    peaks: list[int] = []
    for i in range(1, len(odf) - 1):
        if odf[i] <= threshold[i]:
            continue
        if odf[i] >= odf[i - 1] and odf[i] >= odf[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    return peaks


def detect_onsets(
    signal: np.ndarray,
    sr: int = 22050,
    method: OnsetMethod = OnsetMethod.SPECTRAL_FLUX,
    frame_size: int = 1024,
    hop_size: int = 512,
    threshold_window: int = 7,
    threshold_delta: float = 0.0,
    threshold_lambda: float = 1.0,
    min_distance: int = 3,
) -> list[Onset]:
    odf_fn = {
        OnsetMethod.SPECTRAL_FLUX: spectral_flux,
        OnsetMethod.HIGH_FREQUENCY_CONTENT: high_frequency_content,
        OnsetMethod.COMPLEX_DOMAIN: complex_domain_odf,
        OnsetMethod.PHASE_DEVIATION: phase_deviation_odf,
        OnsetMethod.ENERGY: energy_odf,
    }[method]
    odf = odf_fn(signal, frame_size, hop_size)
    if len(odf) == 0:
        return []
    thresh = adaptive_threshold(odf, threshold_window, threshold_delta, threshold_lambda)
    frames = pick_peaks(odf, thresh, min_distance)
    hop_time = hop_size / sr
    return [
        Onset(
            frame=f,
            time=f * hop_time,
            strength=float(odf[f]),
            method=method,
        )
        for f in frames
    ]


@dataclass
class OnsetFeatures:
    inter_onset_intervals: np.ndarray
    mean_ioi: float
    std_ioi: float
    regularity: float
    density: float


def compute_onset_features(onsets: Sequence[Onset], total_duration: float) -> OnsetFeatures:
    if len(onsets) < 2:
        return OnsetFeatures(
            inter_onset_intervals=np.array([]),
            mean_ioi=0.0,
            std_ioi=0.0,
            regularity=0.0,
            density=len(onsets) / max(total_duration, 1e-10),
        )
    times = np.array([o.time for o in onsets])
    ioi = np.diff(times)
    mean_ioi = float(np.mean(ioi))
    std_ioi = float(np.std(ioi))
    regularity = 1.0 - (std_ioi / max(mean_ioi, 1e-10))
    regularity = float(np.clip(regularity, 0.0, 1.0))
    density = len(onsets) / max(total_duration, 1e-10)
    return OnsetFeatures(
        inter_onset_intervals=ioi,
        mean_ioi=mean_ioi,
        std_ioi=std_ioi,
        regularity=regularity,
        density=density,
    )


def multi_method_onset_fusion(
    signal: np.ndarray,
    sr: int = 22050,
    methods: Sequence[OnsetMethod] | None = None,
    vote_threshold: int = 2,
    frame_size: int = 1024,
    hop_size: int = 512,
    alignment_tolerance: int = 2,
) -> list[Onset]:
    if methods is None:
        methods = [OnsetMethod.SPECTRAL_FLUX, OnsetMethod.HIGH_FREQUENCY_CONTENT, OnsetMethod.COMPLEX_DOMAIN]
    all_onsets: dict[OnsetMethod, list[Onset]] = {}
    for m in methods:
        all_onsets[m] = detect_onsets(signal, sr, m, frame_size, hop_size)
    n_frames = 1 + (len(signal) - frame_size) // hop_size
    if n_frames <= 0:
        return []
    votes = np.zeros(n_frames)
    strength_sum = np.zeros(n_frames)
    for m, onsets in all_onsets.items():
        for o in onsets:
            lo = max(0, o.frame - alignment_tolerance)
            hi = min(n_frames, o.frame + alignment_tolerance + 1)
            votes[lo:hi] += 1
            strength_sum[lo:hi] += o.strength
    fused: list[Onset] = []
    hop_time = hop_size / sr
    for i in range(1, n_frames - 1):
        if votes[i] < vote_threshold:
            continue
        if votes[i] >= votes[i - 1] and votes[i] >= votes[i + 1]:
            if not fused or (i - fused[-1].frame) >= 3:
                fused.append(Onset(
                    frame=i,
                    time=i * hop_time,
                    strength=float(strength_sum[i] / max(votes[i], 1)),
                    method=OnsetMethod.SPECTRAL_FLUX,
                ))
    return fused


def onset_to_beat_alignment(
    onsets: Sequence[Onset],
    beat_times: Sequence[float],
    tolerance: float = 0.05,
) -> dict[str, float]:
    if not onsets or not beat_times:
        return {"aligned_ratio": 0.0, "mean_offset": 0.0, "n_aligned": 0}
    beats = np.array(beat_times)
    n_aligned = 0
    offsets = []
    for o in onsets:
        diffs = np.abs(beats - o.time)
        min_diff = float(np.min(diffs))
        if min_diff <= tolerance:
            n_aligned += 1
            offsets.append(min_diff)
    ratio = n_aligned / len(onsets)
    mean_offset = float(np.mean(offsets)) if offsets else 0.0
    return {"aligned_ratio": ratio, "mean_offset": mean_offset, "n_aligned": n_aligned}
