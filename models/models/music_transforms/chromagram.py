"""Chromagram transforms: HPCP, chroma energy, key profile correlation.

Harmonic Pitch Class Profile (Gómez 2006) and Krumhansl-Schmuckler key-finding
from chromagram features. Pitch-class energy normalized across octaves.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class ChromagramResult:
    chroma: np.ndarray  # (12, n_frames)
    times: np.ndarray
    sr: int


class ChromagramExtractor:
    """STFT-based chromagram with harmonic weighting."""

    @staticmethod
    def stft_chromagram(
        signal: np.ndarray,
        sr: int = 22050,
        n_fft: int = 4096,
        hop_length: int = 512,
        n_harmonics: int = 3,
        fmin: float = 65.41,
    ) -> ChromagramResult:
        """Compute 12-bin chromagram via harmonic pitch class summation.

        Each frequency bin is mapped to its nearest pitch class, weighted
        by 1/harmonic_number for harmonics 1..n_harmonics.
        """
        n_frames = max(1, 1 + (len(signal) - n_fft) // hop_length)
        window = np.hanning(n_fft)
        chroma = np.zeros((12, n_frames))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        for t in range(n_frames):
            start = t * hop_length
            frame = signal[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            mag = np.abs(np.fft.rfft(frame * window))

            for h in range(1, n_harmonics + 1):
                weight = 1.0 / h
                for i, f in enumerate(freqs):
                    f_fundamental = f / h
                    if f_fundamental < fmin or f_fundamental < 1.0:
                        continue
                    midi = 12 * np.log2(f_fundamental / 440.0) + 69
                    pc = int(round(midi)) % 12
                    chroma[pc, t] += mag[i] * weight

        col_norm = chroma.max(axis=0, keepdims=True)
        col_norm[col_norm < 1e-12] = 1.0
        chroma /= col_norm

        times = np.arange(n_frames) * hop_length / sr
        return ChromagramResult(chroma=chroma, times=times, sr=sr)

    @staticmethod
    def chroma_energy(chroma: np.ndarray) -> np.ndarray:
        """Per-pitch-class energy across all frames."""
        return np.sum(chroma ** 2, axis=1)

    @staticmethod
    def chroma_centroid(chroma: np.ndarray) -> np.ndarray:
        """Spectral centroid within each frame's pitch-class distribution."""
        pcs = np.arange(12).reshape(12, 1)
        total = chroma.sum(axis=0, keepdims=True)
        total[total < 1e-12] = 1.0
        return (pcs * chroma / total).sum(axis=0)

    @staticmethod
    def chroma_contrast(chroma: np.ndarray, quantile: float = 0.2) -> np.ndarray:
        """Difference between top and bottom quantile energies per frame."""
        n_low = max(1, int(12 * quantile))
        contrast = np.zeros(chroma.shape[1])
        for t in range(chroma.shape[1]):
            sorted_vals = np.sort(chroma[:, t])
            contrast[t] = sorted_vals[-n_low:].mean() - sorted_vals[:n_low].mean()
        return contrast


class KeyProfile(str, Enum):
    KRUMHANSL = "krumhansl"
    TEMPERLEY = "temperley"
    AARDEN = "aarden"


KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

AARDEN_MAJOR = np.array([17.77, 0.15, 14.93, 0.16, 19.80, 11.36, 0.29, 22.06, 0.15, 8.15, 0.23, 4.95])
AARDEN_MINOR = np.array([18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 21.07, 7.49, 1.53, 0.92, 10.21])


@dataclass
class KeyEstimate:
    key: str  # e.g. "C major", "A minor"
    correlation: float
    pitch_class: int
    mode: str  # "major" or "minor"
    all_correlations: np.ndarray  # (24,) — 12 major + 12 minor


class KeyDetector:
    """Krumhansl-Schmuckler key-finding algorithm from chromagram."""

    @staticmethod
    def detect_key(
        chroma: np.ndarray,
        profile: KeyProfile = KeyProfile.KRUMHANSL,
    ) -> KeyEstimate:
        """Estimate musical key by correlating chroma distribution with key profiles.

        Parameters
        ----------
        chroma : (12, n_frames) chromagram
        profile : which key-profile weights to use

        Returns
        -------
        KeyEstimate with best key, correlation, and all 24 correlations
        """
        if profile == KeyProfile.TEMPERLEY:
            maj_prof, min_prof = TEMPERLEY_MAJOR, TEMPERLEY_MINOR
        elif profile == KeyProfile.AARDEN:
            maj_prof, min_prof = AARDEN_MAJOR, AARDEN_MINOR
        else:
            maj_prof, min_prof = KRUMHANSL_MAJOR, KRUMHANSL_MINOR

        chroma_sum = chroma.sum(axis=1)
        if chroma_sum.sum() < 1e-12:
            return KeyEstimate("C major", 0.0, 0, "major", np.zeros(24))

        correlations = np.zeros(24)
        for shift in range(12):
            rotated = np.roll(chroma_sum, -shift)
            r_maj = np.corrcoef(rotated, maj_prof)[0, 1] if np.std(rotated) > 1e-12 else 0.0
            r_min = np.corrcoef(rotated, min_prof)[0, 1] if np.std(rotated) > 1e-12 else 0.0
            correlations[shift] = r_maj
            correlations[12 + shift] = r_min

        best = int(np.argmax(correlations))
        if best < 12:
            return KeyEstimate(
                key=f"{PITCH_CLASSES[best]} major",
                correlation=float(correlations[best]),
                pitch_class=best,
                mode="major",
                all_correlations=correlations,
            )
        else:
            pc = best - 12
            return KeyEstimate(
                key=f"{PITCH_CLASSES[pc]} minor",
                correlation=float(correlations[best]),
                pitch_class=pc,
                mode="minor",
                all_correlations=correlations,
            )

    @staticmethod
    def key_strength(estimate: KeyEstimate) -> float:
        """Confidence: ratio of best correlation to runner-up."""
        sorted_corrs = np.sort(estimate.all_correlations)[::-1]
        if len(sorted_corrs) < 2 or abs(sorted_corrs[1]) < 1e-12:
            return float("inf") if estimate.correlation > 0 else 0.0
        return float(sorted_corrs[0] / max(sorted_corrs[1], 1e-12))

    @staticmethod
    def modulation_detection(
        chroma: np.ndarray,
        frame_size: int = 20,
        hop: int = 10,
        profile: KeyProfile = KeyProfile.KRUMHANSL,
    ) -> list[KeyEstimate]:
        """Detect key changes over time by windowed key estimation."""
        n_frames = chroma.shape[1]
        keys = []
        for start in range(0, n_frames - frame_size + 1, hop):
            window = chroma[:, start : start + frame_size]
            keys.append(KeyDetector.detect_key(window, profile))
        return keys
