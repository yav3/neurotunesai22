"""Chromagram computation and Tonnetz projection from chroma vectors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True)
class Chromagram:
    chroma: np.ndarray
    times: np.ndarray
    sr: int
    hop: int


@dataclass(frozen=True)
class TonnetzCoords:
    fifths: np.ndarray
    minor_thirds: np.ndarray
    major_thirds: np.ndarray


def compute_chromagram(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    hop: int = 2048,
    n_chroma: int = 12,
    norm: str = "l2",
) -> Chromagram:
    n = len(signal)
    n_frames = max(1, (n - n_fft) // hop + 1)
    chroma = np.zeros((n_chroma, n_frames))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    window = np.hanning(n_fft)

    freq_to_chroma = np.zeros(len(freqs), dtype=int)
    for k in range(1, len(freqs)):
        if freqs[k] > 0:
            midi = 12.0 * np.log2(freqs[k] / 440.0) + 69.0
            freq_to_chroma[k] = int(round(midi)) % n_chroma

    for t in range(n_frames):
        start = t * hop
        frame = signal[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        for k in range(1, len(spectrum)):
            chroma[freq_to_chroma[k], t] += spectrum[k]

    chroma = _normalize_chroma(chroma, norm)
    times = np.arange(n_frames) * hop / sr
    return Chromagram(chroma=chroma, times=times, sr=sr, hop=hop)


def _normalize_chroma(chroma: np.ndarray, norm: str) -> np.ndarray:
    result = chroma.copy()
    for t in range(result.shape[1]):
        col = result[:, t]
        if norm == "l1":
            s = np.sum(np.abs(col))
            if s > 1e-15:
                result[:, t] = col / s
        elif norm == "l2":
            s = np.sqrt(np.sum(col ** 2))
            if s > 1e-15:
                result[:, t] = col / s
        elif norm == "max":
            m = np.max(np.abs(col))
            if m > 1e-15:
                result[:, t] = col / m
    return result


def hpcp(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    hop: int = 2048,
    ref_freq: float = 440.0,
    harmonics: int = 3,
    n_bins: int = 12,
) -> np.ndarray:
    n = len(signal)
    n_frames = max(1, (n - n_fft) // hop + 1)
    profile = np.zeros((n_bins, n_frames))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    window = np.hanning(n_fft)

    for t in range(n_frames):
        start = t * hop
        frame = signal[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spectrum = np.abs(np.fft.rfft(frame * window))

        for k in range(1, len(freqs)):
            if freqs[k] < 20 or freqs[k] > sr / 2:
                continue
            for h in range(1, harmonics + 1):
                f0 = freqs[k] / h
                if f0 < 20:
                    continue
                midi = 12.0 * np.log2(f0 / ref_freq) + 69.0
                pc = int(round(midi)) % n_bins
                weight = spectrum[k] / h
                profile[pc, t] += weight

    for t in range(n_frames):
        m = profile[:, t].max()
        if m > 1e-15:
            profile[:, t] /= m
    return profile


def chroma_to_tonnetz(chroma: np.ndarray) -> TonnetzCoords:
    n_chroma, n_frames = chroma.shape
    if n_chroma != 12:
        raise ValueError(f"Expected 12 pitch classes, got {n_chroma}")

    fifths = np.zeros(n_frames)
    minor_thirds = np.zeros(n_frames)
    major_thirds = np.zeros(n_frames)

    angles = np.arange(12) * 2 * np.pi / 12.0

    for t in range(n_frames):
        c = chroma[:, t]
        s = c.sum()
        if s < 1e-15:
            continue
        c_norm = c / s
        fifths_angle = angles * 7
        fifths[t] = float(np.dot(c_norm, np.cos(fifths_angle)))
        minor_thirds[t] = float(np.dot(c_norm, np.sin(angles * 3)))
        major_thirds[t] = float(np.dot(c_norm, np.cos(angles * 4)))

    return TonnetzCoords(
        fifths=fifths,
        minor_thirds=minor_thirds,
        major_thirds=major_thirds,
    )


def chroma_correlation(chroma: np.ndarray) -> np.ndarray:
    n_chroma = chroma.shape[0]
    corr = np.zeros((n_chroma, n_chroma))
    for i in range(n_chroma):
        for j in range(n_chroma):
            xi = chroma[i] - chroma[i].mean()
            xj = chroma[j] - chroma[j].mean()
            denom = np.sqrt(np.sum(xi ** 2) * np.sum(xj ** 2))
            if denom > 1e-15:
                corr[i, j] = float(np.dot(xi, xj) / denom)
    return corr


def key_profile_correlation(
    chroma_vector: np.ndarray,
) -> list[tuple[str, str, float]]:
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    results = []
    for shift in range(12):
        shifted = np.roll(chroma_vector, shift)
        for mode, profile in [("major", major_profile), ("minor", minor_profile)]:
            a = shifted - shifted.mean()
            b = profile - profile.mean()
            denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
            corr = float(np.dot(a, b) / denom) if denom > 1e-15 else 0.0
            root = PITCH_CLASSES[(12 - shift) % 12]
            results.append((root, mode, corr))
    results.sort(key=lambda x: x[2], reverse=True)
    return results
