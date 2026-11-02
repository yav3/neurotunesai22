"""Chromagram computation and chroma-based harmonic analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "ChromaConfig", "ChromaResult", "ChordTemplate",
    "compute_chromagram", "stft_to_chroma", "harmonic_chromagram",
    "chord_templates", "template_match_chords",
    "chroma_distance", "tonal_centroid",
    "tonnetz_transform", "chroma_cross_correlation",
    "key_strength_profile", "chroma_flux",
    "median_filter_chroma", "log_compression_chroma",
]


@dataclass
class ChromaConfig:
    sr: int = 22050
    n_fft: int = 4096
    hop_length: int = 512
    n_chroma: int = 12
    norm: str = "max"
    tuning: float = 0.0


@dataclass
class ChromaResult:
    chroma: np.ndarray
    times: np.ndarray
    sr: int


@dataclass
class ChordTemplate:
    name: str
    root: int
    template: np.ndarray


def compute_chromagram(
    audio: np.ndarray, sr: int = 22050, n_fft: int = 4096,
    hop_length: int = 512, n_chroma: int = 12,
) -> ChromaResult:
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames <= 0:
        return ChromaResult(np.zeros((n_chroma, 0)), np.array([]), sr)
    window = np.hanning(n_fft)
    chroma = np.zeros((n_chroma, n_frames))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    for t in range(n_frames):
        start = t * hop_length
        frame = audio[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        for k, f in enumerate(freqs):
            if f < 20 or f > sr / 2:
                continue
            midi = 69 + 12 * np.log2(f / 440.0)
            pc = int(round(midi)) % n_chroma
            chroma[pc, t] += spectrum[k] ** 2
    mx = chroma.max()
    if mx > 0:
        chroma /= mx
    times = np.arange(n_frames) * hop_length / sr
    return ChromaResult(chroma=chroma, times=times, sr=sr)


def stft_to_chroma(
    stft_mag: np.ndarray, freqs: np.ndarray, n_chroma: int = 12,
) -> np.ndarray:
    n_frames = stft_mag.shape[1]
    chroma = np.zeros((n_chroma, n_frames))
    for k, f in enumerate(freqs):
        if f < 20:
            continue
        midi = 69 + 12 * np.log2(max(f, 1e-9) / 440.0)
        pc = int(round(midi)) % n_chroma
        chroma[pc] += stft_mag[k]
    mx = chroma.max()
    return chroma / mx if mx > 0 else chroma


def harmonic_chromagram(
    audio: np.ndarray, sr: int = 22050, n_fft: int = 4096,
    hop_length: int = 512, n_harmonics: int = 5,
) -> np.ndarray:
    base = compute_chromagram(audio, sr, n_fft, hop_length)
    chroma = base.chroma.copy()
    for h in range(2, n_harmonics + 1):
        shift = int(round(12 * np.log2(h)))
        rolled = np.roll(chroma, -shift, axis=0)
        chroma += rolled / h
    mx = chroma.max()
    return chroma / mx if mx > 0 else chroma


def chord_templates() -> List[ChordTemplate]:
    templates = []
    names_major = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    for root in range(12):
        t_maj = np.zeros(12)
        t_maj[[root, (root + 4) % 12, (root + 7) % 12]] = 1.0
        templates.append(ChordTemplate(f"{names_major[root]}", root, t_maj / np.linalg.norm(t_maj)))
        t_min = np.zeros(12)
        t_min[[root, (root + 3) % 12, (root + 7) % 12]] = 1.0
        templates.append(ChordTemplate(f"{names_major[root]}m", root, t_min / np.linalg.norm(t_min)))
        t_dim = np.zeros(12)
        t_dim[[root, (root + 3) % 12, (root + 6) % 12]] = 1.0
        templates.append(ChordTemplate(f"{names_major[root]}dim", root, t_dim / np.linalg.norm(t_dim)))
        t_aug = np.zeros(12)
        t_aug[[root, (root + 4) % 12, (root + 8) % 12]] = 1.0
        templates.append(ChordTemplate(f"{names_major[root]}aug", root, t_aug / np.linalg.norm(t_aug)))
        t_7 = np.zeros(12)
        t_7[[root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12]] = 1.0
        templates.append(ChordTemplate(f"{names_major[root]}7", root, t_7 / np.linalg.norm(t_7)))
    return templates


def template_match_chords(
    chroma: np.ndarray, templates: Optional[List[ChordTemplate]] = None,
) -> List[Tuple[str, float]]:
    if templates is None:
        templates = chord_templates()
    n_frames = chroma.shape[1]
    results = []
    for t in range(n_frames):
        frame = chroma[:, t]
        fn = np.linalg.norm(frame)
        if fn < 1e-9:
            results.append(("N", 0.0))
            continue
        best_name, best_score = "N", 0.0
        for tmpl in templates:
            score = float(np.dot(frame / fn, tmpl.template))
            if score > best_score:
                best_score = score
                best_name = tmpl.name
        results.append((best_name, best_score))
    return results


def chroma_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    n1 = np.linalg.norm(c1)
    n2 = np.linalg.norm(c2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 1.0
    return 1.0 - float(np.dot(c1 / n1, c2 / n2))


def tonal_centroid(chroma_frame: np.ndarray) -> np.ndarray:
    angles = 2 * np.pi * np.arange(12) / 12
    r1 = np.array([np.sin(angles), np.cos(angles)])
    r2 = np.array([np.sin(2 * angles), np.cos(2 * angles)])
    r3 = np.array([np.sin(3 * angles), np.cos(3 * angles)])
    s = chroma_frame.sum()
    if s < 1e-9:
        return np.zeros(6)
    w = chroma_frame / s
    return np.concatenate([r1 @ w, r2 @ w, r3 @ w])


def tonnetz_transform(chroma: np.ndarray) -> np.ndarray:
    n_frames = chroma.shape[1]
    tonnetz = np.zeros((6, n_frames))
    for t in range(n_frames):
        tonnetz[:, t] = tonal_centroid(chroma[:, t])
    return tonnetz


def chroma_cross_correlation(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    corrs = np.zeros(12)
    for shift in range(12):
        rolled = np.roll(c2, shift, axis=0)
        n1 = np.linalg.norm(c1)
        n2 = np.linalg.norm(rolled)
        if n1 > 1e-9 and n2 > 1e-9:
            corrs[shift] = float(np.sum(c1 * rolled) / (n1 * n2))
    return corrs


def key_strength_profile(chroma: np.ndarray) -> np.ndarray:
    profile = chroma.mean(axis=1) if chroma.ndim == 2 else chroma
    strengths = np.zeros(24)
    krumhansl_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    krumhansl_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    for key in range(12):
        rotated = np.roll(profile, -key)
        strengths[key] = float(np.corrcoef(rotated, krumhansl_major)[0, 1])
        strengths[key + 12] = float(np.corrcoef(rotated, krumhansl_minor)[0, 1])
    return strengths


def chroma_flux(chroma: np.ndarray) -> np.ndarray:
    if chroma.shape[1] < 2:
        return np.array([0.0])
    diff = np.diff(chroma, axis=1)
    return np.sqrt(np.sum(diff ** 2, axis=0))


def median_filter_chroma(chroma: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    n_chroma, n_frames = chroma.shape
    filtered = np.zeros_like(chroma)
    half = kernel_size // 2
    for t in range(n_frames):
        start = max(0, t - half)
        end = min(n_frames, t + half + 1)
        filtered[:, t] = np.median(chroma[:, start:end], axis=1)
    return filtered


def log_compression_chroma(chroma: np.ndarray, gamma: float = 10.0) -> np.ndarray:
    return np.log1p(gamma * chroma)
