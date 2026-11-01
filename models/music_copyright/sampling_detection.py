"""Sample detection: cross-correlation search, pitch shift, time stretch, loop finding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SampleMatch:
    offset: int
    correlation: float
    pitch_shift_semitones: float
    time_stretch_ratio: float


@dataclass
class LoopDetection:
    loop_start: int
    loop_length: int
    confidence: float


def cross_correlation_search(
    haystack: np.ndarray,
    needle: np.ndarray,
    hop: int = 512,
    threshold: float = 0.7,
) -> list[tuple[int, float]]:
    """Find occurrences of *needle* in *haystack* via normalised cross-correlation."""
    if len(needle) > len(haystack) or len(needle) == 0:
        return []
    needle_norm = needle - needle.mean()
    needle_energy = np.sqrt(np.sum(needle_norm ** 2))
    if needle_energy < 1e-12:
        return []
    matches: list[tuple[int, float]] = []
    nlen = len(needle)
    for start in range(0, len(haystack) - nlen + 1, hop):
        seg = haystack[start : start + nlen]
        seg_norm = seg - seg.mean()
        seg_energy = np.sqrt(np.sum(seg_norm ** 2))
        if seg_energy < 1e-12:
            continue
        corr = float(np.sum(needle_norm * seg_norm) / (needle_energy * seg_energy))
        if corr >= threshold:
            matches.append((start, corr))
    return matches


def spectral_envelope(signal: np.ndarray, n_fft: int = 2048) -> np.ndarray:
    """Smoothed magnitude spectrum (LPC-style envelope via cepstral liftering)."""
    windowed = signal[: n_fft] * np.hanning(min(n_fft, len(signal)))
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    log_spec = np.log(spectrum + 1e-12)
    cepstrum = np.fft.irfft(log_spec)
    lifter_order = 30
    cepstrum[lifter_order : -lifter_order + 1 if lifter_order > 1 else len(cepstrum)] = 0.0
    envelope = np.exp(np.fft.rfft(cepstrum).real)
    return envelope


def envelope_similarity(env_a: np.ndarray, env_b: np.ndarray) -> float:
    """Cosine similarity between two spectral envelopes."""
    min_len = min(len(env_a), len(env_b))
    a, b = env_a[:min_len], env_b[:min_len]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def detect_pitch_shift(
    original: np.ndarray,
    candidate: np.ndarray,
    sr: int = 44100,
    max_semitones: float = 12.0,
) -> float:
    """Estimate pitch shift in semitones between two signals via autocorrelation peak."""
    def _fundamental(sig: np.ndarray) -> float:
        ac = np.correlate(sig[:4096], sig[:4096], mode="full")
        ac = ac[len(ac) // 2 :]
        min_lag = sr // 2000  # 2 kHz upper bound
        max_lag = sr // 50     # 50 Hz lower bound
        ac_search = ac[min_lag:max_lag]
        if len(ac_search) == 0:
            return 0.0
        peak = int(np.argmax(ac_search)) + min_lag
        return sr / peak if peak > 0 else 0.0

    f0_orig = _fundamental(original)
    f0_cand = _fundamental(candidate)
    if f0_orig < 1.0 or f0_cand < 1.0:
        return 0.0
    semitones = 12.0 * np.log2(f0_cand / f0_orig)
    if abs(semitones) > max_semitones:
        return 0.0
    return float(semitones)


def detect_time_stretch(
    original: np.ndarray,
    candidate: np.ndarray,
    sr: int = 44100,
    hop: int = 512,
) -> float:
    """Estimate time-stretch ratio via onset density comparison."""
    def _onset_rate(sig: np.ndarray) -> float:
        energy = np.array([
            np.sum(sig[i : i + hop] ** 2)
            for i in range(0, len(sig) - hop, hop)
        ])
        if len(energy) < 2:
            return 0.0
        diff = np.diff(energy)
        threshold = np.mean(np.abs(diff)) + 2.0 * np.std(np.abs(diff))
        onsets = np.sum(diff > threshold)
        duration = len(sig) / sr
        return onsets / max(duration, 1e-6)

    rate_orig = _onset_rate(original)
    rate_cand = _onset_rate(candidate)
    if rate_orig < 1e-6:
        return 1.0
    return float(rate_cand / rate_orig)


def detect_loops(
    signal: np.ndarray,
    sr: int = 44100,
    hop: int = 512,
    min_loop_ms: float = 100.0,
    max_loop_ms: float = 5000.0,
    threshold: float = 0.5,
) -> list[LoopDetection]:
    """Detect repeating loops via autocorrelation of frame energy."""
    frame_energy = np.array([
        np.sum(signal[i : i + hop] ** 2)
        for i in range(0, len(signal) - hop, hop)
    ], dtype=np.float64)
    if len(frame_energy) < 4:
        return []
    fe_norm = frame_energy - frame_energy.mean()
    energy_std = np.std(fe_norm)
    if energy_std < 1e-12:
        return []
    fe_norm = fe_norm / energy_std
    ac = np.correlate(fe_norm, fe_norm, mode="full")
    ac = ac[len(ac) // 2 :]
    ac = ac / (ac[0] + 1e-12)
    min_lag = max(1, int(min_loop_ms / 1000.0 * sr / hop))
    max_lag = min(len(ac) - 1, int(max_loop_ms / 1000.0 * sr / hop))
    if min_lag >= max_lag:
        return []
    loops: list[LoopDetection] = []
    ac_search = ac[min_lag : max_lag + 1]
    for i in range(1, len(ac_search) - 1):
        if ac_search[i] > ac_search[i - 1] and ac_search[i] > ac_search[i + 1]:
            if ac_search[i] >= threshold:
                lag = i + min_lag
                loop_samples = lag * hop
                loops.append(LoopDetection(
                    loop_start=0,
                    loop_length=loop_samples,
                    confidence=float(ac_search[i]),
                ))
    loops.sort(key=lambda x: -x.confidence)
    return loops


def find_sample_matches(
    reference: np.ndarray,
    query: np.ndarray,
    sr: int = 44100,
    search_hop: int = 2048,
    min_correlation: float = 0.6,
) -> list[SampleMatch]:
    """Full pipeline: find where *query* appears in *reference* with transform estimates."""
    raw_matches = cross_correlation_search(reference, query, hop=search_hop, threshold=min_correlation)
    results: list[SampleMatch] = []
    for offset, corr in raw_matches:
        seg = reference[offset : offset + len(query)]
        ps = detect_pitch_shift(query, seg, sr=sr)
        ts = detect_time_stretch(query, seg, sr=sr)
        results.append(SampleMatch(
            offset=offset,
            correlation=corr,
            pitch_shift_semitones=ps,
            time_stretch_ratio=ts,
        ))
    return results
