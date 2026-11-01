"""Shazam-style audio fingerprinting via spectral peak constellation maps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralPeak:
    time_frame: int
    freq_bin: int
    magnitude: float


@dataclass
class PeakPair:
    anchor: SpectralPeak
    target: SpectralPeak
    delta_t: int


@dataclass
class FingerprintHash:
    hash_value: int
    time_offset: int


def find_spectral_peaks(
    spectrogram: np.ndarray,
    threshold_db: float = -60.0,
    neighborhood: tuple[int, int] = (10, 5),
) -> list[SpectralPeak]:
    n_freq, n_time = spectrogram.shape
    freq_radius, time_radius = neighborhood
    mag_db = 20.0 * np.log10(np.maximum(np.abs(spectrogram), 1e-10))
    peaks: list[SpectralPeak] = []
    for t in range(time_radius, n_time - time_radius):
        for f in range(freq_radius, n_freq - freq_radius):
            val = mag_db[f, t]
            if val < threshold_db:
                continue
            region = mag_db[
                f - freq_radius : f + freq_radius + 1,
                t - time_radius : t + time_radius + 1,
            ]
            if val >= np.max(region):
                peaks.append(SpectralPeak(t, f, float(val)))
    return peaks


def build_constellation(
    peaks: list[SpectralPeak],
    target_zone: tuple[int, int, int, int] = (1, 30, 0, 200),
) -> list[PeakPair]:
    dt_min, dt_max, df_min, df_max = target_zone
    pairs: list[PeakPair] = []
    sorted_peaks = sorted(peaks, key=lambda p: p.time_frame)
    for i, anchor in enumerate(sorted_peaks):
        for j in range(i + 1, len(sorted_peaks)):
            target = sorted_peaks[j]
            dt = target.time_frame - anchor.time_frame
            if dt < dt_min:
                continue
            if dt > dt_max:
                break
            df = abs(target.freq_bin - anchor.freq_bin)
            if df_min <= df <= df_max:
                pairs.append(PeakPair(anchor, target, dt))
    return pairs


def hash_pair(pair: PeakPair, freq_bits: int = 10, dt_bits: int = 8) -> int:
    f1 = pair.anchor.freq_bin & ((1 << freq_bits) - 1)
    f2 = pair.target.freq_bin & ((1 << freq_bits) - 1)
    dt = pair.delta_t & ((1 << dt_bits) - 1)
    return (f1 << (freq_bits + dt_bits)) | (f2 << dt_bits) | dt


def fingerprint_audio(
    spectrogram: np.ndarray,
    threshold_db: float = -60.0,
    target_zone: tuple[int, int, int, int] = (1, 30, 0, 200),
) -> list[FingerprintHash]:
    peaks = find_spectral_peaks(spectrogram, threshold_db)
    pairs = build_constellation(peaks, target_zone)
    return [
        FingerprintHash(hash_value=hash_pair(p), time_offset=p.anchor.time_frame)
        for p in pairs
    ]


def match_fingerprints(
    query: list[FingerprintHash], reference: list[FingerprintHash],
) -> tuple[int, float]:
    if not query or not reference:
        return (0, 0.0)
    ref_map: dict[int, list[int]] = {}
    for fh in reference:
        ref_map.setdefault(fh.hash_value, []).append(fh.time_offset)
    offsets: list[int] = []
    for qh in query:
        if qh.hash_value in ref_map:
            for rt in ref_map[qh.hash_value]:
                offsets.append(rt - qh.time_offset)
    if not offsets:
        return (0, 0.0)
    offset_counts: dict[int, int] = {}
    for o in offsets:
        offset_counts[o] = offset_counts.get(o, 0) + 1
    best_offset = max(offset_counts, key=offset_counts.get)  # type: ignore[arg-type]
    best_count = offset_counts[best_offset]
    confidence = best_count / max(len(query), 1)
    return (best_count, min(confidence, 1.0))


def match_score(
    query: list[FingerprintHash], reference: list[FingerprintHash],
) -> float:
    _, confidence = match_fingerprints(query, reference)
    return confidence
