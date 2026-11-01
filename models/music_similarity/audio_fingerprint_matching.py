"""Audio fingerprint matching: landmark-based fingerprinting, hash matching,
and robust audio identification.

Implements Shazam-style spectral peak constellation matching (Wang 2003),
chromaprint-like chroma hashing, and time-frequency landmark extraction.
Pure numpy/scipy — no external fingerprinting libraries.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class HashMethod(Enum):
    CONSTELLATION = "constellation"
    CHROMA = "chroma"
    MFCC_HASH = "mfcc_hash"


@dataclass
class SpectralPeak:
    time_bin: int
    freq_bin: int
    magnitude: float


@dataclass
class Landmark:
    anchor: SpectralPeak
    point: SpectralPeak
    hash_value: int


@dataclass
class FingerprintMatch:
    query_offset: int
    reference_offset: int
    confidence: float
    n_matching_hashes: int


def stft_peaks(signal: np.ndarray, sr: int = 22050,
               n_fft: int = 2048, hop: int = 512,
               threshold_ratio: float = 0.3,
               neighborhood: int = 10) -> List[SpectralPeak]:
    n_frames = (len(signal) - n_fft) // hop + 1
    if n_frames <= 0:
        return []
    window = np.hanning(n_fft)
    peaks = []
    for t in range(n_frames):
        frame = signal[t * hop: t * hop + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        threshold = threshold_ratio * np.max(spectrum)
        for f in range(neighborhood, len(spectrum) - neighborhood):
            if spectrum[f] < threshold:
                continue
            local = spectrum[f - neighborhood: f + neighborhood + 1]
            if spectrum[f] == np.max(local):
                peaks.append(SpectralPeak(t, f, float(spectrum[f])))
    return peaks


def peaks_to_landmarks(peaks: List[SpectralPeak],
                       fan_out: int = 5,
                       time_window: int = 50,
                       freq_range: int = 200) -> List[Landmark]:
    landmarks = []
    for i, anchor in enumerate(peaks):
        targets = []
        for j in range(i + 1, len(peaks)):
            p = peaks[j]
            dt = p.time_bin - anchor.time_bin
            if dt <= 0:
                continue
            if dt > time_window:
                break
            if abs(p.freq_bin - anchor.freq_bin) > freq_range:
                continue
            targets.append(p)
            if len(targets) >= fan_out:
                break
        for target in targets:
            hash_val = (anchor.freq_bin & 0x1FF) << 20
            hash_val |= (target.freq_bin & 0x1FF) << 11
            hash_val |= ((target.time_bin - anchor.time_bin) & 0x7FF)
            landmarks.append(Landmark(anchor, target, hash_val))
    return landmarks


def fingerprint_signal(signal: np.ndarray, sr: int = 22050,
                       n_fft: int = 2048, hop: int = 512) -> List[Landmark]:
    peaks = stft_peaks(signal, sr, n_fft, hop)
    return peaks_to_landmarks(peaks)


def build_hash_table(landmarks: List[Landmark],
                     track_id: int = 0) -> Dict[int, List[Tuple[int, int]]]:
    table: Dict[int, List[Tuple[int, int]]] = {}
    for lm in landmarks:
        h = lm.hash_value
        if h not in table:
            table[h] = []
        table[h].append((lm.anchor.time_bin, track_id))
    return table


def match_fingerprints(query_landmarks: List[Landmark],
                       db_table: Dict[int, List[Tuple[int, int]]],
                       min_matches: int = 5) -> List[FingerprintMatch]:
    offset_counts: Dict[Tuple[int, int], int] = {}
    for lm in query_landmarks:
        h = lm.hash_value
        if h not in db_table:
            continue
        for ref_time, track_id in db_table[h]:
            offset = ref_time - lm.anchor.time_bin
            key = (track_id, offset)
            offset_counts[key] = offset_counts.get(key, 0) + 1

    matches = []
    for (track_id, offset), count in offset_counts.items():
        if count >= min_matches:
            confidence = min(1.0, count / (len(query_landmarks) + 1e-10))
            matches.append(FingerprintMatch(
                query_offset=0, reference_offset=offset,
                confidence=confidence, n_matching_hashes=count
            ))
    return sorted(matches, key=lambda m: -m.confidence)


def chroma_fingerprint(signal: np.ndarray, sr: int = 22050,
                       n_fft: int = 4096, hop: int = 2048,
                       n_bits: int = 12) -> np.ndarray:
    n_frames = max(1, (len(signal) - n_fft) // hop + 1)
    window = np.hanning(n_fft)
    chroma_seq = np.zeros((n_frames, 12))

    for t in range(n_frames):
        start = t * hop
        end = min(start + n_fft, len(signal))
        frame = np.zeros(n_fft)
        frame[:end - start] = signal[start:end]
        frame *= window
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        for i, (f, m) in enumerate(zip(freqs, spectrum)):
            if f < 60 or f > 5000 or m < 1e-6:
                continue
            note = 12 * np.log2(f / 440.0 + 1e-30) + 69
            pc = int(round(note)) % 12
            chroma_seq[t, pc] += m

    hashes = np.zeros(n_frames, dtype=np.int64)
    for t in range(n_frames):
        c = chroma_seq[t]
        median_c = np.median(c)
        bits = (c > median_c).astype(int)
        h = 0
        for b in range(min(n_bits, 12)):
            h |= bits[b] << b
        hashes[t] = h
    return hashes


def hamming_distance(a: int, b: int) -> int:
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def chroma_match_score(query_hashes: np.ndarray,
                       ref_hashes: np.ndarray,
                       max_hamming: int = 2) -> Tuple[float, int]:
    best_score = 0.0
    best_offset = 0
    q_len = len(query_hashes)
    r_len = len(ref_hashes)

    for offset in range(max(1, r_len - q_len + 1)):
        matches = 0
        compare_len = min(q_len, r_len - offset)
        for i in range(compare_len):
            if hamming_distance(int(query_hashes[i]), int(ref_hashes[offset + i])) <= max_hamming:
                matches += 1
        score = matches / max(compare_len, 1)
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_score, best_offset


def bit_error_rate(hash_a: np.ndarray, hash_b: np.ndarray) -> float:
    if len(hash_a) == 0 or len(hash_b) == 0:
        return 1.0
    min_len = min(len(hash_a), len(hash_b))
    total_bits = 0
    error_bits = 0
    for i in range(min_len):
        xor = int(hash_a[i]) ^ int(hash_b[i])
        n_bits = max(int(hash_a[i]).bit_length(), int(hash_b[i]).bit_length(), 1)
        total_bits += n_bits
        while xor:
            error_bits += xor & 1
            xor >>= 1
    return error_bits / max(total_bits, 1)


def robustness_score(original: np.ndarray, degraded: np.ndarray,
                     sr: int = 22050) -> float:
    fp_orig = fingerprint_signal(original, sr)
    fp_deg = fingerprint_signal(degraded, sr)
    if not fp_orig or not fp_deg:
        return 0.0

    table = build_hash_table(fp_orig)
    matches = match_fingerprints(fp_deg, table, min_matches=1)
    if not matches:
        return 0.0
    return matches[0].confidence


def discriminability(signals: List[np.ndarray], labels: np.ndarray,
                     sr: int = 22050) -> float:
    n = len(signals)
    fps = [fingerprint_signal(s, sr) for s in signals]
    tables = [build_hash_table(fp, i) for i, fp in enumerate(fps)]

    merged: Dict[int, List[Tuple[int, int]]] = {}
    for t in tables:
        for h, entries in t.items():
            if h not in merged:
                merged[h] = []
            merged[h].extend(entries)

    correct = 0
    total = 0
    for i in range(n):
        matches = match_fingerprints(fps[i], merged, min_matches=1)
        if matches:
            best = matches[0]
            if best.reference_offset == 0:
                correct += 1
        total += 1
    return correct / max(total, 1)
