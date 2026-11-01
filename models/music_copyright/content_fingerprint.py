"""Audio fingerprinting and content similarity: constellation maps, melodic/harmonic matching.

Shazam-like spectral peak hashing, edit-distance melodic comparison,
rhythmic cross-correlation, harmonic n-gram matching, and originality scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sig


# ---------------------------------------------------------------------------
# Constellation-map fingerprinting
# ---------------------------------------------------------------------------

@dataclass
class SpectralPeak:
    time_frame: int
    freq_bin: int
    magnitude: float


@dataclass
class FingerprintHash:
    hash_value: int
    anchor_time: int


def find_spectral_peaks(spectrogram: np.ndarray,
                        neighborhood_time: int = 10,
                        neighborhood_freq: int = 5,
                        threshold_db: float = -60.0) -> list[SpectralPeak]:
    """Find local maxima in a spectrogram (constellation points)."""
    S_db = 10.0 * np.log10(spectrogram + 1e-10)
    F, T = S_db.shape
    peaks: list[SpectralPeak] = []

    for t in range(neighborhood_time, T - neighborhood_time):
        for f in range(neighborhood_freq, F - neighborhood_freq):
            val = S_db[f, t]
            if val < threshold_db:
                continue
            patch = S_db[
                f - neighborhood_freq : f + neighborhood_freq + 1,
                t - neighborhood_time : t + neighborhood_time + 1,
            ]
            if val >= np.max(patch):
                peaks.append(SpectralPeak(t, f, float(val)))

    return peaks


def hash_peak_pairs(peaks: list[SpectralPeak],
                    fan_out: int = 15,
                    max_dt: int = 50,
                    freq_bits: int = 10) -> list[FingerprintHash]:
    """Form (f1, f2, dt) hashes from peak pairs within a fan-out window."""
    hashes: list[FingerprintHash] = []
    sorted_peaks = sorted(peaks, key=lambda p: p.time_frame)

    for i, anchor in enumerate(sorted_peaks):
        count = 0
        for j in range(i + 1, len(sorted_peaks)):
            target = sorted_peaks[j]
            dt = target.time_frame - anchor.time_frame
            if dt > max_dt:
                break
            if dt < 1:
                continue

            f1 = anchor.freq_bin & ((1 << freq_bits) - 1)
            f2 = target.freq_bin & ((1 << freq_bits) - 1)
            dt_bits = dt & 0xFFFF

            h = (f1 << (freq_bits + 16)) | (f2 << 16) | dt_bits
            hashes.append(FingerprintHash(h, anchor.time_frame))

            count += 1
            if count >= fan_out:
                break

    return hashes


def fingerprint(spectrogram: np.ndarray, **kwargs) -> list[FingerprintHash]:
    """Full pipeline: spectrogram → peaks → hashes."""
    peaks = find_spectral_peaks(spectrogram, **{
        k: v for k, v in kwargs.items()
        if k in ("neighborhood_time", "neighborhood_freq", "threshold_db")
    })
    return hash_peak_pairs(peaks, **{
        k: v for k, v in kwargs.items()
        if k in ("fan_out", "max_dt", "freq_bits")
    })


# ---------------------------------------------------------------------------
# Fingerprint matching
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    track_id: str
    offset: int
    match_count: int
    confidence: float


def match_fingerprints(query_hashes: list[FingerprintHash],
                       database: dict[int, list[tuple[str, int]]],
                       min_matches: int = 5) -> list[MatchResult]:
    """Match query fingerprints against a database.

    database: hash_value → [(track_id, anchor_time), ...]
    """
    # Collect (track_id, offset) votes
    offset_hist: dict[tuple[str, int], int] = {}
    for qh in query_hashes:
        entries = database.get(qh.hash_value, [])
        for track_id, db_time in entries:
            offset = db_time - qh.anchor_time
            key = (track_id, offset)
            offset_hist[key] = offset_hist.get(key, 0) + 1

    # Group by track, find best offset
    track_best: dict[str, tuple[int, int]] = {}
    for (tid, offset), count in offset_hist.items():
        if tid not in track_best or count > track_best[tid][1]:
            track_best[tid] = (offset, count)

    results: list[MatchResult] = []
    total_q = max(len(query_hashes), 1)
    for tid, (offset, count) in track_best.items():
        if count >= min_matches:
            results.append(MatchResult(
                track_id=tid,
                offset=offset,
                match_count=count,
                confidence=count / total_q,
            ))

    results.sort(key=lambda r: -r.confidence)
    return results


# ---------------------------------------------------------------------------
# Melodic similarity (interval-based)
# ---------------------------------------------------------------------------

def interval_sequence(pitches: np.ndarray) -> np.ndarray:
    """Convert pitch sequence to interval sequence (transposition-invariant)."""
    return np.diff(pitches.astype(np.float64))


def edit_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Levenshtein edit distance between two sequences."""
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[m, n])


def longest_common_subsequence(a: np.ndarray, b: np.ndarray) -> int:
    """Length of LCS between two sequences."""
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[m, n])


@dataclass
class MelodicSimilarity:
    edit_distance: int
    normalised_edit: float  # 0..1 (0 = identical)
    lcs_length: int
    lcs_ratio: float        # LCS / max(len_a, len_b)
    interval_correlation: float


def melodic_similarity(melody_a: np.ndarray,
                       melody_b: np.ndarray) -> MelodicSimilarity:
    """Compare two melodies using multiple similarity metrics."""
    iv_a = interval_sequence(melody_a)
    iv_b = interval_sequence(melody_b)

    ed = edit_distance(iv_a, iv_b)
    max_len = max(len(iv_a), len(iv_b), 1)
    ned = ed / max_len

    lcs = longest_common_subsequence(iv_a, iv_b)
    lcs_r = lcs / max_len

    min_len = min(len(iv_a), len(iv_b))
    if min_len > 1 and np.std(iv_a[:min_len]) > 1e-10 and np.std(iv_b[:min_len]) > 1e-10:
        corr = float(np.corrcoef(iv_a[:min_len], iv_b[:min_len])[0, 1])
    else:
        corr = 0.0

    return MelodicSimilarity(
        edit_distance=ed,
        normalised_edit=ned,
        lcs_length=lcs,
        lcs_ratio=lcs_r,
        interval_correlation=corr,
    )


# ---------------------------------------------------------------------------
# Rhythmic similarity
# ---------------------------------------------------------------------------

def onset_grid(onset_times: np.ndarray, grid_resolution: float = 0.05,
               duration: float | None = None) -> np.ndarray:
    """Quantise onsets onto a binary grid."""
    if duration is None:
        duration = float(onset_times[-1]) + 1.0
    n_bins = int(duration / grid_resolution) + 1
    grid = np.zeros(n_bins, dtype=np.float64)
    for t in onset_times:
        idx = int(t / grid_resolution)
        if 0 <= idx < n_bins:
            grid[idx] = 1.0
    return grid


def rhythmic_similarity(onsets_a: np.ndarray, onsets_b: np.ndarray,
                        grid_res: float = 0.05) -> float:
    """Rhythmic similarity via cross-correlation of quantised onset grids.

    Returns peak normalised cross-correlation ∈ [0, 1].
    """
    dur = max(float(onsets_a[-1]), float(onsets_b[-1])) + 1.0
    ga = onset_grid(onsets_a, grid_res, dur)
    gb = onset_grid(onsets_b, grid_res, dur)

    if np.sum(ga) == 0 or np.sum(gb) == 0:
        return 0.0

    corr = sig.correlate(ga, gb, mode="full")
    norm = np.sqrt(np.sum(ga ** 2) * np.sum(gb ** 2))
    return float(np.max(corr) / (norm + 1e-10))


# ---------------------------------------------------------------------------
# Harmonic similarity (chord n-grams)
# ---------------------------------------------------------------------------

def chord_ngram_similarity(chords_a: list[str], chords_b: list[str],
                           n: int = 3) -> float:
    """Fraction of shared n-grams between two chord progressions."""
    if len(chords_a) < n or len(chords_b) < n:
        return 0.0
    ngrams_a = set(tuple(chords_a[i:i+n]) for i in range(len(chords_a) - n + 1))
    ngrams_b = set(tuple(chords_b[i:i+n]) for i in range(len(chords_b) - n + 1))
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Originality scoring
# ---------------------------------------------------------------------------

def originality_score(query_hashes: list[FingerprintHash],
                      database: dict[int, list[tuple[str, int]]]) -> float:
    """Fraction of query hashes not found in the database. Higher = more original."""
    if not query_hashes:
        return 1.0
    unmatched = sum(1 for h in query_hashes if h.hash_value not in database)
    return unmatched / len(query_hashes)
