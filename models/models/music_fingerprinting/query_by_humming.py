"""Query-by-humming: pitch contour extraction, melody matching, DTW-based retrieval."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MelodyContour:
    pitches_hz: np.ndarray
    confidences: np.ndarray
    hop_s: float
    sr: int


@dataclass
class MatchResult:
    index: int
    distance: float
    offset: int
    confidence: float


def extract_pitch_contour(
    audio: np.ndarray,
    sr: int,
    hop_size: int = 256,
    min_freq: float = 80.0,
    max_freq: float = 800.0,
) -> MelodyContour:
    """Extract pitch contour from a hummed/sung query using autocorrelation.

    Returns a MelodyContour with per-frame f0 estimates and voicing confidence.
    """
    x = np.asarray(audio, dtype=float)
    frame_len = int(sr / min_freq) * 2
    min_lag = int(sr / max_freq)
    max_lag = int(sr / min_freq)
    n_frames = max(1, (len(x) - frame_len) // hop_size + 1)
    pitches = np.zeros(n_frames)
    confs = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        frame = x[start : start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        frame = frame - np.mean(frame)
        energy = np.sum(frame ** 2)
        if energy < 1e-10:
            continue
        frame_w = frame * np.hanning(len(frame))
        acf = np.correlate(frame_w, frame_w, mode='full')
        acf = acf[len(frame_w) - 1:]
        if acf[0] < 1e-10:
            continue
        acf = acf / acf[0]
        search = acf[min_lag:max_lag + 1]
        if len(search) == 0:
            continue
        peak_idx = np.argmax(search)
        peak_val = search[peak_idx]
        lag = peak_idx + min_lag
        if lag > 0 and peak_val > 0.3:
            pitches[i] = sr / lag
            confs[i] = float(peak_val)
    return MelodyContour(
        pitches_hz=pitches,
        confidences=confs,
        hop_s=hop_size / sr,
        sr=sr,
    )


def pitch_to_midi_cents(freq: np.ndarray) -> np.ndarray:
    """Convert Hz to continuous MIDI pitch (with cents resolution).

    Unvoiced frames (freq <= 0) map to 0.
    """
    f = np.asarray(freq, dtype=float)
    result = np.zeros_like(f)
    voiced = f > 0
    result[voiced] = 69.0 + 12.0 * np.log2(f[voiced] / 440.0)
    return result


def normalize_contour(
    contour: np.ndarray,
    method: str = "relative",
) -> np.ndarray:
    """Normalize a pitch contour for transposition-invariant matching.

    Parameters
    ----------
    method : str
        "relative" — convert to interval sequence (semitone differences).
        "mean_centered" — subtract mean pitch.
        "z_score" — standardize to zero mean, unit variance.
    """
    c = np.asarray(contour, dtype=float)
    voiced = c > 0
    if np.sum(voiced) < 2:
        return np.zeros_like(c)
    if method == "relative":
        result = np.zeros_like(c)
        prev = -1
        for i in range(len(c)):
            if c[i] > 0:
                if prev >= 0:
                    result[i] = c[i] - c[prev]
                prev = i
        return result
    elif method == "mean_centered":
        mean_pitch = np.mean(c[voiced])
        result = c.copy()
        result[voiced] -= mean_pitch
        result[~voiced] = 0
        return result
    elif method == "z_score":
        mean_p = np.mean(c[voiced])
        std_p = np.std(c[voiced])
        if std_p < 1e-6:
            std_p = 1.0
        result = c.copy()
        result[voiced] = (result[voiced] - mean_p) / std_p
        result[~voiced] = 0
        return result
    raise ValueError(f"Unknown method: {method}")


def dtw_distance(
    query: np.ndarray,
    reference: np.ndarray,
    skip_unvoiced: bool = True,
) -> float:
    """Dynamic Time Warping distance between two pitch contours.

    Returns the normalized DTW cost (lower = more similar).
    """
    q = np.asarray(query, dtype=float)
    r = np.asarray(reference, dtype=float)
    if skip_unvoiced:
        q_voiced = q[q != 0]
        r_voiced = r[r != 0]
        if len(q_voiced) == 0 or len(r_voiced) == 0:
            return float('inf')
        q, r = q_voiced, r_voiced
    n, m = len(q), len(r)
    if n == 0 or m == 0:
        return float('inf')
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(q[i - 1] - r[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))


def subsequence_dtw(
    query: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, int]:
    """Subsequence DTW: find the best-matching position of query within reference.

    Returns (distance, start_offset) where start_offset is the best alignment
    position in the reference.
    """
    q = np.asarray(query, dtype=float)
    r = np.asarray(reference, dtype=float)
    q_voiced = q[q != 0]
    r_voiced = r[r != 0]
    if len(q_voiced) == 0 or len(r_voiced) == 0:
        return float('inf'), 0
    n, m = len(q_voiced), len(r_voiced)
    if n > m:
        return dtw_distance(q_voiced, r_voiced), 0
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, :] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(q_voiced[i - 1] - r_voiced[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    best_j = int(np.argmin(cost[n, 1:])) + 1
    best_dist = float(cost[n, best_j] / n)
    start_offset = max(0, best_j - n)
    return best_dist, start_offset


def build_melody_database(
    melodies: list[np.ndarray],
    normalize: bool = True,
) -> list[np.ndarray]:
    """Pre-process a collection of melodies for efficient search.

    Converts each melody to a normalized MIDI-cent contour.
    """
    db = []
    for mel in melodies:
        midi = pitch_to_midi_cents(mel)
        if normalize:
            midi = normalize_contour(midi, method="relative")
        db.append(midi)
    return db


def search_melody_database(
    query_contour: MelodyContour,
    database: list[np.ndarray],
    top_k: int = 5,
    normalize: bool = True,
) -> list[MatchResult]:
    """Search a melody database for the closest matches to a hummed query.

    Returns top_k matches sorted by DTW distance (ascending).
    """
    q_midi = pitch_to_midi_cents(query_contour.pitches_hz)
    if normalize:
        q_midi = normalize_contour(q_midi, method="relative")
    results = []
    for idx, ref in enumerate(database):
        dist, offset = subsequence_dtw(q_midi, ref)
        q_voiced = np.sum(query_contour.confidences > 0.3)
        conf = max(0, 1.0 - dist / 5.0) * min(q_voiced / 10.0, 1.0)
        results.append(MatchResult(
            index=idx,
            distance=dist,
            offset=offset,
            confidence=float(np.clip(conf, 0, 1)),
        ))
    results.sort(key=lambda r: r.distance)
    return results[:top_k]


def contour_similarity(
    contour_a: np.ndarray,
    contour_b: np.ndarray,
    method: str = "cosine",
) -> float:
    """Compute similarity between two pitch contours.

    Parameters
    ----------
    method : str
        "cosine" — cosine similarity of normalized interval sequences.
        "correlation" — Pearson correlation of voiced frames.
        "dtw" — 1 / (1 + dtw_distance).
    """
    a = np.asarray(contour_a, dtype=float)
    b = np.asarray(contour_b, dtype=float)
    if method == "dtw":
        d = dtw_distance(a, b)
        return 1.0 / (1.0 + d) if np.isfinite(d) else 0.0
    a_midi = pitch_to_midi_cents(a)
    b_midi = pitch_to_midi_cents(b)
    a_norm = normalize_contour(a_midi, "relative")
    b_norm = normalize_contour(b_midi, "relative")
    a_v = a_norm[a_norm != 0]
    b_v = b_norm[b_norm != 0]
    min_len = min(len(a_v), len(b_v))
    if min_len < 2:
        return 0.0
    a_v = a_v[:min_len]
    b_v = b_v[:min_len]
    if method == "cosine":
        dot = np.dot(a_v, b_v)
        norm_a = np.linalg.norm(a_v)
        norm_b = np.linalg.norm(b_v)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(dot / (norm_a * norm_b))
    elif method == "correlation":
        if np.std(a_v) < 1e-10 or np.std(b_v) < 1e-10:
            return 0.0
        return float(np.corrcoef(a_v, b_v)[0, 1])
    raise ValueError(f"Unknown method: {method}")
