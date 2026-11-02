"""Motivic analysis: motive extraction, transformation tracking, and development patterns.

Implements computational approaches to motivic analysis:
- Motive extraction via repeated subsequence detection
- Transformation classification (augmentation, diminution, inversion, retrograde)
- Motivic development tracking across a piece
- Similarity measures adapted for motivic comparison (Mongeau-Sankoff, geometric)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class TransformationType(Enum):
    """Standard motivic transformations."""
    EXACT = "exact"
    TRANSPOSITION = "transposition"
    INVERSION = "inversion"
    RETROGRADE = "retrograde"
    RETROGRADE_INVERSION = "retrograde_inversion"
    AUGMENTATION = "augmentation"
    DIMINUTION = "diminution"
    FRAGMENTATION = "fragmentation"
    SEQUENCE = "sequence"


@dataclass
class Motive:
    """A musical motive: a short, recognizable melodic/rhythmic unit."""
    pitches: np.ndarray
    durations: np.ndarray
    onset: float = 0.0
    label: str = ""

    @property
    def intervals(self) -> np.ndarray:
        return np.diff(self.pitches)

    @property
    def contour(self) -> np.ndarray:
        return np.sign(np.diff(self.pitches)).astype(int)

    @property
    def duration_ratios(self) -> np.ndarray:
        if len(self.durations) < 2:
            return np.array([])
        return self.durations[1:] / (self.durations[:-1] + 1e-10)


@dataclass
class MotiveOccurrence:
    """A detected occurrence of a motive within a melody."""
    motive_id: int
    start_index: int
    end_index: int
    transformation: TransformationType
    transposition: int = 0
    time_scale: float = 1.0
    similarity: float = 1.0


@dataclass
class MotivicDevelopment:
    """Tracks how a motive is developed across a piece."""
    motive: Motive
    occurrences: List[MotiveOccurrence] = field(default_factory=list)
    transformation_histogram: Dict[str, int] = field(default_factory=dict)
    density_curve: Optional[np.ndarray] = None


def interval_sequence(pitches: np.ndarray) -> np.ndarray:
    """Extract the interval sequence from a pitch sequence."""
    if len(pitches) < 2:
        return np.array([], dtype=float)
    return np.diff(pitches.astype(float))


def contour_sequence(pitches: np.ndarray) -> np.ndarray:
    """Extract contour (+1, 0, -1) from a pitch sequence."""
    if len(pitches) < 2:
        return np.array([], dtype=int)
    return np.sign(np.diff(pitches)).astype(int)


def invert_intervals(intervals: np.ndarray) -> np.ndarray:
    """Invert an interval sequence (negate all intervals)."""
    return -intervals


def retrograde_intervals(intervals: np.ndarray) -> np.ndarray:
    """Retrograde an interval sequence (reverse and negate)."""
    return -intervals[::-1]


def augment_durations(durations: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Augment durations by a multiplicative factor."""
    return durations * factor


def diminish_durations(durations: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Diminish durations by a divisive factor."""
    return durations / factor


def mongeau_sankoff_distance(m1: Motive, m2: Motive,
                              w_pitch: float = 1.0,
                              w_rhythm: float = 0.5) -> float:
    """Mongeau-Sankoff melodic similarity distance.

    Weighted edit distance combining pitch interval distance
    and duration ratio distance. Lower = more similar.
    """
    n = len(m1.pitches)
    m = len(m2.pitches)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(n + 1):
        dp[i, 0] = i * (w_pitch + w_rhythm)
    for j in range(m + 1):
        dp[0, j] = j * (w_pitch + w_rhythm)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pitch_cost = w_pitch * abs(float(m1.pitches[i - 1]) - float(m2.pitches[j - 1]))
            dur_cost = w_rhythm * abs(np.log2(m1.durations[i - 1] / (m2.durations[j - 1] + 1e-10) + 1e-10))
            subst = dp[i - 1, j - 1] + pitch_cost + dur_cost
            delete = dp[i - 1, j] + w_pitch + w_rhythm
            insert = dp[i, j - 1] + w_pitch + w_rhythm
            dp[i, j] = min(subst, delete, insert)
    return float(dp[n, m])


def contour_similarity(c1: np.ndarray, c2: np.ndarray) -> float:
    """Similarity between two contour sequences via longest common subsequence ratio.

    Returns value in [0, 1]; 1 = identical contour.
    """
    n = len(c1)
    m = len(c2)
    if n == 0 or m == 0:
        return 0.0
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if c1[i - 1] == c2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return float(dp[n, m]) / max(n, m)


def classify_transformation(original: Motive,
                             candidate: Motive,
                             pitch_tolerance: float = 0.5,
                             rhythm_tolerance: float = 0.15) -> Tuple[TransformationType, int, float]:
    """Classify how a candidate relates to the original motive.

    Returns (transformation_type, transposition_semitones, time_scale_factor).
    """
    o_int = original.intervals
    c_int = candidate.intervals

    if len(o_int) == 0 or len(c_int) == 0:
        return TransformationType.EXACT, 0, 1.0

    o_dur = original.duration_ratios
    c_dur = candidate.duration_ratios

    def intervals_match(a: np.ndarray, b: np.ndarray) -> bool:
        if len(a) != len(b):
            return False
        return bool(np.all(np.abs(a - b) <= pitch_tolerance))

    def durations_match(a: np.ndarray, b: np.ndarray) -> bool:
        if len(a) != len(b) or len(a) == 0:
            return True
        ratios = a / (b + 1e-10)
        return bool(np.std(ratios) < rhythm_tolerance)

    transposition = int(round(float(candidate.pitches[0]) - float(original.pitches[0])))

    if intervals_match(o_int, c_int):
        dur_ratio = np.mean(candidate.durations) / (np.mean(original.durations) + 1e-10)
        if abs(dur_ratio - 1.0) < rhythm_tolerance:
            if transposition == 0:
                return TransformationType.EXACT, 0, 1.0
            return TransformationType.TRANSPOSITION, transposition, 1.0
        elif dur_ratio > 1.0 + rhythm_tolerance:
            return TransformationType.AUGMENTATION, transposition, dur_ratio
        else:
            return TransformationType.DIMINUTION, transposition, dur_ratio

    inv_int = invert_intervals(o_int)
    if intervals_match(inv_int, c_int):
        return TransformationType.INVERSION, transposition, 1.0

    ret_int = retrograde_intervals(o_int)
    if intervals_match(ret_int, c_int):
        return TransformationType.RETROGRADE, transposition, 1.0

    ri_int = -o_int[::-1]
    if intervals_match(ri_int, c_int):
        return TransformationType.RETROGRADE_INVERSION, transposition, 1.0

    if len(c_int) < len(o_int) and len(c_int) >= 2:
        for start in range(len(o_int) - len(c_int) + 1):
            sub = o_int[start:start + len(c_int)]
            if intervals_match(sub, c_int):
                return TransformationType.FRAGMENTATION, transposition, 1.0

    if len(c_int) == len(o_int) and len(c_int) >= 2:
        diffs = c_int - o_int
        if np.std(diffs) < pitch_tolerance and abs(np.mean(diffs)) > pitch_tolerance:
            return TransformationType.SEQUENCE, transposition, 1.0

    return TransformationType.TRANSPOSITION, transposition, 1.0


def extract_repeated_subsequences(pitches: np.ndarray,
                                   min_length: int = 3,
                                   max_length: int = 12,
                                   min_occurrences: int = 2) -> List[Tuple[np.ndarray, List[int]]]:
    """Find repeated pitch subsequences that may be motives.

    Uses a suffix-array-inspired approach: hash all subsequences of
    each length, count occurrences, return those meeting the threshold.
    """
    n = len(pitches)
    results = []
    for length in range(min_length, min(max_length + 1, n + 1)):
        seen: Dict[tuple, List[int]] = {}
        for i in range(n - length + 1):
            intervals = tuple(int(x) for x in np.diff(pitches[i:i + length]))
            if intervals not in seen:
                seen[intervals] = []
            seen[intervals].append(i)
        for intervals_key, positions in seen.items():
            if len(positions) >= min_occurrences:
                representative = pitches[positions[0]:positions[0] + length]
                results.append((representative.copy(), positions))
    results.sort(key=lambda x: len(x[0]) * len(x[1]), reverse=True)
    return results


def _remove_overlapping(positions: List[int], length: int) -> List[int]:
    """Keep non-overlapping positions, greedily from left."""
    if not positions:
        return []
    kept = [positions[0]]
    for p in positions[1:]:
        if p >= kept[-1] + length:
            kept.append(p)
    return kept


def extract_motives(pitches: np.ndarray,
                     durations: np.ndarray,
                     min_length: int = 3,
                     max_length: int = 8,
                     min_occurrences: int = 2) -> List[Motive]:
    """Extract candidate motives from a melody by finding repeated interval patterns."""
    repeated = extract_repeated_subsequences(pitches, min_length, max_length, min_occurrences)
    motives = []
    used = set()
    for rep_pitches, positions in repeated:
        length = len(rep_pitches)
        clean_pos = _remove_overlapping(positions, length)
        if len(clean_pos) < min_occurrences:
            continue
        pos = clean_pos[0]
        overlap = False
        for u_start, u_end in used:
            if pos < u_end and pos + length > u_start:
                overlap = True
                break
        if overlap:
            continue
        m = Motive(
            pitches=rep_pitches.copy(),
            durations=durations[pos:pos + length].copy(),
            onset=float(np.sum(durations[:pos])),
            label=f"m{len(motives) + 1}",
        )
        motives.append(m)
        for p in clean_pos:
            used.add((p, p + length))
    return motives


def track_motive_occurrences(motive: Motive,
                              melody_pitches: np.ndarray,
                              melody_durations: np.ndarray,
                              window_sizes: Optional[List[int]] = None,
                              pitch_tolerance: float = 1.0) -> List[MotiveOccurrence]:
    """Find all occurrences of a motive (including transformed) in a melody."""
    n = len(melody_pitches)
    m_len = len(motive.pitches)
    if window_sizes is None:
        window_sizes = [m_len - 1, m_len, m_len + 1]
    occurrences = []
    for w in window_sizes:
        if w < 2 or w > n:
            continue
        for i in range(n - w + 1):
            candidate = Motive(
                pitches=melody_pitches[i:i + w],
                durations=melody_durations[i:i + w],
            )
            transform, transp, scale = classify_transformation(
                motive, candidate, pitch_tolerance=pitch_tolerance
            )
            c_contour = candidate.contour
            m_contour = motive.contour
            sim = contour_similarity(m_contour, c_contour)
            if sim < 0.5 and transform not in (
                TransformationType.EXACT,
                TransformationType.TRANSPOSITION,
                TransformationType.INVERSION,
                TransformationType.RETROGRADE,
                TransformationType.RETROGRADE_INVERSION,
                TransformationType.FRAGMENTATION,
            ):
                continue
            occ = MotiveOccurrence(
                motive_id=0,
                start_index=i,
                end_index=i + w,
                transformation=transform,
                transposition=transp,
                time_scale=scale,
                similarity=sim,
            )
            occurrences.append(occ)
    filtered = []
    for occ in sorted(occurrences, key=lambda o: -o.similarity):
        overlap = False
        for f in filtered:
            if occ.start_index < f.end_index and occ.end_index > f.start_index:
                overlap = True
                break
        if not overlap:
            filtered.append(occ)
    return sorted(filtered, key=lambda o: o.start_index)


def motivic_density_curve(occurrences: List[MotiveOccurrence],
                           melody_length: int,
                           window: int = 16) -> np.ndarray:
    """Compute a density curve of motive occurrences over time.

    Higher values indicate regions of intense motivic activity.
    """
    hits = np.zeros(melody_length)
    for occ in occurrences:
        hits[occ.start_index:min(occ.end_index, melody_length)] += occ.similarity
    if window <= 1:
        return hits
    kernel = np.ones(window) / window
    padded = np.pad(hits, (window // 2, window // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='same')
    return smoothed[window // 2:window // 2 + melody_length]


def analyze_motivic_development(motive: Motive,
                                 melody_pitches: np.ndarray,
                                 melody_durations: np.ndarray) -> MotivicDevelopment:
    """Full motivic development analysis: find occurrences, classify, build density curve."""
    occurrences = track_motive_occurrences(motive, melody_pitches, melody_durations)
    hist: Dict[str, int] = {}
    for occ in occurrences:
        key = occ.transformation.value
        hist[key] = hist.get(key, 0) + 1
    density = motivic_density_curve(occurrences, len(melody_pitches))
    return MotivicDevelopment(
        motive=motive,
        occurrences=occurrences,
        transformation_histogram=hist,
        density_curve=density,
    )


def transformation_distance(t1: TransformationType,
                             t2: TransformationType) -> float:
    """Heuristic distance between two transformation types.

    Based on Schoenberg's hierarchy: exact < transposition < inversion
    < retrograde < retrograde-inversion.
    """
    order = {
        TransformationType.EXACT: 0,
        TransformationType.TRANSPOSITION: 1,
        TransformationType.AUGMENTATION: 2,
        TransformationType.DIMINUTION: 2,
        TransformationType.SEQUENCE: 2,
        TransformationType.INVERSION: 3,
        TransformationType.RETROGRADE: 4,
        TransformationType.RETROGRADE_INVERSION: 5,
        TransformationType.FRAGMENTATION: 3,
    }
    return abs(order.get(t1, 5) - order.get(t2, 5)) / 5.0


def motivic_coherence(occurrences: List[MotiveOccurrence]) -> float:
    """Measure how coherently a motive is used across a piece.

    Considers density of occurrences and diversity of transformations.
    Returns value in [0, 1].
    """
    if not occurrences:
        return 0.0
    n = len(occurrences)
    avg_sim = np.mean([o.similarity for o in occurrences])
    transform_types = set(o.transformation for o in occurrences)
    diversity = len(transform_types) / len(TransformationType)
    coherence = avg_sim * (0.7 + 0.3 * diversity)
    return float(min(1.0, coherence))
