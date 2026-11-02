"""Structural and compositional similarity analysis for music copyright comparison.

Compares two musical works across multiple dimensions — melody, rhythm,
harmony, and large-scale form — to produce a composite similarity report
suitable for preliminary copyright-infringement screening.

All algorithms are symbolic (operate on MIDI-like representations, not raw
audio) and use only numpy/scipy.  The DTW implementation follows Müller (2015),
*Fundamentals of Music Processing*, ch. 4 (subsequence DTW with Sakoe–Chiba
band), adapted for melodic pitch sequences.

References
----------
Müller, M. (2015). *Fundamentals of Music Processing*. Springer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cosine as _cosine_dist


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SegmentAlignment:
    """Aligned segment pair produced by DTW or LCS back-tracking."""
    query_start: int
    query_end: int
    ref_start: int
    ref_end: int
    score: float


@dataclass
class SimilarityReport:
    """Composite similarity report across multiple musical dimensions."""
    melodic_sim: float
    rhythmic_sim: float
    harmonic_sim: float
    structural_sim: float
    overall: float
    verdict: str
    detail: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parsons_code(pitches: Sequence[float]) -> List[int]:
    """Return the Parsons contour code: +1 (up), -1 (down), 0 (repeat).

    The code captures the *direction* of successive pitch intervals and is
    transposition-invariant by construction.
    """
    code: List[int] = []
    for i in range(1, len(pitches)):
        diff = pitches[i] - pitches[i - 1]
        if diff > 0:
            code.append(1)
        elif diff < 0:
            code.append(-1)
        else:
            code.append(0)
    return code


def _normalised_edit_distance(seq_a: Sequence, seq_b: Sequence) -> float:
    """Levenshtein edit distance normalised to [0, 1] by max length."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return 0.0
    # Standard DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # delete
                dp[i][j - 1] + 1,       # insert
                dp[i - 1][j - 1] + cost  # substitute
            )
    return dp[n][m] / max(n, m)


def _interval_sequence(pitches: Sequence[float]) -> List[int]:
    """Convert a pitch sequence to a list of signed semitone intervals."""
    return [int(round(pitches[i] - pitches[i - 1])) for i in range(1, len(pitches))]


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Symmetric KL divergence (Jensen–Shannon-style average)."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


# ---------------------------------------------------------------------------
# Core similarity functions
# ---------------------------------------------------------------------------

def melodic_contour_similarity(
    pitches_a: Sequence[float],
    pitches_b: Sequence[float],
) -> float:
    """Compare melodic contours via Parsons-code correlation.

    Returns a value in [0, 1] where 1 means identical contour direction
    sequences and 0 means maximally dissimilar.
    """
    code_a = _parsons_code(pitches_a)
    code_b = _parsons_code(pitches_b)
    if len(code_a) == 0 or len(code_b) == 0:
        return 0.0
    # Truncate to shorter length for element-wise comparison
    min_len = min(len(code_a), len(code_b))
    a = np.array(code_a[:min_len], dtype=float)
    b = np.array(code_b[:min_len], dtype=float)
    # Correlation coefficient, mapped to [0, 1]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        # One sequence is constant — compare equality
        return float(np.mean(a == b))
    corr = float(np.corrcoef(a, b)[0, 1])
    return (corr + 1.0) / 2.0


def interval_sequence_similarity(
    pitches_a: Sequence[float],
    pitches_b: Sequence[float],
) -> float:
    """Compare interval sequences using normalised edit distance.

    Returns 1 − normalised_edit_distance, so 1 means identical interval
    patterns and 0 means maximally different.
    """
    ints_a = _interval_sequence(pitches_a)
    ints_b = _interval_sequence(pitches_b)
    if len(ints_a) == 0 and len(ints_b) == 0:
        return 1.0
    if len(ints_a) == 0 or len(ints_b) == 0:
        return 0.0
    return 1.0 - _normalised_edit_distance(ints_a, ints_b)


def rhythmic_pattern_similarity(
    durations_a: Sequence[float],
    durations_b: Sequence[float],
) -> float:
    """Compare duration patterns using inter-onset interval (IOI) ratios.

    Each duration sequence is converted to a vector of successive IOI ratios
    (dur[i+1] / dur[i]), log-transformed, then compared via cosine
    similarity.  This makes the measure tempo-invariant.

    Returns a value in [0, 1].
    """
    def _ioi_ratios(durs: Sequence[float]) -> np.ndarray:
        ratios = []
        for i in range(1, len(durs)):
            if durs[i - 1] > 0:
                ratios.append(math.log2(max(durs[i], 1e-9) / durs[i - 1]))
            else:
                ratios.append(0.0)
        return np.array(ratios, dtype=float)

    ra = _ioi_ratios(durations_a)
    rb = _ioi_ratios(durations_b)
    if len(ra) == 0 or len(rb) == 0:
        return 0.0
    # Align lengths to the shorter one
    min_len = min(len(ra), len(rb))
    ra, rb = ra[:min_len], rb[:min_len]
    if np.linalg.norm(ra) < 1e-12 or np.linalg.norm(rb) < 1e-12:
        return float(np.allclose(ra, rb, atol=1e-6))
    return float(max(0.0, 1.0 - _cosine_dist(ra, rb)))


def harmonic_progression_similarity(
    chords_a: Sequence[str],
    chords_b: Sequence[str],
) -> float:
    """Compare chord progressions via Roman-numeral abstraction + edit distance.

    Chords are first reduced to a simplified quality string (e.g. ``"Cmaj"``
    → ``"I"``, ``"Am"`` → ``"vi"``).  When Roman numeral conversion is not
    possible (e.g. chords are already in Roman-numeral form or are free-form
    labels), the raw strings are compared directly.

    Returns 1 − normalised_edit_distance in [0, 1].
    """
    # Map to a canonical form: strip whitespace, lowercase for comparison
    def _canon(ch: str) -> str:
        return ch.strip()

    a = [_canon(c) for c in chords_a]
    b = [_canon(c) for c in chords_b]
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return 1.0 - _normalised_edit_distance(a, b)


def pitch_class_distribution_similarity(
    pitches_a: Sequence[float],
    pitches_b: Sequence[float],
) -> float:
    """Compare pitch-class histograms using Jensen–Shannon divergence.

    Each pitch (MIDI number) is reduced mod 12 to its pitch class.  The
    resulting 12-bin histograms are compared with a symmetric KL divergence
    (Jensen–Shannon divergence).  The raw JSD is bounded [0, ln(2)]; we map
    to a similarity score in [0, 1] as ``1 − JSD / ln(2)``.
    """
    def _histogram(pitches: Sequence[float]) -> np.ndarray:
        bins = np.zeros(12, dtype=float)
        for p in pitches:
            pc = int(round(p)) % 12
            bins[pc] += 1.0
        total = bins.sum()
        if total > 0:
            bins /= total
        else:
            bins[:] = 1.0 / 12
        return bins

    ha = _histogram(pitches_a)
    hb = _histogram(pitches_b)
    jsd = _kl_divergence(ha, hb)
    return float(max(0.0, 1.0 - jsd / math.log(2)))


def dtw_melodic_alignment(
    pitches_a: Sequence[float],
    pitches_b: Sequence[float],
    *,
    band_fraction: float = 0.2,
) -> Tuple[List[Tuple[int, int]], float]:
    """DTW alignment for melody comparison (Müller 2015, ch. 4).

    Parameters
    ----------
    pitches_a, pitches_b : sequences of MIDI pitch values.
    band_fraction : Sakoe–Chiba band width as a fraction of sequence length
        (0 = no band, 1 = full matrix).

    Returns
    -------
    path : list of ``(i, j)`` index pairs along the optimal warping path.
    distance : the normalised DTW distance (total cost / path length).
    """
    a = np.asarray(pitches_a, dtype=float)
    b = np.asarray(pitches_b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return [], float("inf")

    band = max(1, int(band_fraction * max(n, m)))
    INF = 1e18
    cost = np.full((n, m), INF, dtype=float)

    for i in range(n):
        j_lo = max(0, int(i * m / n) - band)
        j_hi = min(m, int(i * m / n) + band + 1)
        for j in range(j_lo, j_hi):
            c = abs(a[i] - b[j])
            prev = INF
            if i > 0 and j > 0:
                prev = min(prev, cost[i - 1, j - 1])
            if i > 0:
                prev = min(prev, cost[i - 1, j])
            if j > 0:
                prev = min(prev, cost[i, j - 1])
            if i == 0 and j == 0:
                prev = 0.0
            cost[i, j] = c + prev

    # Back-track to recover the optimal path
    path: List[Tuple[int, int]] = [(n - 1, m - 1)]
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        candidates = []
        if i > 0 and j > 0:
            candidates.append((cost[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((cost[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((cost[i, j - 1], i, j - 1))
        _, bi, bj = min(candidates, key=lambda t: t[0])
        path.append((bi, bj))
        i, j = bi, bj
    path.reverse()

    distance = cost[n - 1, m - 1] / len(path) if path else float("inf")
    return path, float(distance)


def longest_common_subsequence_ratio(
    seq_a: Sequence,
    seq_b: Sequence,
) -> float:
    """LCS length normalised by the shorter sequence length.

    Returns a value in [0, 1]; 1 means the shorter sequence is entirely
    embedded in the longer one.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0.0
    # Standard DP for LCS length
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[n][m]
    return lcs_len / min(n, m)


def structural_form_similarity(
    sections_a: Sequence[str],
    sections_b: Sequence[str],
) -> float:
    """Compare high-level form labels (e.g. ``['A','A','B','A']``) via edit distance.

    Returns 1 − normalised_edit_distance in [0, 1].
    """
    if len(sections_a) == 0 and len(sections_b) == 0:
        return 1.0
    if len(sections_a) == 0 or len(sections_b) == 0:
        return 0.0
    return 1.0 - _normalised_edit_distance(list(sections_a), list(sections_b))


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------

# Default weights (melodic emphasis — melody is the most legally significant
# dimension in US case law, see *Three Boys Music v. Michael Bolton*).
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "melodic": 0.35,
    "rhythmic": 0.20,
    "harmonic": 0.25,
    "structural": 0.20,
}


def generate_similarity_report(
    melody_a: Sequence[float],
    rhythm_a: Sequence[float],
    chords_a: Sequence[str],
    melody_b: Sequence[float],
    rhythm_b: Sequence[float],
    chords_b: Sequence[str],
    *,
    sections_a: Optional[Sequence[str]] = None,
    sections_b: Optional[Sequence[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> SimilarityReport:
    """Generate a composite structural-similarity report.

    Each dimension is scored independently, then combined with *weights*
    (default: melodic 0.35, rhythmic 0.20, harmonic 0.25, structural 0.20).
    The *verdict* string is one of ``"low"``, ``"moderate"``, ``"high"``, or
    ``"very_high"`` similarity.

    Parameters
    ----------
    melody_a, melody_b : MIDI pitch sequences.
    rhythm_a, rhythm_b : note duration sequences (seconds or beats).
    chords_a, chords_b : chord label sequences.
    sections_a, sections_b : optional form-letter sequences (e.g. ``['A','B','A']``).
    weights : optional mapping overriding the default dimension weights.
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)
    # Normalise weights to sum to 1
    total_w = sum(w.values())
    if total_w > 0:
        w = {k: v / total_w for k, v in w.items()}

    # --- Melodic dimension (average of contour, interval, pitch-class) ---
    contour = melodic_contour_similarity(melody_a, melody_b)
    interval = interval_sequence_similarity(melody_a, melody_b)
    pcd = pitch_class_distribution_similarity(melody_a, melody_b)
    lcs = longest_common_subsequence_ratio(
        _interval_sequence(melody_a),
        _interval_sequence(melody_b),
    )
    melodic = 0.30 * contour + 0.30 * interval + 0.20 * pcd + 0.20 * lcs

    # --- Rhythmic dimension ---
    rhythmic = rhythmic_pattern_similarity(rhythm_a, rhythm_b)

    # --- Harmonic dimension ---
    harmonic = harmonic_progression_similarity(chords_a, chords_b)

    # --- Structural dimension ---
    if sections_a is not None and sections_b is not None:
        structural = structural_form_similarity(sections_a, sections_b)
    else:
        structural = 0.0

    overall = (
        w.get("melodic", 0.0) * melodic
        + w.get("rhythmic", 0.0) * rhythmic
        + w.get("harmonic", 0.0) * harmonic
        + w.get("structural", 0.0) * structural
    )
    overall = float(np.clip(overall, 0.0, 1.0))

    # Verdict thresholds (heuristic, not legal advice)
    if overall >= 0.85:
        verdict = "very_high"
    elif overall >= 0.70:
        verdict = "high"
    elif overall >= 0.45:
        verdict = "moderate"
    else:
        verdict = "low"

    return SimilarityReport(
        melodic_sim=float(melodic),
        rhythmic_sim=float(rhythmic),
        harmonic_sim=float(harmonic),
        structural_sim=float(structural),
        overall=overall,
        verdict=verdict,
        detail={
            "contour_correlation": float(contour),
            "interval_edit_sim": float(interval),
            "pitch_class_distribution": float(pcd),
            "interval_lcs_ratio": float(lcs),
        },
    )


# ---------------------------------------------------------------------------
# Legal-threshold heuristic
# ---------------------------------------------------------------------------

def substantial_similarity_threshold(report: SimilarityReport) -> bool:
    """Heuristic legal threshold: overall >= 0.7 flags potential infringement concern.

    This is a *screening* heuristic only.  Actual substantial-similarity
    analysis in US copyright law weighs both *extrinsic* (objective,
    analytical) and *intrinsic* (subjective, audience) tests and depends
    heavily on which elements are protectable.  A True return means the
    quantitative evidence warrants closer inspection by a human expert.
    """
    return report.overall >= 0.7
