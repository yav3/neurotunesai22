"""Harmonic progression similarity for copyright analysis.

Chord sequence alignment with functional equivalence classes,
Roman-numeral abstraction, and statistical n-gram significance
testing for infringement detection.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ── Chord representation ───────────────────────────────────────────

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Roman numeral function → semitone offset from tonic in major key
ROMAN_TO_SEMITONE: dict[str, int] = {
    "I": 0, "bII": 1, "II": 2, "bIII": 3, "III": 4,
    "IV": 5, "#IV": 6, "V": 7, "bVI": 8, "VI": 9,
    "bVII": 10, "VII": 11,
}

# Quality equivalences: merge enharmonic spellings
QUALITY_MAP: dict[str, str] = {
    "maj": "M", "major": "M", "M": "M", "": "M",
    "min": "m", "minor": "m", "m": "m",
    "dim": "d", "diminished": "d", "o": "d",
    "aug": "A", "augmented": "A", "+": "A",
    "dom7": "7", "7": "7",
    "maj7": "M7", "M7": "M7",
    "min7": "m7", "m7": "m7",
    "dim7": "d7", "o7": "d7",
    "sus2": "s2", "sus4": "s4",
}


def normalise_quality(q: str) -> str:
    return QUALITY_MAP.get(q, q)


def chord_to_pc(root: str, quality: str = "M") -> tuple[int, str]:
    """Parse a chord label into (pitch class int, normalised quality)."""
    root_clean = root.strip().capitalize()
    if root_clean.endswith("B") and len(root_clean) > 1 and root_clean != "B":
        root_clean = root_clean[:-1] + "b"
    flat_map = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
                "Ab": "G#", "Bb": "A#", "Cb": "B"}
    root_clean = flat_map.get(root_clean, root_clean)
    pc = PITCH_CLASSES.index(root_clean) if root_clean in PITCH_CLASSES else 0
    return pc, normalise_quality(quality)


# ── Key-invariant representation ───────────────────────────────────

def chords_to_intervals(roots: Sequence[int]) -> list[int]:
    """Convert root pitch-class sequence to interval sequence (mod 12)."""
    return [(roots[i + 1] - roots[i]) % 12 for i in range(len(roots) - 1)]


def chords_to_roman(roots: Sequence[int], tonic: int = 0) -> list[str]:
    """Map root pitch-classes to Roman numeral functions relative to *tonic*."""
    inv = {v: k for k, v in ROMAN_TO_SEMITONE.items()}
    return [inv.get((r - tonic) % 12, "?") for r in roots]


# ── Edit-distance alignment ───────────────────────────────────────

def _chord_edit_distance(
    seq_a: list[tuple[int, str]],
    seq_b: list[tuple[int, str]],
    quality_weight: float = 0.5,
) -> float:
    """Weighted edit distance between two chord sequences.

    Substitution cost for two chords sharing the same root (mod 12) but
    different quality is *quality_weight*; completely different chords cost 1.
    """
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            ra, qa = seq_a[i - 1]
            rb, qb = seq_b[j - 1]
            if ra == rb and qa == qb:
                cost = 0.0
            elif ra == rb:
                cost = quality_weight
            else:
                cost = 1.0
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[m, n])


def harmonic_sequence_similarity(
    seq_a: list[tuple[int, str]],
    seq_b: list[tuple[int, str]],
    quality_weight: float = 0.5,
) -> float:
    """Normalised similarity ∈ [0, 1] between two chord sequences."""
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    dist = _chord_edit_distance(seq_a, seq_b, quality_weight)
    max_len = max(len(seq_a), len(seq_b))
    return 1.0 - dist / max_len


# ── Transposition-invariant comparison ─────────────────────────────

def _transpose_seq(seq: list[tuple[int, str]], semitones: int) -> list[tuple[int, str]]:
    return [((r + semitones) % 12, q) for r, q in seq]


def transposition_invariant_harmonic_similarity(
    seq_a: list[tuple[int, str]],
    seq_b: list[tuple[int, str]],
    quality_weight: float = 0.5,
) -> tuple[float, int]:
    """Best similarity over all 12 transpositions; returns (similarity, best_shift)."""
    if not seq_a and not seq_b:
        return 1.0, 0
    if not seq_a or not seq_b:
        return 0.0, 0
    best_sim = -1.0
    best_shift = 0
    for t in range(12):
        shifted = _transpose_seq(seq_b, t)
        sim = harmonic_sequence_similarity(seq_a, shifted, quality_weight)
        if sim > best_sim:
            best_sim = sim
            best_shift = t
    return best_sim, best_shift


# ── Functional equivalence classes ─────────────────────────────────

FUNCTION_CLASSES: dict[str, list[str]] = {
    "tonic": ["I", "III", "VI"],
    "subdominant": ["II", "IV"],
    "dominant": ["V", "VII"],
}


def roman_to_function(numeral: str) -> str:
    """Map a Roman numeral to its functional class (tonic/subdominant/dominant)."""
    base = numeral.upper().rstrip("7O+").lstrip("B#")
    for func, numerals in FUNCTION_CLASSES.items():
        if base in numerals:
            return func
    return "other"


def functional_reduction(roman_seq: list[str]) -> list[str]:
    """Reduce a Roman numeral sequence to functional class labels."""
    return [roman_to_function(r) for r in roman_seq]


def functional_similarity(seq_a: list[str], seq_b: list[str]) -> float:
    """Similarity of two Roman numeral sequences at the functional level."""
    fa = functional_reduction(seq_a)
    fb = functional_reduction(seq_b)
    if not fa and not fb:
        return 1.0
    if not fa or not fb:
        return 0.0
    m, n = len(fa), len(fb)
    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if fa[i - 1] == fb[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    dist = float(dp[m, n])
    return 1.0 - dist / max(m, n)


# ── N-gram overlap with significance ──────────────────────────────

def chord_ngram_overlap(
    seq_a: list[tuple[int, str]],
    seq_b: list[tuple[int, str]],
    n: int = 3,
) -> float:
    """Jaccard overlap of chord n-grams between two progressions."""
    if len(seq_a) < n or len(seq_b) < n:
        return 0.0
    grams_a = {tuple(seq_a[i:i + n]) for i in range(len(seq_a) - n + 1)}
    grams_b = {tuple(seq_b[i:i + n]) for i in range(len(seq_b) - n + 1)}
    union = grams_a | grams_b
    if not union:
        return 0.0
    return len(grams_a & grams_b) / len(union)


def ngram_significance(
    overlap: float,
    len_a: int,
    len_b: int,
    n: int = 3,
    vocab_size: int = 48,
) -> float:
    """p-value for observed n-gram overlap under uniform random baseline.

    Models each n-gram as drawn uniformly from vocab_size^n possibilities.
    The expected Jaccard under independence is ~(na·nb) / (na + nb - na·nb/V^n)
    where na, nb are the number of n-grams from each sequence.
    Returns an approximate upper-tail p-value via normal approximation to the
    binomial indicator sum.
    """
    na = max(len_a - n + 1, 0)
    nb = max(len_b - n + 1, 0)
    if na == 0 or nb == 0:
        return 1.0
    v = vocab_size ** n
    p_single = 1.0 / max(v, 1)
    expected_matches = na * nb * p_single
    variance = na * nb * p_single * (1.0 - p_single)
    if variance < 1e-12:
        return 0.0 if overlap > 0 else 1.0
    observed_matches = overlap * (na + nb) / (1.0 + overlap) if overlap < 1.0 else min(na, nb)
    z = (observed_matches - expected_matches) / np.sqrt(variance)
    from scipy.stats import norm as _norm
    return float(1.0 - _norm.cdf(z))


# ── Sliding-window harmonic comparison ─────────────────────────────

@dataclass
class HarmonicMatchSegment:
    offset_a: int
    offset_b: int
    length: int
    similarity: float
    transposition: int


def sliding_harmonic_compare(
    prog_a: list[tuple[int, str]],
    prog_b: list[tuple[int, str]],
    window: int = 4,
    threshold: float = 0.8,
) -> list[HarmonicMatchSegment]:
    """Find all window-sized segments of prog_a with high similarity to some
    segment of prog_b (best over transpositions)."""
    if len(prog_a) < window or len(prog_b) < window:
        return []
    matches: list[HarmonicMatchSegment] = []
    for i in range(len(prog_a) - window + 1):
        sub_a = prog_a[i:i + window]
        best_sim = 0.0
        best_j = 0
        best_t = 0
        for j in range(len(prog_b) - window + 1):
            sub_b = prog_b[j:j + window]
            sim, t = transposition_invariant_harmonic_similarity(sub_a, sub_b)
            if sim > best_sim:
                best_sim = sim
                best_j = j
                best_t = t
        if best_sim >= threshold:
            matches.append(HarmonicMatchSegment(i, best_j, window, best_sim, best_t))
    return matches
