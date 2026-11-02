"""Approximate pattern matching for musical sequences.

Finds near-repetitions in melodic, rhythmic, and chroma sequences using
edit-distance variants tuned for music: pitch-interval edit distance
(transposition-invariant), rhythmic edit distance (duration-ratio based),
and chroma sequence alignment (cosine-distance substitution cost).

Also provides Smith-Waterman local alignment for finding the best matching
sub-sequence, and a motif discovery algorithm based on the matrix profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# --- Edit distances ---

def pitch_interval_distance(
    seq_a: Sequence[float],
    seq_b: Sequence[float],
    gap_cost: float = 2.0,
    sub_cost_scale: float = 1.0,
) -> float:
    """Transposition-invariant edit distance on pitch sequences.

    Converts pitches to interval sequences (first differences) then
    computes Levenshtein distance with a continuous substitution cost
    proportional to the interval difference in semitones.
    """
    a = np.diff(np.asarray(seq_a, dtype=float))
    b = np.diff(np.asarray(seq_b, dtype=float))
    m, n = len(a), len(b)
    D = np.zeros((m + 1, n + 1))
    D[:, 0] = np.arange(m + 1) * gap_cost
    D[0, :] = np.arange(n + 1) * gap_cost
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = D[i - 1, j - 1] + sub_cost_scale * abs(a[i - 1] - b[j - 1])
            delete = D[i - 1, j] + gap_cost
            insert = D[i, j - 1] + gap_cost
            D[i, j] = min(sub, delete, insert)
    return float(D[m, n])


def rhythmic_edit_distance(
    durations_a: Sequence[float],
    durations_b: Sequence[float],
    gap_cost: float = 1.0,
) -> float:
    """Edit distance on duration sequences using log-ratio substitution cost.

    Substitution cost = |log2(d_a / d_b)|, so a dotted-quarter vs quarter
    costs log2(1.5) ≈ 0.58, while a quarter vs eighth costs 1.0.
    """
    a = np.asarray(durations_a, dtype=float)
    b = np.asarray(durations_b, dtype=float)
    a = np.maximum(a, 1e-6)
    b = np.maximum(b, 1e-6)
    m, n = len(a), len(b)
    D = np.zeros((m + 1, n + 1))
    D[:, 0] = np.arange(m + 1) * gap_cost
    D[0, :] = np.arange(n + 1) * gap_cost
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = D[i - 1, j - 1] + abs(np.log2(a[i - 1] / b[j - 1]))
            delete = D[i - 1, j] + gap_cost
            insert = D[i, j - 1] + gap_cost
            D[i, j] = min(sub, delete, insert)
    return float(D[m, n])


def chroma_sequence_distance(
    chroma_a: NDArray,
    chroma_b: NDArray,
    gap_cost: float = 0.5,
) -> float:
    """Edit distance on chroma frame sequences using cosine distance as
    the substitution cost."""
    A = np.asarray(chroma_a, dtype=float)
    B = np.asarray(chroma_b, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    m, n = A.shape[0], B.shape[0]
    D = np.zeros((m + 1, n + 1))
    D[:, 0] = np.arange(m + 1) * gap_cost
    D[0, :] = np.arange(n + 1) * gap_cost

    def _cosine_dist(u: NDArray, v: NDArray) -> float:
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu < 1e-12 or nv < 1e-12:
            return 1.0
        return float(1.0 - np.dot(u, v) / (nu * nv))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = D[i - 1, j - 1] + _cosine_dist(A[i - 1], B[j - 1])
            delete = D[i - 1, j] + gap_cost
            insert = D[i, j - 1] + gap_cost
            D[i, j] = min(sub, delete, insert)
    return float(D[m, n])


# --- Smith-Waterman local alignment ---

@dataclass
class LocalAlignment:
    """Result of a Smith-Waterman local alignment."""
    score: float
    start_a: int
    end_a: int
    start_b: int
    end_b: int
    aligned_a: list[float | None]
    aligned_b: list[float | None]


def smith_waterman(
    seq_a: Sequence[float],
    seq_b: Sequence[float],
    match_score: float = 2.0,
    mismatch_penalty: float = -1.0,
    gap_penalty: float = -1.5,
    tolerance: float = 0.5,
) -> LocalAlignment:
    """Smith-Waterman local alignment for numeric sequences.

    Two values 'match' if |a - b| <= tolerance.
    """
    a = list(seq_a)
    b = list(seq_b)
    m, n = len(a), len(b)
    H = np.zeros((m + 1, n + 1))
    traceback = np.zeros((m + 1, n + 1), dtype=int)  # 0=stop, 1=diag, 2=up, 3=left

    best_score = 0.0
    best_i, best_j = 0, 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = match_score if abs(a[i - 1] - b[j - 1]) <= tolerance else mismatch_penalty
            diag = H[i - 1, j - 1] + match
            up = H[i - 1, j] + gap_penalty
            left = H[i, j - 1] + gap_penalty
            H[i, j] = max(0, diag, up, left)
            if H[i, j] == 0:
                traceback[i, j] = 0
            elif H[i, j] == diag:
                traceback[i, j] = 1
            elif H[i, j] == up:
                traceback[i, j] = 2
            else:
                traceback[i, j] = 3
            if H[i, j] > best_score:
                best_score = H[i, j]
                best_i, best_j = i, j

    # Traceback
    aligned_a: list[float | None] = []
    aligned_b: list[float | None] = []
    i, j = best_i, best_j
    while i > 0 and j > 0 and H[i, j] > 0:
        if traceback[i, j] == 1:
            aligned_a.append(a[i - 1])
            aligned_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif traceback[i, j] == 2:
            aligned_a.append(a[i - 1])
            aligned_b.append(None)
            i -= 1
        else:
            aligned_a.append(None)
            aligned_b.append(b[j - 1])
            j -= 1
    aligned_a.reverse()
    aligned_b.reverse()
    start_a = i
    start_b = j

    return LocalAlignment(
        score=float(best_score),
        start_a=start_a,
        end_a=best_i,
        start_b=start_b,
        end_b=best_j,
        aligned_a=aligned_a,
        aligned_b=aligned_b,
    )


# --- Matrix profile for motif discovery ---

@dataclass
class MotifMatch:
    """A pair of approximately matching sub-sequences."""
    index_a: int
    index_b: int
    distance: float
    length: int


def matrix_profile_1d(ts: NDArray, m: int) -> tuple[NDArray, NDArray]:
    """Compute the matrix profile and profile index for a 1-D time series.

    Uses the STAMP algorithm (Yeh et al. 2016): for each sub-sequence of
    length m, finds the nearest-neighbour sub-sequence by z-normalised
    Euclidean distance.

    Parameters
    ----------
    ts : 1-D time series.
    m : sub-sequence length.

    Returns
    -------
    mp : matrix profile (nearest-neighbour distances).
    mpi : matrix profile index (index of the nearest neighbour).
    """
    ts = np.asarray(ts, dtype=float)
    n = len(ts)
    if n < 2 * m:
        return np.array([]), np.array([], dtype=int)

    n_subs = n - m + 1
    mp = np.full(n_subs, np.inf)
    mpi = np.zeros(n_subs, dtype=int)

    # Precompute sub-sequence means and stds
    means = np.array([ts[i:i + m].mean() for i in range(n_subs)])
    stds = np.array([ts[i:i + m].std() for i in range(n_subs)])
    stds = np.maximum(stds, 1e-8)  # avoid division by zero

    for i in range(n_subs):
        sub_i = (ts[i:i + m] - means[i]) / stds[i]
        for j in range(n_subs):
            if abs(i - j) < m // 2:
                continue  # skip trivial matches (overlapping)
            sub_j = (ts[j:j + m] - means[j]) / stds[j]
            d = np.sqrt(np.sum((sub_i - sub_j) ** 2))
            if d < mp[i]:
                mp[i] = d
                mpi[i] = j
    return mp, mpi


def discover_motifs(
    ts: NDArray,
    m: int,
    k: int = 3,
    exclusion_zone: int | None = None,
) -> list[MotifMatch]:
    """Discover the top-k motifs (recurring patterns) in a time series.

    Parameters
    ----------
    ts : 1-D time series.
    m : motif length (sub-sequence length).
    k : number of motifs to return.
    exclusion_zone : minimum index distance between reported motif pairs.
                     Defaults to m // 2.
    """
    mp, mpi = matrix_profile_1d(ts, m)
    if len(mp) == 0:
        return []
    if exclusion_zone is None:
        exclusion_zone = m // 2

    motifs: list[MotifMatch] = []
    used = np.zeros(len(mp), dtype=bool)

    order = np.argsort(mp)
    for idx in order:
        if used[idx]:
            continue
        partner = int(mpi[idx])
        if used[partner]:
            continue
        if abs(idx - partner) < exclusion_zone:
            continue
        motifs.append(MotifMatch(
            index_a=int(idx),
            index_b=partner,
            distance=float(mp[idx]),
            length=m,
        ))
        # Mark exclusion zone
        for ex in range(max(0, idx - exclusion_zone), min(len(mp), idx + exclusion_zone + 1)):
            used[ex] = True
        for ex in range(max(0, partner - exclusion_zone), min(len(mp), partner + exclusion_zone + 1)):
            used[ex] = True
        if len(motifs) >= k:
            break
    return motifs


def normalised_match_score(
    seq_a: Sequence[float],
    seq_b: Sequence[float],
    method: str = "pitch_interval",
) -> float:
    """Normalised similarity in [0, 1] where 1 = identical.

    Parameters
    ----------
    method : 'pitch_interval', 'rhythmic', or 'smith_waterman'.
    """
    a = list(seq_a)
    b = list(seq_b)
    if not a or not b:
        return 0.0
    if method == "pitch_interval":
        d = pitch_interval_distance(a, b)
        max_d = max(len(a), len(b)) * 2.0
        return float(max(0.0, 1.0 - d / max_d))
    elif method == "rhythmic":
        d = rhythmic_edit_distance(a, b)
        max_d = max(len(a), len(b)) * 1.0
        return float(max(0.0, 1.0 - d / max_d))
    elif method == "smith_waterman":
        result = smith_waterman(a, b)
        max_score = min(len(a), len(b)) * 2.0
        return float(result.score / max(max_score, 1.0))
    else:
        raise ValueError(f"Unknown method: {method}")
