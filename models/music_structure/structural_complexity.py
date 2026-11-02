"""Structural complexity metrics for music.

Quantifies how complex a piece's structure is via information-theoretic
and graph-theoretic measures applied to the segment label sequence,
the self-similarity matrix, and the transition graph.

Core measures:
- Label entropy (Shannon, normalised)
- Lempel-Ziv complexity (compressibility of the label string)
- Transition entropy (predictability of section ordering)
- Hierarchical depth (nesting levels from a StructureTree)
- Fractal dimension of the self-similarity matrix diagonal structure
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# --- Label-sequence metrics ---

def label_entropy(labels: Sequence[str], normalise: bool = True) -> float:
    """Shannon entropy of the section-label distribution.

    Parameters
    ----------
    labels : sequence of section labels (e.g. ['A','B','A','C','A','B']).
    normalise : if True, divide by log2(n_unique) to give H in [0, 1].
    """
    if not labels:
        return 0.0
    from collections import Counter
    counts = Counter(labels)
    n = len(labels)
    probs = np.array([c / n for c in counts.values()])
    H = -np.sum(probs * np.log2(probs + 1e-15))
    if normalise and len(counts) > 1:
        H /= np.log2(len(counts))
    return float(H)


def lempel_ziv_complexity(labels: Sequence[str]) -> float:
    """Normalised Lempel-Ziv complexity of a label string.

    Counts the number of distinct sub-patterns discovered by a left-to-right
    scan (Lempel & Ziv 1976). Normalised by n / log2(n) so the result is
    roughly 1.0 for a random sequence and < 1.0 for structured ones.
    """
    if not labels:
        return 0.0
    s = list(labels)
    n = len(s)
    i = 0
    c = 1  # complexity counter
    k = 1
    l = 1
    while k + l <= n:
        sub = s[k: k + l]
        # Check if sub appears in s[i:k+l-1]
        found = False
        window = s[i: k + l - 1]
        for start in range(len(window) - l + 1):
            if window[start: start + l] == sub:
                found = True
                break
        if found:
            l += 1
        else:
            c += 1
            k += l
            l = 1
            i = 0  # reset scan origin (full history variant)
    # Normalise
    norm = n / max(math.log2(n + 1), 1.0)
    return c / norm


def transition_entropy(labels: Sequence[str], normalise: bool = True) -> float:
    """Conditional entropy of transitions between consecutive section labels.

    H(L_{t+1} | L_t) measures how predictable the next section is given
    the current one. Low = predictable (e.g. always A→B→A→B).
    """
    if len(labels) < 2:
        return 0.0
    from collections import Counter
    bigrams = [(labels[i], labels[i + 1]) for i in range(len(labels) - 1)]
    unigram_counts = Counter(labels[:-1])
    bigram_counts = Counter(bigrams)
    H = 0.0
    n_transitions = len(bigrams)
    for (a, b), count in bigram_counts.items():
        p_ab = count / n_transitions
        p_b_given_a = count / unigram_counts[a]
        H -= p_ab * math.log2(p_b_given_a + 1e-15)
    if normalise:
        n_unique = len(set(labels))
        if n_unique > 1:
            H /= math.log2(n_unique)
    return float(H)


def repetition_rate(labels: Sequence[str]) -> float:
    """Fraction of labels that repeat a previously seen label."""
    if not labels:
        return 0.0
    seen: set[str] = set()
    reps = 0
    for lbl in labels:
        if lbl in seen:
            reps += 1
        seen.add(lbl)
    return reps / len(labels)


# --- Self-similarity matrix metrics ---

def ssm_block_entropy(ssm: NDArray, n_blocks: int = 8) -> float:
    """Entropy of the block-diagonal structure of a self-similarity matrix.

    Divides the SSM into n_blocks × n_blocks grid and computes Shannon
    entropy of the mean similarity per block — captures how uniform vs
    varied the block structure is.
    """
    ssm = np.asarray(ssm, dtype=float)
    n = ssm.shape[0]
    if n < n_blocks:
        n_blocks = max(n, 1)
    block_size = n // n_blocks
    means = np.zeros(n_blocks * n_blocks)
    idx = 0
    for i in range(n_blocks):
        for j in range(n_blocks):
            r0, r1 = i * block_size, (i + 1) * block_size
            c0, c1 = j * block_size, (j + 1) * block_size
            means[idx] = ssm[r0:r1, c0:c1].mean()
            idx += 1
    # Normalise to a distribution
    total = means.sum()
    if total < 1e-12:
        return 0.0
    probs = means / total
    probs = probs[probs > 1e-15]
    return float(-np.sum(probs * np.log2(probs)))


def diagonal_periodicity(ssm: NDArray) -> NDArray:
    """Auto-correlation of diagonal stripes to detect periodic structure.

    For each lag k, computes the mean similarity along the k-th diagonal.
    Periodic structure shows up as peaks in this lag spectrum.

    Returns
    -------
    lag_spectrum : 1-D array of mean diagonal similarity for lag 0..n-1.
    """
    ssm = np.asarray(ssm, dtype=float)
    n = ssm.shape[0]
    spectrum = np.zeros(n)
    for k in range(n):
        diag = np.diag(ssm, k)
        if len(diag) > 0:
            spectrum[k] = diag.mean()
    return spectrum


def fractal_dimension_ssm(ssm: NDArray, scales: int = 8) -> float:
    """Box-counting fractal dimension of the SSM's thresholded binary image.

    Measures the self-similar complexity of the recurrence structure.
    A perfectly block-diagonal matrix has D ≈ 1; a random matrix has D ≈ 2.
    """
    ssm = np.asarray(ssm, dtype=float)
    n = ssm.shape[0]
    if n < 4:
        return 1.0
    # Threshold at median
    binary = (ssm > np.median(ssm)).astype(int)
    # Box counting
    sizes = []
    counts = []
    for s in range(1, min(scales + 1, n)):
        box_size = max(1, n // (2 ** s))
        if box_size < 1:
            break
        count = 0
        for i in range(0, n, box_size):
            for j in range(0, n, box_size):
                block = binary[i:i + box_size, j:j + box_size]
                if block.sum() > 0:
                    count += 1
        if count > 0:
            sizes.append(box_size)
            counts.append(count)
    if len(sizes) < 2:
        return 1.0
    log_sizes = np.log(np.array(sizes, dtype=float))
    log_counts = np.log(np.array(counts, dtype=float))
    # Linear regression: D = -slope
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(-coeffs[0])


# --- Composite complexity score ---

@dataclass
class StructuralComplexity:
    """Aggregate structural complexity report."""
    label_entropy: float
    lz_complexity: float
    transition_entropy: float
    repetition_rate: float
    ssm_block_entropy: float | None
    fractal_dimension: float | None
    composite_score: float


def compute_structural_complexity(
    labels: Sequence[str],
    ssm: NDArray | None = None,
    weights: tuple[float, ...] = (0.25, 0.25, 0.2, 0.15, 0.15),
) -> StructuralComplexity:
    """Compute a composite structural complexity score.

    Parameters
    ----------
    labels : section label sequence.
    ssm : optional self-similarity matrix for SSM-derived metrics.
    weights : (label_entropy, lz, transition, ssm_block, fractal).
    """
    le = label_entropy(labels)
    lz = lempel_ziv_complexity(labels)
    te = transition_entropy(labels)
    rr = repetition_rate(labels)

    sbe = ssm_block_entropy(ssm) if ssm is not None else None
    fd = fractal_dimension_ssm(ssm) if ssm is not None else None

    # Composite from available components
    vals = [le, lz, te]
    ws = list(weights[:3])
    if sbe is not None:
        vals.append(min(sbe / 6.0, 1.0))  # normalise block entropy
        ws.append(weights[3] if len(weights) > 3 else 0.15)
    if fd is not None:
        vals.append(min((fd - 1.0), 1.0))  # D in [1,2] -> [0,1]
        ws.append(weights[4] if len(weights) > 4 else 0.15)

    total_w = sum(ws)
    composite = sum(v * w for v, w in zip(vals, ws)) / max(total_w, 1e-12)

    return StructuralComplexity(
        label_entropy=le,
        lz_complexity=lz,
        transition_entropy=te,
        repetition_rate=rr,
        ssm_block_entropy=sbe,
        fractal_dimension=fd,
        composite_score=float(composite),
    )
