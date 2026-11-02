"""Musical expectation and prediction models.

Narmour implication-realization, Markov chain transition probabilities,
information-theoretic surprise (Shannon entropy, information content).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ImplicationRealization:
    """Narmour implication-realization analysis result."""
    implied_interval: int
    realized_interval: int
    registral_direction: str  # "same", "reversal", "change"
    intervallic_difference: str  # "similar", "different"
    closure: bool


def narmour_implication(interval1: int, interval2: int) -> ImplicationRealization:
    """Analyse a pair of successive intervals via Narmour's I-R model.

    Parameters
    ----------
    interval1 : int  Implicative interval in semitones (signed).
    interval2 : int  Realized interval in semitones (signed).
    """
    same_dir = (interval1 > 0 and interval2 > 0) or (interval1 < 0 and interval2 < 0)
    if interval1 == 0 or interval2 == 0:
        reg_dir = "same"
    elif same_dir:
        reg_dir = "same"
    else:
        reg_dir = "reversal"

    abs1, abs2 = abs(interval1), abs(interval2)
    int_diff = "similar" if abs(abs1 - abs2) <= 2 else "different"

    closure = (reg_dir == "reversal") and (abs2 <= abs1)

    return ImplicationRealization(
        implied_interval=interval1,
        realized_interval=interval2,
        registral_direction=reg_dir,
        intervallic_difference=int_diff,
        closure=closure,
    )


def build_transition_matrix(
    sequence: np.ndarray,
    n_states: int = 12,
) -> np.ndarray:
    """Build a first-order Markov transition matrix from a pitch-class sequence.

    Returns row-normalised matrix T where T[i,j] = P(next=j | current=i).
    """
    T = np.zeros((n_states, n_states))
    seq = np.asarray(sequence, dtype=int) % n_states
    for i in range(len(seq) - 1):
        T[seq[i], seq[i + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return T / row_sums


def information_content(
    event: int,
    context: np.ndarray,
    n_states: int = 12,
) -> float:
    """Compute information content (surprisal) of *event* given preceding *context*.

    IC(x) = −log₂ P(x | context)
    """
    T = build_transition_matrix(context, n_states)
    if len(context) == 0:
        return np.log2(n_states)
    prev = int(context[-1]) % n_states
    prob = T[prev, event % n_states]
    if prob < 1e-12:
        return np.log2(n_states)  # maximum surprise
    return -np.log2(prob)


def entropy_rate(sequence: np.ndarray, n_states: int = 12) -> float:
    """Estimate the entropy rate of a pitch-class sequence."""
    T = build_transition_matrix(sequence, n_states)
    # stationary distribution via eigenvector
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.abs(eigvecs[:, idx])
    pi_sum = pi.sum()
    if pi_sum < 1e-12:
        return 0.0
    pi = pi / pi_sum

    H = 0.0
    for i in range(n_states):
        for j in range(n_states):
            if T[i, j] > 1e-12:
                H -= pi[i] * T[i, j] * np.log2(T[i, j])
    return float(H)


def surprise_curve(sequence: np.ndarray, n_states: int = 12) -> np.ndarray:
    """Compute information content at each position of a sequence."""
    seq = np.asarray(sequence, dtype=int) % n_states
    ic = np.zeros(len(seq))
    for i in range(1, len(seq)):
        ic[i] = information_content(seq[i], seq[:i], n_states)
    return ic


def prediction_accuracy(
    sequence: np.ndarray,
    n_states: int = 12,
) -> float:
    """Fraction of correctly predicted next events using MAP from transition matrix."""
    if len(sequence) < 3:
        return 0.0
    seq = np.asarray(sequence, dtype=int) % n_states
    # train on first half, test on second
    mid = len(seq) // 2
    T = build_transition_matrix(seq[:mid], n_states)
    correct = 0
    total = 0
    for i in range(mid, len(seq) - 1):
        predicted = int(np.argmax(T[seq[i]]))
        if predicted == seq[i + 1]:
            correct += 1
        total += 1
    return correct / max(total, 1)


def melodic_expectancy_profile(
    intervals: np.ndarray,
) -> np.ndarray:
    """Score each interval pair for expectancy fulfilment via Narmour.

    Returns array of length len(intervals)-1; 1 = fulfilled, 0 = violated.
    """
    results = np.zeros(len(intervals) - 1)
    for i in range(len(intervals) - 1):
        ir = narmour_implication(int(intervals[i]), int(intervals[i + 1]))
        score = 0.0
        if ir.registral_direction == "same":
            score += 0.5
        if ir.intervallic_difference == "similar":
            score += 0.5
        results[i] = score
    return results
