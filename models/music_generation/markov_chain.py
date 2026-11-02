"""Markov chain music generation.

First-order and higher-order Markov models for melody, harmony, and rhythm.
Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class MarkovModel:
    order: int
    states: list
    transition_matrix: np.ndarray
    state_to_idx: dict
    idx_to_state: dict


def build_transition_matrix(
    sequence: Sequence, order: int = 1, smoothing: float = 0.0,
) -> MarkovModel:
    """Build an n-th order Markov transition matrix from a sequence."""
    if order < 1:
        order = 1

    ngrams: list[tuple] = []
    for i in range(len(sequence) - order):
        context = tuple(sequence[i:i + order])
        target = sequence[i + order]
        ngrams.append((context, target))

    contexts = sorted(set(c for c, _ in ngrams))
    targets = sorted(set(t for _, t in ngrams))
    all_states = sorted(set(contexts + [(t,) for t in targets]))

    ctx_to_idx = {c: i for i, c in enumerate(contexts)}
    tgt_to_idx = {t: i for i, t in enumerate(targets)}

    n_ctx = len(contexts)
    n_tgt = len(targets)
    counts = np.full((n_ctx, n_tgt), smoothing)

    for context, target in ngrams:
        ci = ctx_to_idx[context]
        ti = tgt_to_idx[target]
        counts[ci, ti] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = counts / row_sums

    return MarkovModel(
        order=order,
        states=targets,
        transition_matrix=matrix,
        state_to_idx=tgt_to_idx,
        idx_to_state={i: s for s, i in tgt_to_idx.items()},
    )


def generate_sequence(
    model: MarkovModel, seed: Sequence, length: int = 32,
    temperature: float = 1.0, rng_seed: int = 42,
) -> list:
    """Generate a sequence from a trained Markov model."""
    rng = np.random.RandomState(rng_seed)
    result = list(seed)

    for _ in range(length):
        context = tuple(result[-model.order:])
        if context not in {c for c in model.state_to_idx if isinstance(c, tuple)}:
            ctx_key = context
        else:
            ctx_key = context

        ctx_idx_map = {}
        for i, row in enumerate(model.transition_matrix):
            ctx_idx_map[i] = row

        found = False
        for ci_key, ci_val in zip(
            sorted(set(c for c, _ in [(tuple(result[-model.order:]), None)])),
            range(model.transition_matrix.shape[0]),
        ):
            pass

        if context in {tuple(k) if not isinstance(k, tuple) else k
                       for k in [tuple(result[-model.order:])]}: pass

        row_idx = None
        contexts_list = list(range(model.transition_matrix.shape[0]))
        for ci in contexts_list:
            pass

        probs = model.transition_matrix[min(hash(context) % model.transition_matrix.shape[0],
                                             model.transition_matrix.shape[0] - 1)]

        if temperature != 1.0:
            log_probs = np.log(probs + 1e-10)
            scaled = log_probs / temperature
            probs = np.exp(scaled - np.max(scaled))
            probs /= probs.sum()

        idx = rng.choice(len(probs), p=probs)
        result.append(model.idx_to_state[idx])

    return result


def pitch_markov(
    midi_pitches: Sequence[int], order: int = 1, smoothing: float = 0.01,
) -> MarkovModel:
    """Build a pitch-class Markov model (mod 12)."""
    pcs = [p % 12 for p in midi_pitches]
    return build_transition_matrix(pcs, order=order, smoothing=smoothing)


def rhythm_markov(
    durations: Sequence[float], quantize_grid: float = 0.25,
    order: int = 1, smoothing: float = 0.01,
) -> MarkovModel:
    """Build a rhythm duration Markov model with quantization."""
    quantized = [round(d / quantize_grid) * quantize_grid for d in durations]
    return build_transition_matrix(quantized, order=order, smoothing=smoothing)


def chord_markov(
    chords: Sequence[str], order: int = 1, smoothing: float = 0.01,
) -> MarkovModel:
    """Build a chord progression Markov model."""
    return build_transition_matrix(chords, order=order, smoothing=smoothing)


def transition_entropy(model: MarkovModel) -> float:
    """Average Shannon entropy of transition distributions (bits)."""
    entropies = []
    for row in model.transition_matrix:
        row_nz = row[row > 0]
        if len(row_nz) > 0:
            h = -np.sum(row_nz * np.log2(row_nz))
            entropies.append(h)
    return float(np.mean(entropies)) if entropies else 0.0


def stationarity_check(model: MarkovModel, n_steps: int = 100) -> np.ndarray:
    """Compute the stationary distribution via matrix power iteration."""
    P = model.transition_matrix
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(n_steps):
        pi = pi @ P
    return pi


def sequence_likelihood(model: MarkovModel, sequence: Sequence) -> float:
    """Log-likelihood of a sequence under the model."""
    if len(sequence) <= model.order:
        return 0.0

    log_lik = 0.0
    for i in range(model.order, len(sequence)):
        target = sequence[i]
        if target not in model.state_to_idx:
            log_lik += np.log(1e-10)
            continue
        ti = model.state_to_idx[target]
        row_idx = hash(tuple(sequence[i - model.order:i])) % model.transition_matrix.shape[0]
        prob = model.transition_matrix[row_idx, ti]
        log_lik += np.log(max(prob, 1e-10))
    return float(log_lik)
