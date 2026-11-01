"""Cognitive load estimation for musical passages.

Models the processing demands of harmony, rhythm, and texture on working memory,
based on information-theoretic and psychoacoustic measures.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CognitiveLoadEstimate:
    harmonic: float  # 0..1
    rhythmic: float  # 0..1
    textural: float  # 0..1
    combined: float  # 0..1 weighted average
    category: str  # "low" / "moderate" / "high" / "very_high"


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def harmonic_complexity_load(
    chord_sequence: list[tuple[int, ...]],
    max_entropy: float = 3.5,
) -> float:
    """Estimate cognitive load from a sequence of chords (pitch-class sets).

    Higher chord diversity and larger chords → higher load.
    Returns 0..1 normalised score.
    """
    if not chord_sequence:
        return 0.0
    unique_chords = set(tuple(sorted(set(p % 12 for p in c)) if c else ()) for c in chord_sequence)
    n = len(chord_sequence)
    chord_counts: dict[tuple[int, ...], int] = {}
    for c in chord_sequence:
        key = tuple(sorted(set(p % 12 for p in c)) if c else ())
        chord_counts[key] = chord_counts.get(key, 0) + 1
    probs = np.array(list(chord_counts.values()), dtype=float) / n
    ent = _entropy(probs)
    avg_card = np.mean([len(c) for c in chord_sequence]) if chord_sequence else 0.0
    card_factor = min(avg_card / 6.0, 1.0)
    ent_factor = min(ent / max_entropy, 1.0)
    return float(np.clip(0.6 * ent_factor + 0.4 * card_factor, 0.0, 1.0))


def rhythmic_complexity_load(
    onset_pattern: np.ndarray,
    meter: np.ndarray | None = None,
    max_entropy: float = 3.0,
) -> float:
    """Estimate cognitive load from an onset pattern and metrical grid.

    Syncopation and irregular subdivisions increase load.
    """
    pat = np.asarray(onset_pattern, dtype=float)
    if len(pat) == 0:
        return 0.0
    density = np.sum(pat > 0) / max(len(pat), 1)
    n_onsets = int(np.sum(pat > 0))
    if n_onsets <= 1:
        return float(density * 0.2)
    ioi_positions = np.where(pat > 0)[0]
    iois = np.diff(ioi_positions)
    if len(iois) == 0:
        return float(density * 0.2)
    unique_iois, counts = np.unique(iois, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    ent = _entropy(probs)
    ent_norm = min(ent / max_entropy, 1.0)
    syncopation = 0.0
    if meter is not None and len(meter) == len(pat):
        for i in range(len(pat)):
            if pat[i] > 0:
                weak = meter[i] < np.median(meter)
                if weak:
                    syncopation += 1.0
        syncopation /= max(n_onsets, 1)
    return float(np.clip(0.4 * ent_norm + 0.3 * density + 0.3 * syncopation, 0.0, 1.0))


def textural_density_load(
    n_voices: int,
    register_spread: float,
    max_voices: int = 8,
    max_spread_semitones: float = 48.0,
) -> float:
    """Estimate cognitive load from the number of simultaneous voices and registral spread.

    More voices and wider spread increase attentional demands.
    """
    voice_norm = min(n_voices / max_voices, 1.0)
    spread_norm = min(register_spread / max_spread_semitones, 1.0)
    return float(np.clip(0.65 * voice_norm + 0.35 * spread_norm, 0.0, 1.0))


def combined_cognitive_load(
    harmonic: float,
    rhythmic: float,
    textural: float,
    weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
) -> CognitiveLoadEstimate:
    """Compute weighted combination and categorise the result."""
    w = np.array(weights, dtype=float)
    w = w / max(w.sum(), 1e-12)
    combined = float(w[0] * harmonic + w[1] * rhythmic + w[2] * textural)
    combined = float(np.clip(combined, 0.0, 1.0))
    if combined < 0.25:
        cat = "low"
    elif combined < 0.50:
        cat = "moderate"
    elif combined < 0.75:
        cat = "high"
    else:
        cat = "very_high"
    return CognitiveLoadEstimate(
        harmonic=harmonic,
        rhythmic=rhythmic,
        textural=textural,
        combined=combined,
        category=cat,
    )


def passage_cognitive_profile(
    chord_sequence: list[tuple[int, ...]],
    onset_pattern: np.ndarray,
    n_voices: int,
    register_spread: float,
    meter: np.ndarray | None = None,
    weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
) -> CognitiveLoadEstimate:
    """End-to-end cognitive load estimation for a musical passage."""
    h = harmonic_complexity_load(chord_sequence)
    r = rhythmic_complexity_load(onset_pattern, meter)
    t = textural_density_load(n_voices, register_spread)
    return combined_cognitive_load(h, r, t, weights)
