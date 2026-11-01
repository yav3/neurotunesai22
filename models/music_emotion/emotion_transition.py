"""Emotion state-transition models for music.

Transition matrices between emotional states, trajectory smoothing,
transition probabilities from audio features, and emotion inertia.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class EmotionState(Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CALM = "calm"
    EXCITED = "excited"
    TENDER = "tender"
    ANXIOUS = "anxious"
    NOSTALGIC = "nostalgic"


EMOTION_INDEX = {e: i for i, e in enumerate(EmotionState)}
N_EMOTIONS = len(EmotionState)

# Plausible musical emotion transition priors (row = from, col = to)
_BASE_TRANSITIONS = np.array([
    # hap  sad  ang  cal  exc  ten  anx  nos
    [0.3, 0.02, 0.05, 0.15, 0.25, 0.1, 0.03, 0.1],  # happy
    [0.05, 0.35, 0.05, 0.15, 0.02, 0.15, 0.08, 0.15],  # sad
    [0.05, 0.08, 0.3, 0.05, 0.2, 0.02, 0.25, 0.05],  # angry
    [0.15, 0.1, 0.02, 0.35, 0.05, 0.2, 0.03, 0.1],  # calm
    [0.2, 0.02, 0.15, 0.05, 0.3, 0.05, 0.15, 0.08],  # excited
    [0.12, 0.12, 0.02, 0.25, 0.05, 0.3, 0.04, 0.1],  # tender
    [0.05, 0.1, 0.2, 0.05, 0.15, 0.03, 0.35, 0.07],  # anxious
    [0.1, 0.15, 0.03, 0.12, 0.05, 0.12, 0.05, 0.38],  # nostalgic
])


@dataclass
class EmotionTrajectory:
    states: List[EmotionState]
    probabilities: np.ndarray  # (T, N_EMOTIONS)
    timestamps: np.ndarray


def transition_matrix_from_features(
    valence: float,
    arousal: float,
    tempo_bpm: float,
) -> np.ndarray:
    """Adapt the base transition matrix given audio features.

    Higher valence biases toward happy/excited; lower toward sad/nostalgic.
    Higher arousal biases toward excited/angry; lower toward calm/tender.
    """
    bias = np.zeros(N_EMOTIONS)
    v = np.clip(valence, -1, 1)
    a = np.clip(arousal, -1, 1)

    bias[EMOTION_INDEX[EmotionState.HAPPY]] = 0.3 * v + 0.2 * a
    bias[EMOTION_INDEX[EmotionState.SAD]] = -0.3 * v - 0.2 * a
    bias[EMOTION_INDEX[EmotionState.ANGRY]] = -0.1 * v + 0.3 * a
    bias[EMOTION_INDEX[EmotionState.CALM]] = 0.1 * v - 0.3 * a
    bias[EMOTION_INDEX[EmotionState.EXCITED]] = 0.2 * v + 0.3 * a
    bias[EMOTION_INDEX[EmotionState.TENDER]] = 0.2 * v - 0.2 * a
    bias[EMOTION_INDEX[EmotionState.ANXIOUS]] = -0.2 * v + 0.2 * a
    bias[EMOTION_INDEX[EmotionState.NOSTALGIC]] = -0.1 * v - 0.1 * a

    T = _BASE_TRANSITIONS.copy()
    for i in range(N_EMOTIONS):
        T[i] *= np.exp(bias)
        row_sum = T[i].sum()
        if row_sum > 0:
            T[i] /= row_sum
    return T


def predict_trajectory(
    initial_state: EmotionState,
    n_steps: int,
    transition_matrix: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> EmotionTrajectory:
    """Sample an emotion trajectory from the Markov chain."""
    T = transition_matrix if transition_matrix is not None else _BASE_TRANSITIONS
    rng = np.random.default_rng(seed)

    states = [initial_state]
    probs = np.zeros((n_steps, N_EMOTIONS))
    probs[0] = T[EMOTION_INDEX[initial_state]]

    for t in range(1, n_steps):
        cur = EMOTION_INDEX[states[-1]]
        probs[t] = T[cur]
        next_idx = rng.choice(N_EMOTIONS, p=T[cur])
        states.append(list(EmotionState)[next_idx])

    return EmotionTrajectory(
        states=states,
        probabilities=probs,
        timestamps=np.arange(n_steps, dtype=float),
    )


def smooth_trajectory(
    probabilities: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Moving-average smooth an emotion probability trajectory."""
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(probabilities)
    for c in range(probabilities.shape[1]):
        smoothed[:, c] = np.convolve(probabilities[:, c], kernel, mode="same")
    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return smoothed / row_sums


def emotion_inertia(trajectory: List[EmotionState]) -> float:
    """Fraction of successive states that remain the same."""
    if len(trajectory) < 2:
        return 1.0
    same = sum(1 for a, b in zip(trajectory, trajectory[1:]) if a == b)
    return same / (len(trajectory) - 1)


def dominant_emotion(probabilities: np.ndarray) -> Tuple[EmotionState, float]:
    """Return the overall dominant emotion and its mean probability."""
    mean_probs = probabilities.mean(axis=0)
    idx = int(np.argmax(mean_probs))
    return list(EmotionState)[idx], float(mean_probs[idx])
