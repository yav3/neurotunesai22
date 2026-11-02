"""Sight-reading difficulty estimation and practice generation.

Models sight-reading complexity based on rhythmic density, pitch range,
accidental frequency, interval difficulty, key complexity, and temporal
irregularity.  Generates progressive difficulty sequences for practice.

Based on Kopiez & Lee (2006) sight-reading predictors, Lehmann & McArthur
(2002) practice strategies.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SightReadingProfile:
    overall_difficulty: float        # [0, 1]
    rhythmic_complexity: float       # [0, 1]
    pitch_complexity: float          # [0, 1]
    key_complexity: float            # [0, 1]
    interval_difficulty: float       # [0, 1]
    accidental_density: float        # [0, 1]
    temporal_irregularity: float     # [0, 1]


@dataclass
class PracticeExercise:
    pitches: np.ndarray
    durations: np.ndarray
    difficulty: float
    focus_area: str


INTERVAL_DIFFICULTY = {
    0: 0.0, 1: 0.3, 2: 0.2, 3: 0.25, 4: 0.25,
    5: 0.15, 6: 0.6, 7: 0.1,
    8: 0.35, 9: 0.4, 10: 0.5, 11: 0.55, 12: 0.3,
}

KEY_ACCIDENTALS = {
    0: 0, 1: 5, 2: 2, 3: 3, 4: 4, 5: 1, 6: 6,
    7: 1, 8: 4, 9: 3, 10: 2, 11: 5,
}


def rhythmic_complexity_score(durations: np.ndarray) -> float:
    """Rhythmic complexity from duration variance and syncopation potential."""
    if len(durations) < 2:
        return 0.0
    cv = float(np.std(durations) / (np.mean(durations) + 1e-10))
    unique_ratio = len(np.unique(np.round(durations, 3))) / len(durations)
    return float(np.clip(0.4 * cv + 0.6 * unique_ratio, 0, 1))


def pitch_complexity_score(pitches: np.ndarray) -> float:
    """Pitch complexity from range, variance, and chromatic density."""
    if len(pitches) < 2:
        return 0.0
    pitch_range = float(np.ptp(pitches))
    range_norm = np.clip(pitch_range / 36.0, 0, 1)
    variance_norm = np.clip(float(np.std(pitches)) / 12.0, 0, 1)
    pcs = set(int(p) % 12 for p in pitches)
    chromatic = len(pcs) / 12.0
    return float(0.3 * range_norm + 0.3 * variance_norm + 0.4 * chromatic)


def interval_difficulty_score(pitches: np.ndarray) -> float:
    """Average interval difficulty weighted by Kopiez & Lee (2006) model."""
    if len(pitches) < 2:
        return 0.0
    intervals = np.abs(np.diff(pitches))
    difficulties = np.array([
        INTERVAL_DIFFICULTY.get(min(int(i), 12), 0.7) for i in intervals
    ])
    return float(np.mean(difficulties))


def key_complexity_score(key_root: int) -> float:
    """Key complexity based on number of accidentals (0=C, 1=C#, etc.)."""
    n_acc = KEY_ACCIDENTALS.get(key_root % 12, 0)
    return np.clip(n_acc / 6.0, 0, 1)


def accidental_density(pitches: np.ndarray, key_root: int = 0) -> float:
    """Fraction of notes that are accidentals relative to the key."""
    major_scale = {(key_root + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11]}
    pcs = [int(p) % 12 for p in pitches]
    if not pcs:
        return 0.0
    n_acc = sum(1 for pc in pcs if pc not in major_scale)
    return n_acc / len(pcs)


def temporal_irregularity(durations: np.ndarray) -> float:
    """Temporal irregularity via normalized pairwise variability index."""
    if len(durations) < 2:
        return 0.0
    pairs = np.abs(np.diff(durations))
    means = (durations[:-1] + durations[1:]) / 2.0
    means = np.maximum(means, 1e-10)
    npvi = 100.0 * float(np.mean(pairs / means))
    return float(np.clip(npvi / 100.0, 0, 1))


def estimate_difficulty(
    pitches: np.ndarray,
    durations: np.ndarray,
    key_root: int = 0,
    weights: Optional[Dict[str, float]] = None,
) -> SightReadingProfile:
    """Estimate overall sight-reading difficulty."""
    if weights is None:
        weights = {
            "rhythmic": 0.25, "pitch": 0.20, "key": 0.10,
            "interval": 0.20, "accidental": 0.15, "temporal": 0.10,
        }
    rc = rhythmic_complexity_score(durations)
    pc = pitch_complexity_score(pitches)
    kc = key_complexity_score(key_root)
    idf = interval_difficulty_score(pitches)
    ad = accidental_density(pitches, key_root)
    ti = temporal_irregularity(durations)
    overall = (
        weights.get("rhythmic", 0.25) * rc
        + weights.get("pitch", 0.20) * pc
        + weights.get("key", 0.10) * kc
        + weights.get("interval", 0.20) * idf
        + weights.get("accidental", 0.15) * ad
        + weights.get("temporal", 0.10) * ti
    )
    return SightReadingProfile(
        overall_difficulty=float(np.clip(overall, 0, 1)),
        rhythmic_complexity=rc,
        pitch_complexity=pc,
        key_complexity=kc,
        interval_difficulty=idf,
        accidental_density=ad,
        temporal_irregularity=ti,
    )


def generate_exercise(
    target_difficulty: float,
    n_notes: int = 16,
    focus: str = "mixed",
    key_root: int = 0,
    seed: int = 42,
) -> PracticeExercise:
    """Generate a sight-reading exercise at the target difficulty level."""
    rng = np.random.default_rng(seed)
    scale = [(key_root + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11]]

    if focus == "pitch":
        octave_spread = int(1 + 3 * target_difficulty)
        pitches_list = []
        for _ in range(n_notes):
            oct = rng.integers(4, 4 + octave_spread + 1)
            degree = rng.choice(scale)
            pitches_list.append(12 * oct + degree)
        pitches = np.array(pitches_list, dtype=float)
        durations = np.full(n_notes, 0.5)
    elif focus == "rhythm":
        pitches = np.full(n_notes, 60 + key_root, dtype=float)
        base_durs = [0.25, 0.5, 1.0, 0.125, 0.75]
        n_types = max(2, int(2 + 3 * target_difficulty))
        pool = base_durs[:n_types]
        durations = np.array(rng.choice(pool, size=n_notes))
    else:
        octave_spread = int(1 + 2 * target_difficulty)
        pitches_list = []
        for _ in range(n_notes):
            oct = rng.integers(4, 4 + octave_spread + 1)
            degree = rng.choice(scale)
            if rng.random() < target_difficulty * 0.3:
                degree = (degree + rng.choice([-1, 1])) % 12
            pitches_list.append(12 * oct + degree)
        pitches = np.array(pitches_list, dtype=float)
        base_durs = [0.25, 0.5, 1.0, 0.125, 0.75, 1.5]
        n_types = max(2, int(2 + 3 * target_difficulty))
        pool = base_durs[:n_types]
        durations = np.array(rng.choice(pool, size=n_notes))

    actual = estimate_difficulty(pitches, durations, key_root)
    return PracticeExercise(
        pitches=pitches,
        durations=durations,
        difficulty=actual.overall_difficulty,
        focus_area=focus,
    )


def progressive_sequence(
    n_exercises: int = 5,
    start_difficulty: float = 0.2,
    end_difficulty: float = 0.8,
    n_notes: int = 16,
    key_root: int = 0,
) -> List[PracticeExercise]:
    """Generate a progressive series of exercises with increasing difficulty."""
    levels = np.linspace(start_difficulty, end_difficulty, n_exercises)
    foci = ["pitch", "rhythm", "mixed"]
    exercises = []
    for i, target in enumerate(levels):
        focus = foci[i % len(foci)]
        ex = generate_exercise(target, n_notes=n_notes, focus=focus,
                               key_root=key_root, seed=42 + i)
        exercises.append(ex)
    return exercises


def difficulty_feature_vector(pitches: np.ndarray, durations: np.ndarray,
                              key_root: int = 0) -> np.ndarray:
    """Numeric feature vector for ML-based difficulty prediction (7 features)."""
    prof = estimate_difficulty(pitches, durations, key_root)
    return np.array([
        prof.overall_difficulty,
        prof.rhythmic_complexity,
        prof.pitch_complexity,
        prof.key_complexity,
        prof.interval_difficulty,
        prof.accidental_density,
        prof.temporal_irregularity,
    ])
