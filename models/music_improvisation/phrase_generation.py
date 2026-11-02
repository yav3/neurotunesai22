"""Phrase generation: melodic contour, rhythmic variation, tension arcs for improvisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PhraseContour:
    pitches: list[int]
    durations: list[float]
    contour_type: str

    @property
    def range(self) -> int:
        if not self.pitches:
            return 0
        return max(self.pitches) - min(self.pitches)

    @property
    def mean_pitch(self) -> float:
        return float(np.mean(self.pitches)) if self.pitches else 0.0


def classify_contour(pitches: Sequence[int]) -> str:
    if len(pitches) < 2:
        return "static"
    diffs = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    ups = sum(1 for d in diffs if d > 0)
    downs = sum(1 for d in diffs if d < 0)
    if ups > 2 * max(downs, 1):
        return "ascending"
    if downs > 2 * max(ups, 1):
        return "descending"
    mid = len(pitches) // 2
    first_half = pitches[:mid]
    second_half = pitches[mid:]
    if np.mean(first_half) < np.mean(second_half) - 2:
        return "arch"
    if np.mean(first_half) > np.mean(second_half) + 2:
        return "inverse_arch"
    return "undulating"


def generate_tension_arc(
    n_beats: int,
    peak_position: float = 0.7,
    peak_tension: float = 0.9,
) -> np.ndarray:
    t = np.linspace(0, 1, n_beats)
    arc = np.where(
        t <= peak_position,
        peak_tension * (t / peak_position),
        peak_tension * (1.0 - (t - peak_position) / (1.0 - peak_position)),
    )
    return np.clip(arc, 0.0, 1.0)


def rhythmic_variation(
    base_durations: Sequence[float],
    variation_amount: float = 0.2,
    rng: np.random.Generator | None = None,
) -> list[float]:
    if rng is None:
        rng = np.random.default_rng()
    result = []
    for d in base_durations:
        scale = 1.0 + rng.uniform(-variation_amount, variation_amount)
        result.append(max(0.05, d * scale))
    return result


def pitch_displacement(
    reference: Sequence[int],
    target_pitch: int,
    gravity: float = 0.3,
) -> list[int]:
    result = []
    for p in reference:
        diff = target_pitch - p
        displaced = p + int(round(diff * gravity))
        result.append(displaced)
    return result


def phrase_density(pitches: Sequence[int], duration_beats: float) -> float:
    if duration_beats <= 0:
        return 0.0
    return len(pitches) / duration_beats


def intervallic_variety(pitches: Sequence[int]) -> float:
    if len(pitches) < 2:
        return 0.0
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    unique = len(set(intervals))
    return unique / len(intervals)


def build_phrase(
    scale_pitches: Sequence[int],
    n_notes: int,
    contour: str = "arch",
    rng: np.random.Generator | None = None,
) -> PhraseContour:
    if rng is None:
        rng = np.random.default_rng()
    if not scale_pitches or n_notes < 1:
        return PhraseContour([], [], "static")
    sp = sorted(scale_pitches)
    n_scale = len(sp)
    if contour == "ascending":
        indices = np.linspace(0, n_scale - 1, n_notes).astype(int)
    elif contour == "descending":
        indices = np.linspace(n_scale - 1, 0, n_notes).astype(int)
    elif contour == "arch":
        half = n_notes // 2
        up = np.linspace(0, n_scale - 1, half + 1).astype(int)
        down = np.linspace(n_scale - 1, 0, n_notes - half).astype(int)
        indices = np.concatenate([up[:-1], down])[:n_notes]
    else:
        indices = rng.integers(0, n_scale, size=n_notes)
    pitches = [int(sp[i]) for i in indices]
    durations = [0.25 + rng.random() * 0.5 for _ in range(n_notes)]
    return PhraseContour(pitches=pitches, durations=durations, contour_type=contour)
