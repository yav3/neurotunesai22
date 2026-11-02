"""Set-list optimization: fatigue-aware scheduling, constraint satisfaction, audience engagement.

Models audience attention and fatigue to optimise the ordering and selection
of tracks in a live set or curated playlist.  Constraints (max consecutive
same-genre, required opener/closer energy, tempo bounds per slot) are
enforced via greedy constraint satisfaction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SetTrack:
    """A candidate track for set-list placement."""
    track_id: int
    energy: float
    tempo: float
    genre: str = ""
    duration_s: float = 240.0
    popularity: float = 0.5


@dataclass
class SetConstraints:
    """Constraints for set-list construction."""
    n_slots: int = 12
    opener_energy_min: float = 0.5
    closer_energy_max: float = 0.6
    max_consecutive_same_genre: int = 2
    tempo_range: tuple[float, float] = (90.0, 160.0)
    max_energy_jump: float = 0.35
    target_duration_s: float = 3600.0


def audience_attention_curve(n_slots: int, peak_position: float = 0.65) -> np.ndarray:
    """Model audience attention over a set.

    Attention rises toward a peak then declines; the peak is placed at
    ``peak_position ∈ (0, 1)``.  Inspired by the Yerkes-Dodson inverted-U.
    Returns an array of shape ``(n_slots,)`` in [0, 1].
    """
    if n_slots < 1:
        return np.array([])
    t = np.linspace(0, 1, n_slots)
    raw = np.exp(-((t - peak_position) ** 2) / (2 * 0.15 ** 2))
    return raw / max(float(raw.max()), 1e-12)


def fatigue_penalty(energies: list[float], window: int = 3) -> np.ndarray:
    """Per-slot fatigue penalty from sustained high energy.

    A rolling mean over ``window`` slots; high sustained energy contributes
    cumulative fatigue.  Returns an array in [0, 1] where 1 = maximum fatigue.
    """
    n = len(energies)
    if n == 0:
        return np.array([])
    e = np.array(energies)
    fatigue = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = e[lo:i + 1]
        fatigue[i] = float(np.mean(seg)) * (len(seg) / window)
    return np.clip(fatigue, 0.0, 1.0)


def engagement_score(
    energies: list[float],
    popularities: list[float],
    attention: np.ndarray | None = None,
) -> float:
    """Weighted engagement: energy × popularity × attention, summed.

    Tracks placed at high-attention slots with matching energy and popularity
    score highest.
    """
    n = len(energies)
    if n == 0:
        return 0.0
    if attention is None:
        attention = audience_attention_curve(n)
    e = np.array(energies)
    p = np.array(popularities)
    att = attention[:n]
    raw = e * p * att
    fatigue = fatigue_penalty(energies)
    adjusted = raw * (1.0 - 0.3 * fatigue)
    return float(np.sum(adjusted) / n)


def _violates_genre_run(order: list[SetTrack], new: SetTrack, max_run: int) -> bool:
    if max_run < 1 or not new.genre:
        return False
    run = 0
    for t in reversed(order):
        if t.genre == new.genre:
            run += 1
        else:
            break
    return run >= max_run


def greedy_set_optimize(
    candidates: list[SetTrack],
    constraints: SetConstraints,
) -> list[int]:
    """Greedy constraint-satisfying set-list builder.

    Places tracks slot-by-slot, choosing the candidate that maximises the
    product of attention-alignment and energy-curve fit while respecting
    all constraints.  Returns indices into ``candidates``.
    """
    n = constraints.n_slots
    if not candidates or n < 1:
        return []

    attention = audience_attention_curve(n)
    target_energy = 0.3 + 0.5 * attention

    available = list(range(len(candidates)))
    order: list[int] = []
    placed: list[SetTrack] = []

    for slot in range(n):
        if not available:
            break
        best_idx = -1
        best_score = -1e9

        for ci in available:
            c = candidates[ci]

            if c.tempo < constraints.tempo_range[0] or c.tempo > constraints.tempo_range[1]:
                continue
            if slot == 0 and c.energy < constraints.opener_energy_min:
                continue
            if slot == n - 1 and c.energy > constraints.closer_energy_max:
                continue
            if placed and abs(c.energy - placed[-1].energy) > constraints.max_energy_jump:
                continue
            if _violates_genre_run(placed, c, constraints.max_consecutive_same_genre):
                continue

            energy_fit = 1.0 - abs(c.energy - target_energy[slot])
            pop_bonus = c.popularity * attention[slot]
            score = 0.5 * energy_fit + 0.3 * pop_bonus + 0.2 * attention[slot]

            if score > best_score:
                best_score = score
                best_idx = ci

        if best_idx < 0:
            for ci in available:
                c = candidates[ci]
                energy_fit = 1.0 - abs(c.energy - target_energy[slot])
                if energy_fit > best_score:
                    best_score = energy_fit
                    best_idx = ci

        if best_idx >= 0:
            order.append(best_idx)
            placed.append(candidates[best_idx])
            available.remove(best_idx)

    return order


def set_duration(candidates: list[SetTrack], order: list[int]) -> float:
    """Total duration (seconds) of the selected set."""
    return sum(candidates[i].duration_s for i in order)


@dataclass(frozen=True)
class SetReport:
    order: list[int]
    engagement: float
    total_duration_s: float
    fatigue_peak: float
    energy_profile: list[float]
    n_genre_violations: int


def optimize_and_report(
    candidates: list[SetTrack],
    constraints: SetConstraints,
) -> SetReport:
    """Run greedy optimization and return a full report."""
    order = greedy_set_optimize(candidates, constraints)
    energies = [candidates[i].energy for i in order]
    pops = [candidates[i].popularity for i in order]
    eng = engagement_score(energies, pops)
    dur = set_duration(candidates, order)
    fat = fatigue_penalty(energies)
    fat_peak = float(fat.max()) if len(fat) > 0 else 0.0

    violations = 0
    for i in range(2, len(order)):
        g = candidates[order[i]].genre
        if g and g == candidates[order[i - 1]].genre == candidates[order[i - 2]].genre:
            violations += 1

    return SetReport(
        order=order,
        engagement=eng,
        total_duration_s=dur,
        fatigue_peak=fat_peak,
        energy_profile=energies,
        n_genre_violations=violations,
    )
