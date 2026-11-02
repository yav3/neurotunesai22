"""Playlist dynamics: energy arcs, journey scoring, narrative strength.

Models the emotional/energy trajectory across a playlist as a sequence of
arcs, scores how well they match named curve templates (build, chill, wave,
plateau, descent), and provides greedy reordering for narrative optimization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EnergyArc:
    track_index: int
    energy_level: float
    valence: float
    arousal: float


@dataclass(frozen=True)
class PlaylistCurve:
    arcs: list[EnergyArc]
    mean_energy: float
    energy_variance: float
    energy_gradient: float


CURVE_TEMPLATES: dict[str, list[float]] = {
    "build": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    "chill": [0.5, 0.45, 0.4, 0.35, 0.3, 0.3, 0.25, 0.25, 0.2, 0.2],
    "wave": [0.3, 0.6, 0.9, 0.6, 0.3, 0.6, 0.9, 0.6, 0.3, 0.5],
    "plateau": [0.3, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.3],
    "descent": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
}


def compute_energy_arc(
    features_sequence: list[dict],
) -> list[EnergyArc]:
    """Map track features to energy arcs."""
    arcs = []
    for i, feat in enumerate(features_sequence):
        energy = float(feat.get("energy", feat.get("rms", 0.5)))
        valence = float(feat.get("valence", 0.5))
        arousal = float(feat.get("arousal", energy))
        arcs.append(EnergyArc(
            track_index=i,
            energy_level=np.clip(energy, 0.0, 1.0),
            valence=np.clip(valence, 0.0, 1.0),
            arousal=np.clip(arousal, 0.0, 1.0),
        ))
    return arcs


def energy_coherence(arcs: list[EnergyArc]) -> float:
    """Smoothness of energy transitions (1.0 = perfectly smooth, 0.0 = maximally jarring)."""
    if len(arcs) < 2:
        return 1.0
    diffs = [abs(arcs[i + 1].energy_level - arcs[i].energy_level) for i in range(len(arcs) - 1)]
    mean_diff = float(np.mean(diffs))
    return max(0.0, 1.0 - mean_diff * 2.0)


def journey_score(arcs: list[EnergyArc], target_curve: str = "build") -> float:
    """Correlation between energy arc and named curve template."""
    template = CURVE_TEMPLATES.get(target_curve)
    if template is None or len(arcs) == 0:
        return 0.0
    energies = [a.energy_level for a in arcs]
    n = len(energies)
    t_len = len(template)
    resampled = np.interp(np.linspace(0, t_len - 1, n), range(t_len), template)
    e_arr = np.array(energies)
    corr = float(np.corrcoef(e_arr, resampled)[0, 1])
    if np.isnan(corr):
        return 0.0
    return (corr + 1.0) / 2.0


def optimal_ordering(
    tracks: list[dict], target: str = "build",
) -> list[int]:
    """Greedy reordering to maximize journey_score against a target curve."""
    n = len(tracks)
    if n <= 1:
        return list(range(n))
    template = CURVE_TEMPLATES.get(target, CURVE_TEMPLATES["build"])
    t_len = len(template)
    target_energies = np.interp(np.linspace(0, t_len - 1, n), range(t_len), template)
    energies = [float(t.get("energy", t.get("rms", 0.5))) for t in tracks]
    remaining = list(range(n))
    result = []
    for pos in range(n):
        target_e = target_energies[pos]
        best_idx = min(remaining, key=lambda i: abs(energies[i] - target_e))
        result.append(best_idx)
        remaining.remove(best_idx)
    return result


def detect_energy_drops(
    arcs: list[EnergyArc], threshold: float = 0.3,
) -> list[int]:
    """Find indices where energy drops sharply (potential playlist disruptions)."""
    drops = []
    for i in range(len(arcs) - 1):
        drop = arcs[i].energy_level - arcs[i + 1].energy_level
        if drop > threshold:
            drops.append(i + 1)
    return drops


def playlist_narrative_strength(arcs: list[EnergyArc]) -> float:
    """Composite narrative quality (0-1): coherence, arc presence, resolution."""
    if len(arcs) < 3:
        return 0.0
    coherence = energy_coherence(arcs)
    energies = [a.energy_level for a in arcs]
    has_climax = max(energies) > 0.7
    has_resolution = energies[-1] < max(energies) - 0.1
    arc_bonus = 0.2 * has_climax + 0.15 * has_resolution
    variance = float(np.var(energies))
    dynamic_range = min(1.0, variance * 10.0)
    return min(1.0, 0.4 * coherence + 0.3 * dynamic_range + 0.3 * arc_bonus / 0.35)


def build_playlist_curve(arcs: list[EnergyArc]) -> PlaylistCurve:
    """Aggregate energy arcs into a PlaylistCurve summary."""
    if not arcs:
        return PlaylistCurve(arcs=[], mean_energy=0.0, energy_variance=0.0, energy_gradient=0.0)
    energies = [a.energy_level for a in arcs]
    gradient = (energies[-1] - energies[0]) / max(len(energies) - 1, 1)
    return PlaylistCurve(
        arcs=arcs,
        mean_energy=float(np.mean(energies)),
        energy_variance=float(np.var(energies)),
        energy_gradient=gradient,
    )
