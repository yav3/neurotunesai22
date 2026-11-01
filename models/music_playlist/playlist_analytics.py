"""Playlist-level analytics: coherence, fatigue, surprise, flow metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaylistStats:
    n_tracks: int
    duration_sec: float
    mean_tempo: float
    tempo_std: float
    mean_energy: float
    energy_std: float
    key_entropy: float
    tempo_range: tuple[float, float]


def playlist_stats(
    tempos: np.ndarray,
    energies: np.ndarray,
    durations: np.ndarray,
    keys: np.ndarray | None = None,
) -> PlaylistStats:
    """Compute summary statistics for a playlist."""
    n = len(tempos)
    if n == 0:
        return PlaylistStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0))
    key_ent = 0.0
    if keys is not None and len(keys) > 0:
        key_ent = _entropy(keys.astype(int), 12)
    return PlaylistStats(
        n_tracks=n,
        duration_sec=float(np.sum(durations)),
        mean_tempo=float(np.mean(tempos)),
        tempo_std=float(np.std(tempos)),
        mean_energy=float(np.mean(energies)),
        energy_std=float(np.std(energies)),
        key_entropy=key_ent,
        tempo_range=(float(np.min(tempos)), float(np.max(tempos))),
    )


def _entropy(labels: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(labels % n_classes, minlength=n_classes)
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def coherence_score(features: np.ndarray) -> float:
    """Mean pairwise cosine similarity across track feature vectors.

    High coherence → stylistically uniform playlist.
    """
    n = features.shape[0]
    if n < 2:
        return 1.0
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = features / norms
    sim_matrix = normed @ normed.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim_matrix[mask].mean())


def surprise_curve(features: np.ndarray) -> np.ndarray:
    """Per-track surprise: 1 − cosine_sim(track_i, track_{i-1}).

    First track has surprise = 0.
    """
    n = features.shape[0]
    if n < 2:
        return np.zeros(max(n, 0))
    result = np.zeros(n)
    for i in range(1, n):
        a, b = features[i - 1], features[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            result[i] = 1.0
        else:
            result[i] = 1.0 - float(np.dot(a, b) / (na * nb))
    return result


def listening_fatigue(energies: np.ndarray, threshold: float = 0.8) -> float:
    """Fraction of consecutive high-energy pairs.

    Sustained high-energy sections cause listener fatigue.
    """
    if len(energies) < 2:
        return 0.0
    high = energies > threshold
    pairs = high[:-1] & high[1:]
    return float(pairs.sum() / max(len(pairs), 1))


def tempo_smoothness(tempos: np.ndarray) -> float:
    """Mean absolute tempo change between consecutive tracks, normalized by mean tempo."""
    if len(tempos) < 2:
        return 1.0
    diffs = np.abs(np.diff(tempos))
    mean_t = max(np.mean(tempos), 1.0)
    return float(1.0 - np.mean(diffs) / mean_t)


def energy_arc_fit(energies: np.ndarray, target_shape: str = "hill") -> float:
    """R² of energies against a target arc shape.

    target_shape: 'hill' (rise-peak-fall), 'ramp' (steady rise),
                  'valley' (fall-low-rise), 'flat' (constant).
    """
    n = len(energies)
    if n < 2:
        return 0.0
    t = np.linspace(0, 1, n)
    if target_shape == "hill":
        target = np.sin(np.pi * t)
    elif target_shape == "ramp":
        target = t
    elif target_shape == "valley":
        target = 1.0 - np.sin(np.pi * t)
    else:
        target = np.ones(n) * 0.5
    ss_res = np.sum((energies - target) ** 2)
    ss_tot = np.sum((energies - np.mean(energies)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(max(0.0, 1.0 - ss_res / ss_tot))


@dataclass(frozen=True)
class FlowScore:
    tempo_smoothness: float
    energy_arc_fit: float
    coherence: float
    fatigue: float
    surprise_mean: float
    overall: float


def compute_flow(
    features: np.ndarray,
    tempos: np.ndarray,
    energies: np.ndarray,
    target_arc: str = "hill",
) -> FlowScore:
    """Composite flow score combining multiple playlist quality dimensions."""
    ts = tempo_smoothness(tempos)
    eaf = energy_arc_fit(energies, target_arc)
    coh = coherence_score(features)
    fat = listening_fatigue(energies)
    surp = float(np.mean(surprise_curve(features)))
    overall = 0.3 * ts + 0.25 * eaf + 0.2 * coh + 0.15 * (1.0 - fat) + 0.1 * (1.0 - surp)
    return FlowScore(
        tempo_smoothness=ts,
        energy_arc_fit=eaf,
        coherence=coh,
        fatigue=fat,
        surprise_mean=surp,
        overall=float(overall),
    )
