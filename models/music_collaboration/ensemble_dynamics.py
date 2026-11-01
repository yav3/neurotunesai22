"""Ensemble dynamics: synchrony, entrainment, blend, and turn-taking analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class EnsembleSynchrony:
    onset_synchrony: float
    tempo_agreement: float
    pitch_consonance: float
    n_parts: int


def _nearest_offset(ref: np.ndarray, query: float) -> float:
    if len(ref) == 0:
        return np.inf
    return float(np.min(np.abs(ref - query)))


def compute_onset_synchrony(
    onsets_per_part: List[np.ndarray],
    tolerance_sec: float = 0.05,
) -> float:
    if len(onsets_per_part) < 2:
        return 1.0
    all_onsets = np.concatenate(onsets_per_part)
    if len(all_onsets) == 0:
        return 1.0
    deviations: list[float] = []
    for i, ons_i in enumerate(onsets_per_part):
        for t in ons_i:
            for j, ons_j in enumerate(onsets_per_part):
                if i == j:
                    continue
                d = _nearest_offset(ons_j, t)
                if d <= tolerance_sec:
                    deviations.append(d)
    if not deviations:
        return 0.0
    return 1.0 - min(float(np.mean(deviations)) / tolerance_sec, 1.0)


def compute_tempo_agreement(tempi_per_part: List[np.ndarray]) -> float:
    if len(tempi_per_part) < 2:
        return 1.0
    means = [float(np.mean(t)) for t in tempi_per_part if len(t) > 0]
    if len(means) < 2:
        return 1.0
    cv = float(np.std(means) / max(np.mean(means), 1e-12))
    return max(1.0 - cv, 0.0)


def entrainment_index(leader_ioi: np.ndarray, follower_ioi: np.ndarray) -> float:
    n = min(len(leader_ioi), len(follower_ioi))
    if n < 2:
        return 0.0
    a = leader_ioi[:n] - np.mean(leader_ioi[:n])
    b = follower_ioi[:n] - np.mean(follower_ioi[:n])
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.max(np.correlate(a, b, mode="full")) / denom)


def turn_taking_analysis(
    onset_times: List[np.ndarray],
    part_labels: List[str],
) -> dict:
    if not onset_times or not part_labels:
        return {"overlap_ratio": 0.0, "solo_ratios": {}}
    n_parts = len(onset_times)
    duration = max(float(np.max(o)) for o in onset_times if len(o) > 0) if any(len(o) > 0 for o in onset_times) else 0.0
    if duration <= 0:
        return {"overlap_ratio": 0.0, "solo_ratios": {k: 0.0 for k in part_labels}}

    resolution = 0.01
    n_bins = int(duration / resolution) + 1
    activity = np.zeros((n_parts, n_bins), dtype=bool)
    note_dur = 0.1
    for i, ons in enumerate(onset_times):
        for t in ons:
            start = int(t / resolution)
            end = min(start + int(note_dur / resolution), n_bins)
            activity[i, start:end] = True

    active_count = np.sum(activity, axis=0)
    overlap_bins = np.sum(active_count > 1)
    any_active = np.sum(active_count > 0)
    overlap_ratio = float(overlap_bins / max(any_active, 1))

    solo_ratios = {}
    for i, label in enumerate(part_labels):
        solo_mask = activity[i] & (active_count == 1)
        solo_ratios[label] = float(np.sum(solo_mask) / max(any_active, 1))

    return {"overlap_ratio": overlap_ratio, "solo_ratios": solo_ratios}


def blend_score(spectra_per_part: List[np.ndarray]) -> float:
    if len(spectra_per_part) < 2:
        return 1.0
    normed = []
    for s in spectra_per_part:
        total = np.sum(np.abs(s))
        if total < 1e-15:
            normed.append(np.zeros_like(s))
        else:
            normed.append(np.abs(s) / total)
    n = len(normed)
    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap = float(np.sum(np.minimum(normed[i], normed[j])))
            overlaps.append(overlap)
    return float(np.mean(overlaps)) if overlaps else 0.0


def analyze_ensemble(
    onsets_per_part: List[np.ndarray],
    tempi_per_part: List[np.ndarray],
    pitch_sets: List[np.ndarray] | None = None,
) -> EnsembleSynchrony:
    sync = compute_onset_synchrony(onsets_per_part)
    tempo = compute_tempo_agreement(tempi_per_part)
    consonance = 0.5
    if pitch_sets is not None and len(pitch_sets) > 0:
        consonance_vals = []
        for ps in pitch_sets:
            if len(ps) < 2:
                continue
            intervals = np.diff(np.sort(ps)) % 12
            cons = np.mean([1.0 if iv in (0, 3, 4, 5, 7, 8, 9) else 0.3 for iv in intervals])
            consonance_vals.append(float(cons))
        if consonance_vals:
            consonance = float(np.mean(consonance_vals))
    return EnsembleSynchrony(
        onset_synchrony=sync,
        tempo_agreement=tempo,
        pitch_consonance=consonance,
        n_parts=len(onsets_per_part),
    )
