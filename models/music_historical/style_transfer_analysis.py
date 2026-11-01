"""Computational analysis of historical style characteristics.

Statistical style fingerprinting via interval distributions, rhythm
histograms, and harmonic vocabulary.  Supports era classification
(nearest-centroid), style distance (KL + EMD), and interpolation
between style distributions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class StyleFingerprint:
    interval_distribution: np.ndarray
    rhythm_histogram: np.ndarray
    harmony_vocab: np.ndarray
    pitch_range: tuple[int, int] = (0, 127)
    voice_count_dist: np.ndarray = field(default_factory=lambda: np.zeros(8))
    ornamentation_density: float = 0.0
    era_label: str = ""


@dataclass
class StyleDistance:
    kl_divergence: float = 0.0
    earth_mover_distance: float = 0.0
    feature_distances: dict[str, float] = field(default_factory=dict)


@dataclass
class EraCharacteristics:
    typical_tempo_range: tuple[float, float] = (60.0, 120.0)
    common_forms: list[str] = field(default_factory=list)
    harmonic_complexity: float = 0.5
    rhythmic_complexity: float = 0.5


ERA_CENTROIDS: dict[str, StyleFingerprint] = {
    "baroque": StyleFingerprint(
        interval_distribution=np.array([0.15, 0.25, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]),
        rhythm_histogram=np.array([0.30, 0.25, 0.20, 0.10, 0.08, 0.04, 0.02, 0.01]),
        harmony_vocab=np.array([0.25, 0.15, 0.10, 0.10, 0.15, 0.10, 0.05, 0.05, 0.03, 0.01, 0.005, 0.005]),
        pitch_range=(36, 84), ornamentation_density=0.15, era_label="baroque",
    ),
    "classical": StyleFingerprint(
        interval_distribution=np.array([0.20, 0.25, 0.18, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01]),
        rhythm_histogram=np.array([0.25, 0.30, 0.20, 0.10, 0.05, 0.05, 0.03, 0.02]),
        harmony_vocab=np.array([0.30, 0.15, 0.10, 0.10, 0.15, 0.08, 0.04, 0.03, 0.02, 0.015, 0.005, 0.01]),
        pitch_range=(40, 88), ornamentation_density=0.08, era_label="classical",
    ),
    "romantic": StyleFingerprint(
        interval_distribution=np.array([0.12, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01]),
        rhythm_histogram=np.array([0.15, 0.20, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05]),
        harmony_vocab=np.array([0.20, 0.12, 0.10, 0.10, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02, 0.02]),
        pitch_range=(28, 96), ornamentation_density=0.05, era_label="romantic",
    ),
    "impressionist": StyleFingerprint(
        interval_distribution=np.array([0.10, 0.12, 0.12, 0.10, 0.10, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.03, 0.02]),
        rhythm_histogram=np.array([0.12, 0.15, 0.15, 0.15, 0.12, 0.12, 0.10, 0.09]),
        harmony_vocab=np.array([0.15, 0.10, 0.08, 0.08, 0.10, 0.10, 0.10, 0.08, 0.07, 0.06, 0.04, 0.04]),
        pitch_range=(32, 96), ornamentation_density=0.03, era_label="impressionist",
    ),
    "twentieth_century": StyleFingerprint(
        interval_distribution=np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04]),
        rhythm_histogram=np.array([0.10, 0.10, 0.12, 0.12, 0.12, 0.12, 0.15, 0.17]),
        harmony_vocab=np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.12]),
        pitch_range=(21, 108), ornamentation_density=0.02, era_label="twentieth_century",
    ),
}


def _normalise(arr: np.ndarray) -> np.ndarray:
    s = arr.sum()
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_n = _normalise(p) + 1e-12
    q_n = _normalise(q) + 1e-12
    n = min(len(p_n), len(q_n))
    return float(np.sum(p_n[:n] * np.log(p_n[:n] / q_n[:n])))


def _emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    p_n = _normalise(p)
    q_n = _normalise(q)
    n = min(len(p_n), len(q_n))
    cdf_diff = np.cumsum(p_n[:n]) - np.cumsum(q_n[:n])
    return float(np.sum(np.abs(cdf_diff)))


def compute_style_fingerprint(
    pitches: Sequence[int],
    onsets: Sequence[float],
    durations: Sequence[float] | None = None,
    chord_roots: Sequence[int] | None = None,
) -> StyleFingerprint:
    p = np.asarray(pitches)
    if len(p) < 2:
        return StyleFingerprint(np.zeros(13), np.zeros(8), np.zeros(12))
    intervals = np.abs(np.diff(p))
    interval_hist = np.zeros(13)
    for iv in intervals:
        idx = min(int(iv), 12)
        interval_hist[idx] += 1
    interval_hist = _normalise(interval_hist)
    o = np.asarray(onsets)
    ioi = np.diff(o)
    ioi = ioi[ioi > 0]
    if len(ioi) > 0:
        log_ioi = np.log2(ioi + 1e-9)
        bins = np.linspace(log_ioi.min(), log_ioi.max() + 1e-9, 9)
        rhythm_hist, _ = np.histogram(log_ioi, bins=bins)
        rhythm_hist = _normalise(rhythm_hist.astype(float))
    else:
        rhythm_hist = np.zeros(8)
    harmony = np.zeros(12)
    if chord_roots is not None:
        for r in chord_roots:
            harmony[int(r) % 12] += 1
    else:
        for pit in p:
            harmony[int(pit) % 12] += 1
    harmony = _normalise(harmony)
    orn = 0.0
    if durations is not None:
        d = np.asarray(durations)
        short_threshold = np.median(d) * 0.25 if len(d) > 0 else 0.1
        orn = float(np.mean(d < short_threshold)) if len(d) > 0 else 0.0
    return StyleFingerprint(
        interval_distribution=interval_hist,
        rhythm_histogram=rhythm_hist,
        harmony_vocab=harmony,
        pitch_range=(int(p.min()), int(p.max())),
        ornamentation_density=orn,
    )


def style_distance(a: StyleFingerprint, b: StyleFingerprint) -> StyleDistance:
    kl_iv = _kl_divergence(a.interval_distribution, b.interval_distribution)
    kl_rh = _kl_divergence(a.rhythm_histogram, b.rhythm_histogram)
    kl_hv = _kl_divergence(a.harmony_vocab, b.harmony_vocab)
    emd_iv = _emd_1d(a.interval_distribution, b.interval_distribution)
    emd_rh = _emd_1d(a.rhythm_histogram, b.rhythm_histogram)
    emd_hv = _emd_1d(a.harmony_vocab, b.harmony_vocab)
    return StyleDistance(
        kl_divergence=(kl_iv + kl_rh + kl_hv) / 3.0,
        earth_mover_distance=(emd_iv + emd_rh + emd_hv) / 3.0,
        feature_distances={
            "interval_kl": kl_iv, "rhythm_kl": kl_rh, "harmony_kl": kl_hv,
            "interval_emd": emd_iv, "rhythm_emd": emd_rh, "harmony_emd": emd_hv,
            "ornamentation_diff": abs(a.ornamentation_density - b.ornamentation_density),
        },
    )


def classify_era(fp: StyleFingerprint, centroids: dict[str, StyleFingerprint] | None = None) -> str:
    cents = centroids or ERA_CENTROIDS
    best_era = ""
    best_dist = float("inf")
    for era, cfp in cents.items():
        d = style_distance(fp, cfp)
        total = d.kl_divergence + d.earth_mover_distance
        if total < best_dist:
            best_dist = total
            best_era = era
    return best_era


def interpolate_styles(a: StyleFingerprint, b: StyleFingerprint, alpha: float = 0.5) -> StyleFingerprint:
    t = np.clip(alpha, 0.0, 1.0)
    n_iv = min(len(a.interval_distribution), len(b.interval_distribution))
    n_rh = min(len(a.rhythm_histogram), len(b.rhythm_histogram))
    n_hv = min(len(a.harmony_vocab), len(b.harmony_vocab))
    return StyleFingerprint(
        interval_distribution=_normalise((1 - t) * a.interval_distribution[:n_iv] + t * b.interval_distribution[:n_iv]),
        rhythm_histogram=_normalise((1 - t) * a.rhythm_histogram[:n_rh] + t * b.rhythm_histogram[:n_rh]),
        harmony_vocab=_normalise((1 - t) * a.harmony_vocab[:n_hv] + t * b.harmony_vocab[:n_hv]),
        pitch_range=(
            int((1 - t) * a.pitch_range[0] + t * b.pitch_range[0]),
            int((1 - t) * a.pitch_range[1] + t * b.pitch_range[1]),
        ),
        ornamentation_density=(1 - t) * a.ornamentation_density + t * b.ornamentation_density,
        era_label=f"{a.era_label}_{b.era_label}_{t:.2f}",
    )


def ornamentation_density(durations: Sequence[float], threshold_ratio: float = 0.25) -> float:
    d = np.asarray(durations, dtype=np.float64)
    if len(d) == 0:
        return 0.0
    med = np.median(d)
    return float(np.mean(d < med * threshold_ratio))


def harmonic_vocabulary_size(chord_roots: Sequence[int], durations: Sequence[float]) -> float:
    if not chord_roots or not durations:
        return 0.0
    total_time = sum(durations)
    unique = len(set(int(r) % 12 for r in chord_roots))
    return float(unique / max(total_time, 1e-9))


def rhythmic_complexity_score(onsets: Sequence[float]) -> float:
    o = np.asarray(onsets, dtype=np.float64)
    if len(o) < 2:
        return 0.0
    ioi = np.diff(o)
    ioi = ioi[ioi > 0]
    if len(ioi) == 0:
        return 0.0
    hist, _ = np.histogram(ioi, bins=max(5, len(ioi) // 3 + 1))
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log2(p)))
    max_entropy = np.log2(len(p)) if len(p) > 1 else 1.0
    return float(entropy / max(max_entropy, 1e-9))


def melodic_range_profile(pitches: Sequence[int], phrase_len: int = 16) -> dict[str, float]:
    p = np.asarray(pitches)
    if len(p) == 0:
        return {"mean_range": 0.0, "std_range": 0.0, "global_range": 0.0}
    ranges = []
    for i in range(0, len(p), phrase_len):
        phrase = p[i : i + phrase_len]
        if len(phrase) > 1:
            ranges.append(float(phrase.max() - phrase.min()))
    return {
        "mean_range": float(np.mean(ranges)) if ranges else 0.0,
        "std_range": float(np.std(ranges)) if ranges else 0.0,
        "global_range": float(p.max() - p.min()),
    }
