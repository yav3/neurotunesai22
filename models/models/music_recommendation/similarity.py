"""Track similarity metrics: class-based (TrackRecord) and function-based (dict) APIs."""
from __future__ import annotations

from typing import Any

import numpy as np

from models.music_data.schema import TrackRecord
from models.music_playlist.transition import CAMELOT_NUMBERS


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---- Function-based API (operates on dicts / raw values) ----

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return _cosine(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def weighted_feature_similarity(
    features_a: dict[str, float],
    features_b: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    common = set(features_a) & set(features_b)
    if not common:
        return 0.0
    if weights is None:
        weights = {k: 1.0 for k in common}
    total_w = sum(weights.get(k, 0.0) for k in common)
    if total_w < 1e-12:
        return 0.0
    score = 0.0
    for k in common:
        w = weights.get(k, 0.0)
        diff = abs(features_a[k] - features_b[k])
        score += w * max(0.0, 1.0 - diff)
    return score / total_w


def mood_similarity(vad_a: tuple[float, float, float], vad_b: tuple[float, float, float]) -> float:
    dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(vad_a, vad_b)))
    return float(max(0.0, 1.0 - dist / np.sqrt(3.0)))


def _camelot_lookup(key_name: str) -> tuple[int, str] | None:
    if key_name in CAMELOT_NUMBERS:
        return CAMELOT_NUMBERS[key_name]
    for k, v in CAMELOT_NUMBERS.items():
        if k.lower() == key_name.lower():
            return v
    return None


def harmonic_compatibility(key_a: str, key_b: str) -> float:
    ca = _camelot_lookup(key_a)
    cb = _camelot_lookup(key_b)
    if ca is None or cb is None:
        return 0.0
    num_a, letter_a = ca
    num_b, letter_b = cb
    if num_a == num_b and letter_a == letter_b:
        return 1.0
    if num_a == num_b and letter_a != letter_b:
        return 0.8
    diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    if diff == 1 and letter_a == letter_b:
        return 0.9
    if diff <= 2:
        return max(0.0, 0.7 - (diff - 1) * 0.3)
    return max(0.0, 0.3 - diff * 0.05)


def rhythm_similarity(
    bpm_a: float, bpm_b: float,
    ts_a: str = "4/4", ts_b: str = "4/4",
) -> float:
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.0
    ratio = min(bpm_a, bpm_b) / max(bpm_a, bpm_b)
    double = min(bpm_a, bpm_b) * 2 / max(bpm_a, bpm_b)
    if abs(double - 1.0) < abs(ratio - 1.0):
        ratio = double * 0.85
    ts_match = 1.0 if ts_a == ts_b else 0.7
    return float(min(1.0, ratio) * ts_match)


# ---- Class-based API (operates on TrackRecord) ----

class FeatureSimilarity:
    def score(self, a: TrackRecord, b: TrackRecord) -> float:
        fa = np.array(list(a.to_feature_dict().values()), dtype=float)
        fb = np.array(list(b.to_feature_dict().values()), dtype=float)
        return max(0.0, _cosine(fa, fb))


class HarmonicSimilarity:
    _CAMELOT = CAMELOT_NUMBERS

    def _key_distance(self, key_a: str, key_b: str) -> int:
        ca = _camelot_lookup(key_a)
        cb = _camelot_lookup(key_b)
        if ca is None or cb is None:
            return 7
        if ca == cb:
            return 0
        num_a, let_a = ca
        num_b, let_b = cb
        if let_a == let_b:
            diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
        else:
            diff = 0 if num_a == num_b else min(abs(num_a - num_b), 12 - abs(num_a - num_b)) + 1
        return diff

    def score(self, a: TrackRecord, b: TrackRecord) -> float:
        key_a = f"{a.key_str} {a.mode}" if a.mode else a.key_str
        key_b = f"{b.key_str} {b.mode}" if b.mode else b.key_str
        dist = self._key_distance(key_a, key_b)
        return max(0.0, 1.0 - dist / 7.0)


class RhythmicSimilarity:
    def score(self, a: TrackRecord, b: TrackRecord) -> float:
        if a.bpm <= 0 or b.bpm <= 0:
            return 0.5
        ratio = min(a.bpm, b.bpm) / max(a.bpm, b.bpm)
        double = min(a.bpm, b.bpm) * 2
        if abs(double - max(a.bpm, b.bpm)) < 5:
            ratio = max(ratio, 0.85)
        ts_match = 1.0 if a.time_signature == b.time_signature else 0.7
        return float(ratio * ts_match)


class EmotionalSimilarity:
    def score(self, a: TrackRecord, b: TrackRecord) -> float:
        dist = np.sqrt(
            (a.valence - b.valence) ** 2
            + (a.arousal - b.arousal) ** 2
            + (a.dominance - b.dominance) ** 2
        )
        return float(max(0.0, 1.0 - dist / np.sqrt(3)))


class CompositeSimilarity:
    def __init__(self, w_feature=0.3, w_harmonic=0.25, w_rhythmic=0.2, w_emotional=0.25):
        self._w = (w_feature, w_harmonic, w_rhythmic, w_emotional)
        self._feat = FeatureSimilarity()
        self._harm = HarmonicSimilarity()
        self._rhythm = RhythmicSimilarity()
        self._emot = EmotionalSimilarity()

    def score(self, a: TrackRecord, b: TrackRecord) -> float:
        scores = [
            self._feat.score(a, b), self._harm.score(a, b),
            self._rhythm.score(a, b), self._emot.score(a, b),
        ]
        return float(sum(w * s for w, s in zip(self._w, scores)))

    def score_detailed(self, a: TrackRecord, b: TrackRecord) -> dict[str, float]:
        return {
            "feature": self._feat.score(a, b),
            "harmonic": self._harm.score(a, b),
            "rhythmic": self._rhythm.score(a, b),
            "emotional": self._emot.score(a, b),
            "composite": self.score(a, b),
        }
