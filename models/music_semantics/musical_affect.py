"""Musical affect: Russell circumplex, Thayer energy-tension, Hevner clusters, emotion arcs."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


HEVNER_CLUSTERS: dict[int, dict] = {
    1: {"adjectives": ["spiritual", "lofty", "awe-inspiring", "dignified", "sacred", "solemn"],
        "valence": 0.5, "arousal": 0.3, "label": "Dignified/Sacred"},
    2: {"adjectives": ["pathetic", "doleful", "sad", "mournful", "tragic", "melancholy", "frustrated"],
        "valence": 0.15, "arousal": 0.25, "label": "Sad/Mournful"},
    3: {"adjectives": ["dreamy", "yielding", "tender", "sentimental", "longing", "plaintive"],
        "valence": 0.35, "arousal": 0.2, "label": "Dreamy/Tender"},
    4: {"adjectives": ["lyrical", "leisurely", "satisfying", "serene", "tranquil", "quiet", "soothing"],
        "valence": 0.7, "arousal": 0.1, "label": "Serene/Tranquil"},
    5: {"adjectives": ["humorous", "playful", "whimsical", "fanciful", "quaint", "sprightly", "delicate"],
        "valence": 0.75, "arousal": 0.55, "label": "Playful/Whimsical"},
    6: {"adjectives": ["happy", "joyous", "gay", "cheerful", "bright", "merry"],
        "valence": 0.9, "arousal": 0.7, "label": "Happy/Joyous"},
    7: {"adjectives": ["exciting", "exhilarating", "dramatic", "passionate", "sensational", "agitated"],
        "valence": 0.55, "arousal": 0.9, "label": "Exciting/Dramatic"},
    8: {"adjectives": ["vigorous", "robust", "emphatic", "martial", "ponderous", "majestic", "exalting"],
        "valence": 0.6, "arousal": 0.8, "label": "Vigorous/Majestic"},
}


THAYER_QUADRANTS: dict[str, dict] = {
    "contentment": {"energy_range": (0.0, 0.5), "tension_range": (0.0, 0.5),
                    "description": "calm, relaxed, serene, peaceful"},
    "depression": {"energy_range": (0.0, 0.5), "tension_range": (0.5, 1.0),
                   "description": "sad, bored, tired, lethargic"},
    "exuberance": {"energy_range": (0.5, 1.0), "tension_range": (0.0, 0.5),
                   "description": "happy, excited, elated, alert"},
    "anxiety": {"energy_range": (0.5, 1.0), "tension_range": (0.5, 1.0),
                "description": "tense, nervous, stressed, angry"},
}


@dataclass
class AffectPoint:
    valence: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5
    timestamp: float = 0.0

    @property
    def energy(self) -> float:
        return self.arousal

    @property
    def tension(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.valence * (1.0 - self.arousal * 0.3)))

    def circumplex_angle(self) -> float:
        x = self.valence - 0.5
        y = self.arousal - 0.5
        return math.atan2(y, x)

    def circumplex_radius(self) -> float:
        x = self.valence - 0.5
        y = self.arousal - 0.5
        return math.sqrt(x * x + y * y)

    def thayer_quadrant(self) -> str:
        e = self.energy
        t = self.tension
        for name, q in THAYER_QUADRANTS.items():
            elo, ehi = q["energy_range"]
            tlo, thi = q["tension_range"]
            if elo <= e <= ehi and tlo <= t <= thi:
                return name
        return "mixed"

    def nearest_hevner_cluster(self) -> int:
        best_id = 1
        best_dist = float("inf")
        for cid, cdata in HEVNER_CLUSTERS.items():
            d = math.sqrt((self.valence - cdata["valence"]) ** 2 +
                          (self.arousal - cdata["arousal"]) ** 2)
            if d < best_dist:
                best_dist = d
                best_id = cid
        return best_id


@dataclass
class EmotionArc:
    points: list[AffectPoint] = field(default_factory=list)

    def add(self, valence: float, arousal: float, dominance: float = 0.5,
            timestamp: float = 0.0) -> None:
        self.points.append(AffectPoint(valence, arousal, dominance, timestamp))

    @property
    def length(self) -> int:
        return len(self.points)

    def valence_trajectory(self) -> list[float]:
        return [p.valence for p in self.points]

    def arousal_trajectory(self) -> list[float]:
        return [p.arousal for p in self.points]

    def mean_affect(self) -> AffectPoint:
        if not self.points:
            return AffectPoint()
        v = sum(p.valence for p in self.points) / len(self.points)
        a = sum(p.arousal for p in self.points) / len(self.points)
        d = sum(p.dominance for p in self.points) / len(self.points)
        return AffectPoint(v, a, d)

    def variance(self) -> tuple[float, float]:
        if len(self.points) < 2:
            return (0.0, 0.0)
        vals = [p.valence for p in self.points]
        aros = [p.arousal for p in self.points]
        mv = sum(vals) / len(vals)
        ma = sum(aros) / len(aros)
        vv = sum((v - mv) ** 2 for v in vals) / len(vals)
        va = sum((a - ma) ** 2 for a in aros) / len(aros)
        return (vv, va)

    def emotional_range(self) -> float:
        if len(self.points) < 2:
            return 0.0
        max_dist = 0.0
        for i, pi in enumerate(self.points):
            for pj in self.points[i + 1:]:
                d = math.sqrt((pi.valence - pj.valence) ** 2 +
                              (pi.arousal - pj.arousal) ** 2)
                if d > max_dist:
                    max_dist = d
        return max_dist

    def direction_changes(self) -> int:
        if len(self.points) < 3:
            return 0
        changes = 0
        for i in range(1, len(self.points) - 1):
            dv_prev = self.points[i].valence - self.points[i - 1].valence
            dv_next = self.points[i + 1].valence - self.points[i].valence
            if dv_prev * dv_next < 0:
                changes += 1
        return changes

    def classify_shape(self) -> str:
        if len(self.points) < 3:
            return "flat"
        vals = self.valence_trajectory()
        mid = len(vals) // 2
        first_half = sum(vals[:mid]) / mid
        second_half = sum(vals[mid:]) / (len(vals) - mid)
        peak_idx = vals.index(max(vals))
        trough_idx = vals.index(min(vals))
        if trough_idx < peak_idx and trough_idx < mid and peak_idx > mid:
            return "rags_to_riches"
        if peak_idx < trough_idx and peak_idx < mid and trough_idx > mid:
            return "riches_to_rags"
        if first_half < second_half and (second_half - first_half) > 0.1:
            return "ascending"
        if first_half > second_half and (first_half - second_half) > 0.1:
            return "descending"
        vv, _ = self.variance()
        if vv < 0.01:
            return "flat"
        return "oscillating"


class CircumplexModel:
    def __init__(self) -> None:
        self._cluster_centers: dict[str, tuple[float, float]] = {
            "happy": (0.8, 0.7),
            "excited": (0.6, 0.9),
            "tense": (0.3, 0.85),
            "angry": (0.15, 0.8),
            "sad": (0.15, 0.25),
            "depressed": (0.2, 0.15),
            "calm": (0.6, 0.15),
            "relaxed": (0.75, 0.25),
            "content": (0.8, 0.4),
            "bored": (0.35, 0.1),
        }

    def classify(self, valence: float, arousal: float, top_k: int = 3) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        for label, (cv, ca) in self._cluster_centers.items():
            d = math.sqrt((valence - cv) ** 2 + (arousal - ca) ** 2)
            sim = max(0.0, 1.0 - d / math.sqrt(2))
            scores.append((label, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def interpolate(self, label_a: str, label_b: str, t: float) -> tuple[float, float]:
        if label_a not in self._cluster_centers or label_b not in self._cluster_centers:
            return (0.5, 0.5)
        va, aa = self._cluster_centers[label_a]
        vb, ab = self._cluster_centers[label_b]
        return (va + t * (vb - va), aa + t * (ab - aa))

    def angular_distance(self, label_a: str, label_b: str) -> float:
        if label_a not in self._cluster_centers or label_b not in self._cluster_centers:
            return math.pi
        va, aa = self._cluster_centers[label_a]
        vb, ab = self._cluster_centers[label_b]
        angle_a = math.atan2(aa - 0.5, va - 0.5)
        angle_b = math.atan2(ab - 0.5, vb - 0.5)
        diff = abs(angle_a - angle_b)
        return min(diff, 2 * math.pi - diff)


class MusicalAffectAnalyzer:
    def __init__(self) -> None:
        self._circumplex = CircumplexModel()

    def features_to_affect(self, tempo: float, mode: float, loudness: float,
                           spectral_centroid: float, key_strength: float) -> AffectPoint:
        arousal = _clamp((tempo - 60) / 120)
        valence = _clamp(0.3 * mode + 0.2 * key_strength + 0.2 * (1.0 - (loudness + 60) / 60) + 0.3 * 0.5)
        dominance = _clamp(0.5 * (loudness + 60) / 60 + 0.3 * arousal + 0.2 * 0.5)
        return AffectPoint(valence, arousal, dominance)

    def analyze_trajectory(self, feature_sequence: Sequence[dict[str, float]]) -> EmotionArc:
        arc = EmotionArc()
        for i, feats in enumerate(feature_sequence):
            ap = self.features_to_affect(
                feats.get("tempo", 120),
                feats.get("mode", 0.5),
                feats.get("loudness", -20),
                feats.get("spectral_centroid", 2000),
                feats.get("key_strength", 0.5),
            )
            ap.timestamp = float(i)
            arc.add(ap.valence, ap.arousal, ap.dominance, ap.timestamp)
        return arc

    def circumplex_classify(self, valence: float, arousal: float,
                             top_k: int = 3) -> list[tuple[str, float]]:
        return self._circumplex.classify(valence, arousal, top_k)

    def hevner_cluster(self, valence: float, arousal: float) -> dict:
        pt = AffectPoint(valence, arousal)
        cid = pt.nearest_hevner_cluster()
        return {"cluster_id": cid, **HEVNER_CLUSTERS[cid]}

    def transition_cost(self, from_v: float, from_a: float,
                        to_v: float, to_a: float) -> float:
        return math.sqrt((to_v - from_v) ** 2 + (to_a - from_a) ** 2)

    def smooth_trajectory(self, arc: EmotionArc, window: int = 3) -> EmotionArc:
        if arc.length < window:
            return arc
        smoothed = EmotionArc()
        vals = arc.valence_trajectory()
        aros = arc.arousal_trajectory()
        half = window // 2
        for i in range(len(vals)):
            lo = max(0, i - half)
            hi = min(len(vals), i + half + 1)
            sv = sum(vals[lo:hi]) / (hi - lo)
            sa = sum(aros[lo:hi]) / (hi - lo)
            smoothed.add(sv, sa, arc.points[i].dominance, arc.points[i].timestamp)
        return smoothed


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def affect_distance(a: AffectPoint, b: AffectPoint) -> float:
    return math.sqrt((a.valence - b.valence) ** 2 +
                     (a.arousal - b.arousal) ** 2 +
                     (a.dominance - b.dominance) ** 2)


def arc_similarity(arc_a: EmotionArc, arc_b: EmotionArc) -> float:
    if arc_a.length == 0 or arc_b.length == 0:
        return 0.0
    n = min(arc_a.length, arc_b.length)
    total_dist = 0.0
    for i in range(n):
        total_dist += affect_distance(arc_a.points[i], arc_b.points[i])
    mean_dist = total_dist / n
    return max(0.0, 1.0 - mean_dist / math.sqrt(3))
