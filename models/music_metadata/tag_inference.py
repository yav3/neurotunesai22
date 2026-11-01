"""Infer musical tags from audio features using rule-based classifiers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TagPrediction:
    tag: str
    confidence: float


# ---------- Genre inference ----------

_GENRE_CENTROIDS: dict[str, dict[str, float]] = {
    "classical": {"centroid": 1200, "spread": 800, "tempo": 90, "zcr": 0.02},
    "electronic": {"centroid": 3500, "spread": 2500, "tempo": 128, "zcr": 0.08},
    "rock": {"centroid": 2500, "spread": 1800, "tempo": 120, "zcr": 0.10},
    "jazz": {"centroid": 2000, "spread": 1500, "tempo": 130, "zcr": 0.05},
    "hip-hop": {"centroid": 1800, "spread": 1200, "tempo": 90, "zcr": 0.06},
    "pop": {"centroid": 2200, "spread": 1600, "tempo": 115, "zcr": 0.07},
    "ambient": {"centroid": 1000, "spread": 600, "tempo": 70, "zcr": 0.02},
    "metal": {"centroid": 3200, "spread": 2200, "tempo": 140, "zcr": 0.14},
    "folk": {"centroid": 1600, "spread": 1000, "tempo": 105, "zcr": 0.04},
    "r&b": {"centroid": 1700, "spread": 1100, "tempo": 85, "zcr": 0.05},
}


def infer_genre_tags(
    spectral_features: dict[str, float],
    rhythm_features: dict[str, float],
) -> list[TagPrediction]:
    """Infer genre tags from spectral and rhythm features.

    *spectral_features* should include ``centroid``, ``spread``.
    *rhythm_features* should include ``tempo``.
    """
    centroid = spectral_features.get("centroid", 2000.0)
    spread = spectral_features.get("spread", 1500.0)
    tempo = rhythm_features.get("tempo", 120.0)

    results: list[TagPrediction] = []
    for genre, proto in _GENRE_CENTROIDS.items():
        d_c = abs(centroid - proto["centroid"]) / 4000.0
        d_s = abs(spread - proto["spread"]) / 3000.0
        d_t = abs(tempo - proto["tempo"]) / 200.0
        dist = np.sqrt(d_c ** 2 + d_s ** 2 + d_t ** 2)
        conf = float(max(0.0, 1.0 - dist))
        if conf > 0.1:
            results.append(TagPrediction(tag=genre, confidence=round(conf, 4)))
    results.sort(key=lambda p: -p.confidence)
    return results[:5]


# ---------- Mood inference ----------

_MOOD_RULES: dict[str, dict[str, tuple[float, float]]] = {
    "happy": {"valence": (0.6, 1.0), "arousal": (0.4, 1.0), "energy": (0.5, 1.0)},
    "sad": {"valence": (0.0, 0.35), "arousal": (0.0, 0.45), "energy": (0.0, 0.5)},
    "angry": {"valence": (0.0, 0.4), "arousal": (0.6, 1.0), "energy": (0.7, 1.0)},
    "relaxed": {"valence": (0.4, 0.8), "arousal": (0.0, 0.4), "energy": (0.0, 0.4)},
    "energetic": {"valence": (0.5, 1.0), "arousal": (0.7, 1.0), "energy": (0.7, 1.0)},
    "melancholic": {"valence": (0.1, 0.45), "arousal": (0.2, 0.55), "energy": (0.1, 0.5)},
    "tense": {"valence": (0.0, 0.5), "arousal": (0.5, 1.0), "energy": (0.5, 1.0)},
    "peaceful": {"valence": (0.5, 1.0), "arousal": (0.0, 0.3), "energy": (0.0, 0.35)},
}


def infer_mood_tags(
    valence: float,
    arousal: float,
    energy: float,
) -> list[TagPrediction]:
    """Infer mood tags from valence-arousal-energy coordinates."""
    results: list[TagPrediction] = []
    for mood, ranges in _MOOD_RULES.items():
        score = 1.0
        for dim_name, (lo, hi) in ranges.items():
            val = {"valence": valence, "arousal": arousal, "energy": energy}[dim_name]
            if lo <= val <= hi:
                centre = (lo + hi) / 2.0
                half = (hi - lo) / 2.0
                score *= 1.0 - abs(val - centre) / (half + 1e-9) * 0.3
            else:
                dist = min(abs(val - lo), abs(val - hi))
                score *= max(0.0, 1.0 - dist * 2.0)
        if score > 0.1:
            results.append(TagPrediction(tag=mood, confidence=round(float(score), 4)))
    results.sort(key=lambda p: -p.confidence)
    return results[:4]


# ---------- Instrument inference ----------

_INSTRUMENT_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "piano": {"centroid": (800, 2500), "zcr": (0.01, 0.06)},
    "guitar": {"centroid": (1500, 3500), "zcr": (0.03, 0.10)},
    "drums": {"centroid": (2000, 6000), "zcr": (0.08, 0.20)},
    "bass": {"centroid": (200, 1200), "zcr": (0.01, 0.04)},
    "violin": {"centroid": (1000, 3000), "zcr": (0.02, 0.08)},
    "flute": {"centroid": (1500, 4000), "zcr": (0.01, 0.04)},
    "synth": {"centroid": (1000, 5000), "zcr": (0.03, 0.12)},
    "voice": {"centroid": (800, 3500), "zcr": (0.02, 0.09)},
}


def infer_instrument_tags(
    spectral_centroid: float,
    mfcc_means: np.ndarray,
    zcr: float,
) -> list[TagPrediction]:
    """Infer likely instruments from spectral centroid, MFCCs, and ZCR."""
    results: list[TagPrediction] = []
    mfcc_energy = float(np.sum(mfcc_means[:4] ** 2)) if len(mfcc_means) >= 4 else 1.0
    for inst, profile in _INSTRUMENT_PROFILES.items():
        c_lo, c_hi = profile["centroid"]
        z_lo, z_hi = profile["zcr"]
        c_score = 1.0
        if spectral_centroid < c_lo:
            c_score = max(0.0, 1.0 - (c_lo - spectral_centroid) / 2000.0)
        elif spectral_centroid > c_hi:
            c_score = max(0.0, 1.0 - (spectral_centroid - c_hi) / 2000.0)
        z_score = 1.0
        if zcr < z_lo:
            z_score = max(0.0, 1.0 - (z_lo - zcr) / 0.2)
        elif zcr > z_hi:
            z_score = max(0.0, 1.0 - (zcr - z_hi) / 0.2)
        mfcc_factor = min(1.0, mfcc_energy / 100.0 + 0.5)
        conf = c_score * z_score * mfcc_factor
        if conf > 0.1:
            results.append(TagPrediction(tag=inst, confidence=round(float(conf), 4)))
    results.sort(key=lambda p: -p.confidence)
    return results[:5]


# ---------- Era inference ----------

_ERA_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "pre-1970s": {"loudness": (-40, -20), "bandwidth": (0, 2500), "stereo_width": (0.0, 0.2)},
    "1970s": {"loudness": (-25, -14), "bandwidth": (1500, 4000), "stereo_width": (0.2, 0.5)},
    "1980s": {"loudness": (-20, -10), "bandwidth": (2000, 5000), "stereo_width": (0.4, 0.7)},
    "1990s": {"loudness": (-16, -8), "bandwidth": (2500, 5500), "stereo_width": (0.4, 0.8)},
    "2000s": {"loudness": (-12, -6), "bandwidth": (3000, 6000), "stereo_width": (0.5, 0.9)},
    "2010s-present": {"loudness": (-10, -4), "bandwidth": (3500, 8000), "stereo_width": (0.5, 1.0)},
}


def infer_era_tags(production_features: dict[str, float]) -> list[TagPrediction]:
    """Infer production era from loudness, bandwidth, stereo width."""
    loudness = production_features.get("loudness", -14.0)
    bandwidth = production_features.get("bandwidth", 3000.0)
    stereo = production_features.get("stereo_width", 0.5)

    results: list[TagPrediction] = []
    for era, ranges in _ERA_PROFILES.items():
        score = 1.0
        for dim, (lo, hi) in ranges.items():
            val = {"loudness": loudness, "bandwidth": bandwidth, "stereo_width": stereo}[dim]
            if lo <= val <= hi:
                centre = (lo + hi) / 2.0
                half = (hi - lo) / 2.0
                score *= 1.0 - abs(val - centre) / (half + 1e-9) * 0.2
            else:
                dist = min(abs(val - lo), abs(val - hi))
                scale = (hi - lo) if (hi - lo) > 0 else 1.0
                score *= max(0.0, 1.0 - dist / scale)
        if score > 0.05:
            results.append(TagPrediction(tag=era, confidence=round(float(score), 4)))
    results.sort(key=lambda p: -p.confidence)
    return results[:3]


# ---------- Combine ----------

def combine_tag_predictions(
    predictions_list: list[list[TagPrediction]],
    min_confidence: float = 0.3,
) -> list[TagPrediction]:
    """Merge multiple tag prediction lists, averaging duplicate tags."""
    agg: dict[str, list[float]] = {}
    for preds in predictions_list:
        for p in preds:
            agg.setdefault(p.tag, []).append(p.confidence)
    merged: list[TagPrediction] = []
    for tag, confs in agg.items():
        avg = float(np.mean(confs))
        if avg >= min_confidence:
            merged.append(TagPrediction(tag=tag, confidence=round(avg, 4)))
    merged.sort(key=lambda p: -p.confidence)
    return merged
