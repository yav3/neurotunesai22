"""Mixed and ambiguous emotion modeling for music.

Models situations where music evokes simultaneous, potentially
contradictory emotions (e.g., bittersweet, nostalgic joy).
Extends the VAD model with blend weights and temporal evolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class BasicEmotion(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


@dataclass
class VADPoint:
    valence: float
    arousal: float
    dominance: float = 0.5


EMOTION_VAD: Dict[BasicEmotion, VADPoint] = {
    BasicEmotion.JOY: VADPoint(0.9, 0.7, 0.7),
    BasicEmotion.SADNESS: VADPoint(0.1, 0.2, 0.2),
    BasicEmotion.ANGER: VADPoint(0.15, 0.85, 0.8),
    BasicEmotion.FEAR: VADPoint(0.1, 0.8, 0.15),
    BasicEmotion.SURPRISE: VADPoint(0.6, 0.9, 0.4),
    BasicEmotion.DISGUST: VADPoint(0.15, 0.5, 0.6),
    BasicEmotion.TRUST: VADPoint(0.8, 0.3, 0.6),
    BasicEmotion.ANTICIPATION: VADPoint(0.65, 0.65, 0.55),
}


@dataclass
class EmotionBlend:
    components: Dict[BasicEmotion, float]
    vad: VADPoint
    ambiguity: float
    tension: float


def normalize_weights(weights: Dict[BasicEmotion, float]) -> Dict[BasicEmotion, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def blend_vad(components: Dict[BasicEmotion, float]) -> VADPoint:
    normed = normalize_weights(components)
    v = sum(normed[e] * EMOTION_VAD[e].valence for e in normed)
    a = sum(normed[e] * EMOTION_VAD[e].arousal for e in normed)
    d = sum(normed[e] * EMOTION_VAD[e].dominance for e in normed)
    return VADPoint(v, a, d)


def emotional_ambiguity(components: Dict[BasicEmotion, float]) -> float:
    normed = normalize_weights(components)
    weights = np.array(list(normed.values()))
    if len(weights) <= 1:
        return 0.0
    entropy = -np.sum(weights * np.log2(np.maximum(weights, 1e-12)))
    max_entropy = np.log2(len(weights))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def emotional_tension(components: Dict[BasicEmotion, float]) -> float:
    normed = normalize_weights(components)
    emotions = list(normed.keys())
    if len(emotions) < 2:
        return 0.0
    total_conflict = 0.0
    pairs = 0
    for i in range(len(emotions)):
        for j in range(i + 1, len(emotions)):
            va = EMOTION_VAD[emotions[i]]
            vb = EMOTION_VAD[emotions[j]]
            dist = math.sqrt(
                (va.valence - vb.valence) ** 2
                + (va.arousal - vb.arousal) ** 2
                + (va.dominance - vb.dominance) ** 2
            )
            w = normed[emotions[i]] * normed[emotions[j]]
            total_conflict += dist * w
            pairs += 1
    max_dist = math.sqrt(3)
    return min(total_conflict / (max_dist * 0.25), 1.0)


def create_blend(components: Dict[BasicEmotion, float]) -> EmotionBlend:
    normed = normalize_weights(components)
    return EmotionBlend(
        components=normed,
        vad=blend_vad(normed),
        ambiguity=emotional_ambiguity(normed),
        tension=emotional_tension(normed),
    )


NAMED_BLENDS: Dict[str, Dict[BasicEmotion, float]] = {
    "bittersweet": {BasicEmotion.JOY: 0.45, BasicEmotion.SADNESS: 0.55},
    "nostalgia": {BasicEmotion.JOY: 0.35, BasicEmotion.SADNESS: 0.4, BasicEmotion.ANTICIPATION: 0.25},
    "awe": {BasicEmotion.SURPRISE: 0.4, BasicEmotion.FEAR: 0.25, BasicEmotion.TRUST: 0.35},
    "melancholy": {BasicEmotion.SADNESS: 0.6, BasicEmotion.TRUST: 0.2, BasicEmotion.ANTICIPATION: 0.2},
    "triumph": {BasicEmotion.JOY: 0.5, BasicEmotion.ANGER: 0.15, BasicEmotion.ANTICIPATION: 0.35},
    "suspense": {BasicEmotion.FEAR: 0.35, BasicEmotion.ANTICIPATION: 0.5, BasicEmotion.SURPRISE: 0.15},
    "serenity": {BasicEmotion.JOY: 0.4, BasicEmotion.TRUST: 0.5, BasicEmotion.SADNESS: 0.1},
    "agitation": {BasicEmotion.ANGER: 0.3, BasicEmotion.FEAR: 0.3, BasicEmotion.ANTICIPATION: 0.4},
}


def named_blend(name: str) -> EmotionBlend:
    if name not in NAMED_BLENDS:
        raise ValueError(f"Unknown blend: {name}. Available: {list(NAMED_BLENDS.keys())}")
    return create_blend(NAMED_BLENDS[name])


@dataclass
class EmotionTrajectory:
    times: np.ndarray
    blends: List[EmotionBlend]


def interpolate_blends(
    blend_a: EmotionBlend,
    blend_b: EmotionBlend,
    t: float,
) -> EmotionBlend:
    t = max(0.0, min(1.0, t))
    all_emotions = set(blend_a.components.keys()) | set(blend_b.components.keys())
    components = {}
    for e in all_emotions:
        wa = blend_a.components.get(e, 0.0)
        wb = blend_b.components.get(e, 0.0)
        components[e] = wa * (1 - t) + wb * t
    return create_blend(components)


def emotion_trajectory(
    keyframes: List[Tuple[float, EmotionBlend]],
    n_points: int = 100,
) -> EmotionTrajectory:
    if len(keyframes) < 2:
        raise ValueError("Need at least 2 keyframes")
    keyframes = sorted(keyframes, key=lambda x: x[0])
    times = np.linspace(keyframes[0][0], keyframes[-1][0], n_points)
    blends = []
    for t in times:
        idx = 0
        for i in range(len(keyframes) - 1):
            if keyframes[i][0] <= t <= keyframes[i + 1][0]:
                idx = i
                break
        span = keyframes[idx + 1][0] - keyframes[idx][0]
        frac = (t - keyframes[idx][0]) / span if span > 0 else 0.0
        blends.append(interpolate_blends(keyframes[idx][1], keyframes[idx + 1][1], frac))
    return EmotionTrajectory(times=times, blends=blends)


def vad_from_features(
    tempo: float,
    mode_major: bool,
    spectral_centroid: float,
    rms_energy: float,
    harmonic_ratio: float = 0.5,
) -> VADPoint:
    valence = 0.3 + 0.3 * float(mode_major) + 0.2 * min(harmonic_ratio, 1.0) + 0.1 * min(spectral_centroid / 4000, 1.0)
    arousal = 0.1 + 0.4 * min(tempo / 180, 1.0) + 0.3 * min(rms_energy, 1.0) + 0.15 * min(spectral_centroid / 4000, 1.0)
    dominance = 0.3 + 0.3 * min(rms_energy, 1.0) + 0.2 * min(tempo / 180, 1.0) + 0.1 * float(mode_major)
    return VADPoint(
        valence=max(0, min(1, valence)),
        arousal=max(0, min(1, arousal)),
        dominance=max(0, min(1, dominance)),
    )


def nearest_blend(vad: VADPoint, top_k: int = 3) -> EmotionBlend:
    distances = []
    for name, comps in NAMED_BLENDS.items():
        blend = create_blend(comps)
        d = math.sqrt(
            (vad.valence - blend.vad.valence) ** 2
            + (vad.arousal - blend.vad.arousal) ** 2
            + (vad.dominance - blend.vad.dominance) ** 2
        )
        distances.append((name, d))
    distances.sort(key=lambda x: x[1])
    best = distances[0][0]
    return create_blend(NAMED_BLENDS[best])
