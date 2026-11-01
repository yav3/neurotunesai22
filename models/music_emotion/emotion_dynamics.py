"""Emotion trajectory tracking and arc classification over time."""
from dataclasses import dataclass
import numpy as np


@dataclass
class EmotionTrajectory:
    timestamps: np.ndarray
    valence_curve: np.ndarray
    arousal_curve: np.ndarray
    dominance_curve: np.ndarray


def track_emotion_over_time(features_sequence: list[dict],
                            window_sec: float = 5.0) -> EmotionTrajectory:
    n = len(features_sequence)
    if n == 0:
        return EmotionTrajectory(np.array([]), np.array([]), np.array([]), np.array([]))
    timestamps = np.arange(n) * window_sec
    valence = np.zeros(n)
    arousal = np.zeros(n)
    dominance = np.zeros(n)
    for i, feat in enumerate(features_sequence):
        valence[i] = feat.get("valence", 0.0)
        arousal[i] = feat.get("arousal", 0.0)
        dominance[i] = feat.get("dominance", 0.0)
    return EmotionTrajectory(timestamps, valence, arousal, dominance)


def emotion_volatility(trajectory: EmotionTrajectory) -> float:
    if len(trajectory.valence_curve) < 2:
        return 0.0
    dv = np.abs(np.diff(trajectory.valence_curve))
    da = np.abs(np.diff(trajectory.arousal_curve))
    dd = np.abs(np.diff(trajectory.dominance_curve))
    return float(np.mean(dv) + np.mean(da) + np.mean(dd)) / 3.0


def emotion_arc_type(trajectory: EmotionTrajectory) -> str:
    v = trajectory.valence_curve
    if len(v) < 3:
        return "flat"
    diffs = np.diff(v)
    trend = np.polyfit(np.arange(len(v)), v, 1)[0]
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    if sign_changes > len(v) * 0.4:
        return "oscillating"
    peak_pos = np.argmax(v)
    valley_pos = np.argmin(v)
    if abs(trend) < 0.01 * (np.max(v) - np.min(v) + 1e-10):
        return "flat"
    if 0.2 * len(v) < peak_pos < 0.8 * len(v) and v[peak_pos] - v[0] > 0.1:
        return "peak"
    if 0.2 * len(v) < valley_pos < 0.8 * len(v) and v[0] - v[valley_pos] > 0.1:
        return "valley"
    if trend > 0:
        return "rising"
    return "falling"


def emotional_surprise(trajectory: EmotionTrajectory) -> np.ndarray:
    n = len(trajectory.valence_curve)
    if n < 3:
        return np.zeros(max(n, 0))
    surprise = np.zeros(n)
    for i in range(2, n):
        pred_v = 2 * trajectory.valence_curve[i-1] - trajectory.valence_curve[i-2]
        pred_a = 2 * trajectory.arousal_curve[i-1] - trajectory.arousal_curve[i-2]
        pred_d = 2 * trajectory.dominance_curve[i-1] - trajectory.dominance_curve[i-2]
        dv = trajectory.valence_curve[i] - pred_v
        da = trajectory.arousal_curve[i] - pred_a
        dd = trajectory.dominance_curve[i] - pred_d
        surprise[i] = np.sqrt(dv**2 + da**2 + dd**2)
    return surprise


def valence_arousal_path_length(trajectory: EmotionTrajectory) -> float:
    n = len(trajectory.valence_curve)
    if n < 2:
        return 0.0
    dv = np.diff(trajectory.valence_curve)
    da = np.diff(trajectory.arousal_curve)
    return float(np.sum(np.sqrt(dv**2 + da**2)))


def emotional_centroid(trajectory: EmotionTrajectory) -> tuple[float, float, float]:
    if len(trajectory.valence_curve) == 0:
        return (0.0, 0.0, 0.0)
    return (float(np.mean(trajectory.valence_curve)),
            float(np.mean(trajectory.arousal_curve)),
            float(np.mean(trajectory.dominance_curve)))
