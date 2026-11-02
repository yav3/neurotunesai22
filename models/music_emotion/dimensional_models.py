"""Dimensional emotion models: VAD mapping, circumplex, Plutchik."""
from __future__ import annotations

import numpy as np


PLUTCHIK_EMOTIONS = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

PLUTCHIK_VAD_CENTERS = {
    "joy":          ( 0.9,  0.7,  0.7),
    "trust":        ( 0.6,  0.2,  0.4),
    "fear":         (-0.7,  0.8, -0.8),
    "surprise":     ( 0.2,  0.9,  0.1),
    "sadness":      (-0.8, -0.5, -0.4),
    "disgust":      (-0.6,  0.3,  0.3),
    "anger":        (-0.5,  0.8,  0.6),
    "anticipation": ( 0.4,  0.6,  0.5),
}


class DimensionalEmotionModel:
    """Map audio features to Valence-Arousal-Dominance space."""

    @staticmethod
    def vad_from_features(features: dict) -> tuple[float, float, float]:
        tempo = features.get("tempo", 120)
        mode = features.get("mode", 1)
        centroid = features.get("spectral_centroid", 2000)
        loudness = features.get("loudness", -20)
        flatness = features.get("spectral_flatness", 0.1)

        valence = 0.3 * (mode * 2 - 1) + 0.2 * np.tanh((tempo - 120) / 40) + 0.1 * np.tanh((centroid - 2000) / 1000)
        arousal = 0.35 * np.tanh((tempo - 100) / 50) + 0.25 * np.tanh((loudness + 10) / 10) + 0.15 * np.tanh((centroid - 1500) / 1000)
        dominance = 0.25 * np.tanh((loudness + 10) / 10) + 0.2 * np.tanh((tempo - 110) / 40) + 0.15 * np.tanh((centroid - 1800) / 800)
        valence = float(np.clip(valence, -1, 1))
        arousal = float(np.clip(arousal, -1, 1))
        dominance = float(np.clip(dominance, -1, 1))
        return valence, arousal, dominance

    @staticmethod
    def circumplex_quadrant(valence: float, arousal: float) -> str:
        if valence >= 0 and arousal >= 0:
            return "happy/excited"
        elif valence < 0 and arousal >= 0:
            return "angry/tense"
        elif valence < 0 and arousal < 0:
            return "sad/depressed"
        else:
            return "calm/relaxed"

    @staticmethod
    def emotion_trajectory(
        vad_sequence: np.ndarray, window: int = 5
    ) -> dict:
        if len(vad_sequence) < 2:
            return {"smoothed": vad_sequence, "arcs": {}}
        kernel = np.ones(min(window, len(vad_sequence))) / min(window, len(vad_sequence))
        smoothed = np.zeros_like(vad_sequence)
        for d in range(vad_sequence.shape[1] if vad_sequence.ndim > 1 else 1):
            col = vad_sequence[:, d] if vad_sequence.ndim > 1 else vad_sequence
            smoothed_col = np.convolve(col, kernel, mode="same")
            if vad_sequence.ndim > 1:
                smoothed[:, d] = smoothed_col
            else:
                smoothed = smoothed_col
        arcs = {}
        dims = ["valence", "arousal", "dominance"]
        for d in range(min(3, vad_sequence.shape[1] if vad_sequence.ndim > 1 else 1)):
            col = smoothed[:, d] if smoothed.ndim > 1 else smoothed
            slope = col[-1] - col[0]
            if slope > 0.1:
                arcs[dims[d]] = "rising"
            elif slope < -0.1:
                arcs[dims[d]] = "falling"
            else:
                arcs[dims[d]] = "stable"
        return {"smoothed": smoothed, "arcs": arcs}

    @staticmethod
    def arousal_energy_correlation(
        arousal_values: np.ndarray, energy_values: np.ndarray
    ) -> float:
        if len(arousal_values) < 3:
            return 0.0
        a = np.asarray(arousal_values)
        e = np.asarray(energy_values)
        n = min(len(a), len(e))
        a, e = a[:n], e[:n]
        if np.std(a) < 1e-10 or np.std(e) < 1e-10:
            return 0.0
        return float(np.corrcoef(a, e)[0, 1])

    @staticmethod
    def valence_mode_mapping(key_name: str, mode: str) -> float:
        key_offsets = {
            "C": 0.0, "C#": -0.05, "D": 0.05, "D#": -0.1, "Eb": -0.1,
            "E": 0.1, "F": -0.05, "F#": 0.0, "G": 0.1, "G#": -0.05,
            "Ab": -0.05, "A": 0.05, "A#": -0.1, "Bb": -0.1, "B": 0.0,
        }
        base = 0.3 if mode.lower() == "major" else -0.3
        offset = key_offsets.get(key_name, 0.0)
        return float(np.clip(base + offset, -1, 1))

    @staticmethod
    def dominance_from_dynamics(
        dynamic_range: float, tempo: float, spectral_centroid: float
    ) -> float:
        d = (
            0.3 * np.tanh(dynamic_range / 30)
            + 0.3 * np.tanh((tempo - 100) / 50)
            + 0.2 * np.tanh((spectral_centroid - 2000) / 1000)
        )
        return float(np.clip(d, -1, 1))

    @staticmethod
    def emotion_distance(vad_a: tuple, vad_b: tuple) -> float:
        weights = np.array([1.0, 0.8, 0.6])
        a = np.array(vad_a[:3])
        b = np.array(vad_b[:3])
        return float(np.sqrt(np.sum(weights * (a - b) ** 2)))

    @staticmethod
    def plutchik_from_vad(
        valence: float, arousal: float, dominance: float
    ) -> tuple[str, str]:
        vad = np.array([valence, arousal, dominance])
        best_emotion = "joy"
        best_dist = np.inf
        for name, center in PLUTCHIK_VAD_CENTERS.items():
            d = np.linalg.norm(vad - np.array(center))
            if d < best_dist:
                best_dist = d
                best_emotion = name
        intensity_val = np.linalg.norm(vad)
        if intensity_val > 1.2:
            intensity = "intense"
        elif intensity_val > 0.6:
            intensity = "moderate"
        else:
            intensity = "mild"
        return best_emotion, intensity


MOOD_LABELS = [
    "happy", "sad", "energetic", "calm",
    "aggressive", "romantic", "melancholic", "uplifting",
]

MOOD_COLORS = {
    "happy": (255, 223, 0),
    "sad": (70, 90, 150),
    "energetic": (255, 100, 0),
    "calm": (100, 180, 220),
    "aggressive": (200, 30, 30),
    "romantic": (200, 100, 150),
    "melancholic": (100, 100, 130),
    "uplifting": (255, 200, 50),
}


class MusicMoodClassifier:
    """Rule-based mood classification from audio features."""

    def __init__(self, n_moods: int = 8):
        self.n_moods = n_moods

    @staticmethod
    def mood_vocabulary() -> list[str]:
        return list(MOOD_LABELS)

    def feature_to_mood_scores(self, features: np.ndarray) -> np.ndarray:
        n = min(len(features), self.n_moods)
        W = np.eye(self.n_moods, len(features)) * 0.5
        scores = W @ features
        scores = 1 / (1 + np.exp(-scores))
        return scores

    @staticmethod
    def mood_from_tempo_mode(
        tempo: float, is_major: bool, energy: float
    ) -> str:
        if is_major and tempo > 120 and energy > 0.5:
            return "happy"
        if not is_major and tempo < 80:
            return "melancholic"
        if tempo > 140 and energy > 0.7:
            return "energetic"
        if tempo < 80 and energy < 0.3:
            return "calm"
        if energy > 0.8 and not is_major:
            return "aggressive"
        if is_major and tempo < 100 and energy < 0.5:
            return "romantic"
        if not is_major and energy < 0.4:
            return "sad"
        return "uplifting"

    @staticmethod
    def mood_transition_probability(
        current_mood: str, duration_fraction: float
    ) -> dict[str, float]:
        base = {m: 0.05 for m in MOOD_LABELS}
        base[current_mood] = 0.6
        if duration_fraction > 0.7:
            if current_mood in ("calm", "melancholic"):
                base["uplifting"] = 0.15
            elif current_mood in ("energetic", "aggressive"):
                base["calm"] = 0.15
        total = sum(base.values())
        return {k: v / total for k, v in base.items()}

    @staticmethod
    def mood_consistency(mood_scores_sequence: np.ndarray) -> float:
        if len(mood_scores_sequence) < 2:
            return 1.0
        entropies = []
        for scores in mood_scores_sequence:
            p = np.abs(scores) + 1e-10
            p = p / p.sum()
            entropies.append(-np.sum(p * np.log(p)))
        return float(1.0 - np.mean(entropies) / np.log(len(mood_scores_sequence[0])))

    @staticmethod
    def mix_moods(
        mood_a: str, score_a: float, mood_b: str, score_b: float
    ) -> str:
        if mood_a == mood_b:
            return mood_a
        if score_a >= score_b:
            return f"{mood_a}-{mood_b}"
        return f"{mood_b}-{mood_a}"

    @staticmethod
    def mood_to_color(mood_label: str) -> tuple[int, int, int]:
        return MOOD_COLORS.get(mood_label, (128, 128, 128))
