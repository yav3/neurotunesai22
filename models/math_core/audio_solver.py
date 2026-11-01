"""Audio feature analysis — spectral, VAD estimation, Camelot mixing.

Spectral centroid:
    c = Σ(f_i · |X(f_i)|) / Σ|X(f_i)|

VAD from spectral features (real_audio_analyzer.py formulas):
    brightness = spectral_centroid / 4000
    valence    = clip(brightness · energy · 2, 0, 1)
    arousal    = clip(rhythm_strength · energy · 3, 0, 1)
    dominance  = clip(energy · brightness, 0, 1)

Mood scores:
    happy     = clip(brightness · energy · 2, 0, 1)
    energetic = clip(rhythm · energy · 3, 0, 1)
    calm      = clip((1-rhythm) · (1-energy), 0, 1)
    sad       = clip(1 - brightness·0.5, 0, 1)
    tense     = 1 - calm

Camelot wheel:
    12 keys × {major(B), minor(A)} = 24 positions on the circle of fifths.
    Compatible mixing: same position, ±1 number same letter, A↔B same number.

Pitch class from frequency:
    semitone = round(12 · log₂(f / 440))
    pitch_class = ((semitone % 12) + 12) % 12
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioFeatures:
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_bandwidth: float = 0.0
    zero_crossing_rate: float = 0.0
    rms_energy: float = 0.0
    dynamic_range: float = 0.0
    bass_energy: float = 0.0
    mid_energy: float = 0.0
    treble_energy: float = 0.0
    tempo_bpm: float = 120.0
    key: str = "C"
    scale: str = "major"
    camelot: str = "8B"
    valence: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5


CAMELOT_WHEEL: dict[tuple[str, str], str] = {
    ("C", "major"): "8B",  ("C", "minor"): "5A",
    ("C#", "major"): "3B", ("C#", "minor"): "12A",
    ("Db", "major"): "3B", ("Db", "minor"): "12A",
    ("D", "major"): "10B", ("D", "minor"): "7A",
    ("D#", "major"): "5B", ("D#", "minor"): "2A",
    ("Eb", "major"): "5B", ("Eb", "minor"): "2A",
    ("E", "major"): "12B", ("E", "minor"): "9A",
    ("F", "major"): "7B",  ("F", "minor"): "4A",
    ("F#", "major"): "2B", ("F#", "minor"): "11A",
    ("Gb", "major"): "2B", ("Gb", "minor"): "11A",
    ("G", "major"): "9B",  ("G", "minor"): "6A",
    ("G#", "major"): "4B", ("G#", "minor"): "1A",
    ("Ab", "major"): "4B", ("Ab", "minor"): "1A",
    ("A", "major"): "11B", ("A", "minor"): "8A",
    ("A#", "major"): "6B", ("A#", "minor"): "3A",
    ("Bb", "major"): "6B", ("Bb", "minor"): "3A",
    ("B", "major"): "1B",  ("B", "minor"): "10A",
}


def _parse_camelot(camelot: str) -> tuple[int, str]:
    """Parse '8B' → (8, 'B')."""
    letter = camelot[-1].upper()
    number = int(camelot[:-1])
    return number, letter


class AudioSolver:

    @staticmethod
    def spectral_centroid(magnitudes: np.ndarray, freqs: np.ndarray) -> float:
        """c = Σ(f_i · |X_i|) / Σ|X_i|."""
        total = np.sum(magnitudes)
        if total <= 0:
            return 0.0
        return float(np.sum(freqs * magnitudes) / total)

    @staticmethod
    def spectral_rolloff(
        magnitudes: np.ndarray, freqs: np.ndarray, percentile: float = 0.85,
    ) -> float:
        """Smallest f such that cumulative energy ≥ percentile · total."""
        total = np.sum(magnitudes)
        if total <= 0:
            return 0.0
        threshold = percentile * total
        cumsum = np.cumsum(magnitudes)
        idx = np.searchsorted(cumsum, threshold)
        idx = min(idx, len(freqs) - 1)
        return float(freqs[idx])

    @staticmethod
    def spectral_entropy(magnitudes: np.ndarray) -> float:
        """H = -Σ p_i log(p_i), p_i = |X_i| / Σ|X|."""
        total = np.sum(magnitudes)
        if total <= 0:
            return 0.0
        p = magnitudes / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def band_energies(
        magnitudes: np.ndarray, freqs: np.ndarray,
    ) -> dict[str, float]:
        """Energy in bass (0–250 Hz), mid (250–4000 Hz), treble (4000+ Hz)."""
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        sq = magnitudes ** 2 * df
        bass_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 4000)
        treble_mask = freqs >= 4000
        return {
            "bass": float(np.sum(sq[bass_mask])),
            "mid": float(np.sum(sq[mid_mask])),
            "treble": float(np.sum(sq[treble_mask])),
        }

    @staticmethod
    def brightness(spectral_centroid: float, reference: float = 4000.0) -> float:
        return float(np.clip(spectral_centroid / reference, 0.0, 1.0))

    @staticmethod
    def estimate_valence(brightness: float, rms_energy: float) -> float:
        return float(np.clip(brightness * rms_energy * 2, 0.0, 1.0))

    @staticmethod
    def estimate_arousal(rhythm_strength: float, rms_energy: float) -> float:
        return float(np.clip(rhythm_strength * rms_energy * 3, 0.0, 1.0))

    @staticmethod
    def estimate_dominance(rms_energy: float, brightness: float) -> float:
        return float(np.clip(rms_energy * brightness, 0.0, 1.0))

    @staticmethod
    def mood_scores(
        brightness: float, rms_energy: float, rhythm_strength: float,
    ) -> dict[str, float]:
        happy = float(np.clip(brightness * rms_energy * 2, 0, 1))
        energetic = float(np.clip(rhythm_strength * rms_energy * 3, 0, 1))
        calm = float(np.clip((1 - rhythm_strength) * (1 - rms_energy), 0, 1))
        sad = float(np.clip(1 - brightness * 0.5, 0, 1))
        tense = 1.0 - calm
        return {
            "happy": happy, "energetic": energetic, "calm": calm,
            "sad": sad, "tense": tense,
        }

    @staticmethod
    def key_to_camelot(key: str, scale: str = "major") -> str:
        return CAMELOT_WHEEL.get((key, scale), "1A")

    @staticmethod
    def camelot_compatibility(key1: str, key2: str) -> float:
        """Score ∈ [0,1]. Same=1.0, ±1 same letter=0.8, A↔B same number=0.6."""
        if key1 == key2:
            return 1.0
        n1, l1 = _parse_camelot(key1)
        n2, l2 = _parse_camelot(key2)
        diff = abs(n1 - n2)
        diff = min(diff, 12 - diff)  # wrap around the wheel
        if l1 == l2 and diff <= 1:
            return 1.0 if diff == 0 else 0.8
        if l1 != l2 and diff == 0:
            return 0.6
        return 0.0

    @staticmethod
    def frequency_to_pitch_class(frequency: float) -> int:
        """semitone = round(12·log₂(f/440)), pitch_class = ((s%12)+12)%12."""
        if frequency <= 0:
            return 0
        semitone = round(12 * math.log2(frequency / 440.0))
        return ((semitone % 12) + 12) % 12
