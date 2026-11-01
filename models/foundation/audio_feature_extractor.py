"""AudioFeatureExtractor — bridges domain extractors to the foundation model.

Maps raw audio waveforms to the 16-dimensional AudioFeatures vector that
the MusicFoundationEncoder expects:

  [spectral_centroid, spectral_rolloff, spectral_bandwidth, zero_crossing_rate,
   rms_energy, dynamic_range, bass_energy, mid_energy, treble_energy,
   tempo_bpm, key, scale, camelot, valence, arousal, dominance]

Uses the domain modules (music_features, music_theory, music_emotion) to
extract real features from audio, then maps and normalizes into the
foundation model's feature space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.music_features.spectral import SpectralExtractor
from models.music_features.rhythm import RhythmExtractor
from models.music_features.harmonic import HarmonicExtractor
from models.music_features.dynamics import DynamicsExtractor
from models.music_features.timbre import TimbreExtractor
from models.music_emotion.music_emotion_profile import (
    MusicEmotionMapper,
    MusicFeatureVector,
)
from models.music_theory.camelot import key_to_camelot

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

FOUNDATION_FEATURE_NAMES = [
    "spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
    "zero_crossing_rate", "rms_energy", "dynamic_range",
    "bass_energy", "mid_energy", "treble_energy",
    "tempo_bpm", "key", "scale", "camelot",
    "valence", "arousal", "dominance",
]


@dataclass
class ExtractedAudioFeatures:
    features: np.ndarray  # (16,) float32
    feature_names: list[str]
    metadata: dict


class AudioFeatureExtractor:
    """Extract the 16-dim foundation feature vector from raw audio."""

    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.spectral = SpectralExtractor(sr=sr)
        self.rhythm = RhythmExtractor(sr=sr)
        self.harmonic = HarmonicExtractor(sr=sr)
        self.dynamics = DynamicsExtractor()
        self.timbre = TimbreExtractor()
        self.emotion_mapper = MusicEmotionMapper()

    def extract(self, audio: np.ndarray, sr: int | None = None) -> ExtractedAudioFeatures:
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float64)
        effective_sr = sr or self.sr

        spec = self.spectral.extract(audio)
        rhythm = self.rhythm.extract(audio)
        harm = self.harmonic.extract(audio)
        dyn = self.dynamics.extract(audio, sr=effective_sr)
        timb = self.timbre.extract(audio, sr=effective_sr)

        key_idx = _key_to_index(harm.key)
        scale_val = 1.0 if harm.mode == "major" else 0.0
        camelot_val = _key_to_camelot_number(harm.key, harm.mode)

        mode_numeric = 1.0 if harm.mode == "major" else 0.0
        emotion_features = MusicFeatureVector(
            bpm=rhythm.bpm,
            loudness_db=dyn.loudness_lufs_approx,
            energy=dyn.rms_mean,
            mode=mode_numeric,
            brightness=timb.brightness,
            danceability=_estimate_danceability(rhythm, dyn),
        )
        profile = self.emotion_mapper.features_to_emotion_profile(emotion_features)

        features = np.array([
            spec.spectral_centroid,
            spec.spectral_rolloff,
            spec.spectral_bandwidth,
            spec.zero_crossing_rate,
            dyn.rms_mean,
            dyn.dynamic_range_db,
            _band_energy(spec, "bass"),
            _band_energy(spec, "mid"),
            _band_energy(spec, "treble"),
            rhythm.bpm,
            float(key_idx),
            scale_val,
            float(camelot_val),
            profile.valence,
            profile.arousal,
            profile.dominance,
        ], dtype=np.float32)

        metadata = {
            "key": harm.key,
            "mode": harm.mode,
            "key_confidence": harm.key_confidence,
            "bpm_confidence": rhythm.bpm_confidence,
            "duration_s": len(audio) / effective_sr,
        }

        return ExtractedAudioFeatures(
            features=features,
            feature_names=FOUNDATION_FEATURE_NAMES,
            metadata=metadata,
        )

    def extract_batch(
        self, audio_list: list[np.ndarray], sr: int | None = None,
    ) -> np.ndarray:
        return np.stack([self.extract(a, sr).features for a in audio_list])


def _key_to_index(key: str) -> int:
    key = key.strip()
    if key in PITCH_CLASSES:
        return PITCH_CLASSES.index(key)
    for i, pc in enumerate(PITCH_CLASSES):
        if key.lower() == pc.lower():
            return i
    return 0


def _key_to_camelot_number(key: str, mode: str) -> int:
    try:
        ck = key_to_camelot(key, mode)
        return ck.number * 2 + (0 if ck.letter == "A" else 1)
    except (ValueError, KeyError):
        return 0


def _band_energy(spec, band: str) -> float:
    chroma = spec.chroma if spec.chroma else [0.0] * 12
    if band == "bass":
        return float(np.mean(chroma[:4])) if len(chroma) >= 4 else 0.0
    elif band == "mid":
        return float(np.mean(chroma[4:8])) if len(chroma) >= 8 else 0.0
    else:
        return float(np.mean(chroma[8:])) if len(chroma) > 8 else 0.0


def _estimate_danceability(rhythm, dyn) -> float:
    tempo_score = 1.0 - abs(rhythm.bpm - 120.0) / 80.0 if rhythm.bpm > 0 else 0.0
    tempo_score = max(0.0, min(1.0, tempo_score))
    regularity = rhythm.beat_regularity
    energy = dyn.rms_mean
    return float(np.clip(0.4 * tempo_score + 0.3 * regularity + 0.3 * energy, 0, 1))
