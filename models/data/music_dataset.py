"""Music dataset — generate and load audio feature data for the music
foundation model.

Two modes:
  1. Synthetic: generate plausible audio feature vectors with correlated
     VAD/mood targets for pretraining and development.
  2. From AudioSolver: extract features from real audio files via the
     existing AudioSolver pipeline.

Each sample is:
    features: (F,) float — 16 audio feature values
    feature_indices: (F,) int — feature dimension IDs [0..15]
    target_vad: (3,) float — valence, arousal, dominance in [0,1]
    target_mood: (5,) float — happy, energetic, calm, sad, tense in [0,1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from models.foundation.music_frontier_bridge import AUDIO_FEATURE_NAMES


@dataclass
class MusicDatasetConfig:
    n_samples: int = 1000
    n_features: int = 16
    seed: int = 2026


FEATURE_RANGES = {
    "spectral_centroid": (500.0, 8000.0),
    "spectral_rolloff": (1000.0, 12000.0),
    "spectral_bandwidth": (500.0, 4000.0),
    "zero_crossing_rate": (0.01, 0.3),
    "rms_energy": (0.01, 1.0),
    "dynamic_range": (5.0, 60.0),
    "bass_energy": (0.0, 1.0),
    "mid_energy": (0.0, 1.0),
    "treble_energy": (0.0, 1.0),
    "tempo_bpm": (40.0, 200.0),
    "key": (0.0, 11.0),
    "scale": (0.0, 1.0),
    "camelot": (0.0, 23.0),
    "valence": (0.0, 1.0),
    "arousal": (0.0, 1.0),
    "dominance": (0.0, 1.0),
}


def generate_synthetic_music_data(
    n_samples: int = 1000,
    n_features: int = 16,
    seed: int = 2026,
) -> dict[str, np.ndarray]:
    """Generate correlated synthetic audio feature data.

    Returns dict with keys: features, feature_indices, target_vad, target_mood.
    """
    rng = np.random.default_rng(seed)

    names = AUDIO_FEATURE_NAMES[:n_features]
    features = np.zeros((n_samples, n_features), dtype=np.float32)

    for i, name in enumerate(names):
        lo, hi = FEATURE_RANGES.get(name, (0.0, 1.0))
        features[:, i] = rng.uniform(lo, hi, n_samples).astype(np.float32)

    brightness = features[:, names.index("spectral_centroid")] if "spectral_centroid" in names else rng.uniform(0, 1, n_samples)
    brightness = (brightness - brightness.min()) / (brightness.max() - brightness.min() + 1e-8)

    energy = features[:, names.index("rms_energy")] if "rms_energy" in names else rng.uniform(0, 1, n_samples)
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

    tempo_norm = features[:, names.index("tempo_bpm")] if "tempo_bpm" in names else rng.uniform(0, 1, n_samples)
    tempo_norm = (tempo_norm - tempo_norm.min()) / (tempo_norm.max() - tempo_norm.min() + 1e-8)

    valence = np.clip(brightness * energy * 1.5 + rng.normal(0, 0.1, n_samples), 0, 1).astype(np.float32)
    arousal = np.clip(tempo_norm * energy * 1.5 + rng.normal(0, 0.1, n_samples), 0, 1).astype(np.float32)
    dominance = np.clip(energy * brightness * 1.2 + rng.normal(0, 0.1, n_samples), 0, 1).astype(np.float32)

    if "valence" in names:
        features[:, names.index("valence")] = valence
    if "arousal" in names:
        features[:, names.index("arousal")] = arousal
    if "dominance" in names:
        features[:, names.index("dominance")] = dominance

    target_vad = np.stack([valence, arousal, dominance], axis=-1)

    happy = np.clip(valence * energy * 1.5, 0, 1).astype(np.float32)
    energetic = np.clip(arousal * energy * 1.5, 0, 1).astype(np.float32)
    calm = np.clip((1 - arousal) * (1 - energy), 0, 1).astype(np.float32)
    sad = np.clip((1 - valence) * 0.8 + rng.normal(0, 0.05, n_samples), 0, 1).astype(np.float32)
    tense = np.clip(arousal * (1 - valence) + rng.normal(0, 0.05, n_samples), 0, 1).astype(np.float32)
    target_mood = np.stack([happy, energetic, calm, sad, tense], axis=-1)

    feature_indices = np.arange(n_features, dtype=np.int64)
    feature_indices = np.tile(feature_indices, (n_samples, 1))

    return {
        "features": features,
        "feature_indices": feature_indices,
        "target_vad": target_vad,
        "target_mood": target_mood,
        "feature_names": names,
    }


class MusicFeatureDataset(Dataset):
    """PyTorch dataset for music foundation model training."""

    def __init__(
        self,
        features: np.ndarray,
        feature_indices: np.ndarray,
        target_vad: np.ndarray,
        target_mood: np.ndarray,
    ):
        self.features = torch.from_numpy(features).float()
        self.feature_indices = torch.from_numpy(feature_indices).long()
        self.target_vad = torch.from_numpy(target_vad).float()
        self.target_mood = torch.from_numpy(target_mood).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "feature_indices": self.feature_indices[idx],
            "target_vad": self.target_vad[idx],
            "target_mood": self.target_mood[idx],
        }

    @classmethod
    def from_synthetic(
        cls, n_samples: int = 1000, n_features: int = 16, seed: int = 2026,
    ) -> "MusicFeatureDataset":
        data = generate_synthetic_music_data(n_samples, n_features, seed)
        return cls(
            features=data["features"],
            feature_indices=data["feature_indices"],
            target_vad=data["target_vad"],
            target_mood=data["target_mood"],
        )
