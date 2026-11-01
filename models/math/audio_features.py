"""Differentiable audio feature estimation and Camelot compatibility.

VAD estimation (learnable approximation of spectral → mood mapping):
    brightness = σ(W_b · x)
    energy     = σ(W_e · x)
    rhythm     = σ(W_r · x)
    valence    = clip(brightness · energy · 2, 0, 1)
    arousal    = clip(rhythm · energy · 3, 0, 1)
    dominance  = clip(energy · brightness, 0, 1)

Camelot compatibility: learned embedding space over 24 keys.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AudioVADEstimator(nn.Module):
    """Estimate valence/arousal/dominance from spectral features."""

    def __init__(self, n_spectral_features: int = 13):
        super().__init__()
        self.brightness_head = nn.Sequential(nn.Linear(n_spectral_features, 1), nn.Sigmoid())
        self.energy_head = nn.Sequential(nn.Linear(n_spectral_features, 1), nn.Sigmoid())
        self.rhythm_head = nn.Sequential(nn.Linear(n_spectral_features, 1), nn.Sigmoid())

    def forward(self, spectral_features: torch.Tensor) -> dict[str, torch.Tensor]:
        brightness = self.brightness_head(spectral_features).squeeze(-1)
        energy = self.energy_head(spectral_features).squeeze(-1)
        rhythm = self.rhythm_head(spectral_features).squeeze(-1)

        valence = (brightness * energy * 2).clamp(0, 1)
        arousal = (rhythm * energy * 3).clamp(0, 1)
        dominance = (energy * brightness).clamp(0, 1)

        happy = (brightness * energy * 2).clamp(0, 1)
        energetic = (rhythm * energy * 3).clamp(0, 1)
        calm = ((1 - rhythm) * (1 - energy)).clamp(0, 1)
        sad = (1 - brightness * 0.5).clamp(0, 1)
        tense = 1 - calm

        return {
            "brightness": brightness,
            "energy": energy,
            "rhythm_strength": rhythm,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "mood_scores": {
                "happy": happy, "energetic": energetic, "calm": calm,
                "sad": sad, "tense": tense,
            },
        }


class CamelotEmbedding(nn.Module):
    """Learned embedding for 24 Camelot keys (1A..12A, 1B..12B)."""

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(24, embed_dim)

    def forward(self, key_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(key_indices)

    def compatibility(self, key_a: torch.Tensor, key_b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity in embedding space."""
        emb_a = nn.functional.normalize(self.embedding(key_a), dim=-1)
        emb_b = nn.functional.normalize(self.embedding(key_b), dim=-1)
        return (emb_a * emb_b).sum(dim=-1)
