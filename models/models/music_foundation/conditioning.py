"""Conditioning system for controlled music generation.

Encodes genre, mood, key, tempo, and free-text tags into conditioning
vectors that modulate the generation process via FiLM (feature-wise
linear modulation) or cross-attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .encoder import LinearLayer, gelu, layer_norm


GENRES = [
    "classical", "jazz", "rock", "pop", "electronic", "hip-hop",
    "r&b", "country", "folk", "metal", "blues", "reggae", "latin",
    "world", "ambient", "soul", "funk", "punk", "indie", "other",
]

MOODS = [
    "happy", "sad", "energetic", "calm", "aggressive", "romantic",
    "melancholic", "uplifting", "dark", "dreamy", "tense", "peaceful",
    "nostalgic", "euphoric", "mysterious", "playful",
]

KEYS = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]

MODES = ["major", "minor"]


@dataclass
class ConditioningConfig:
    genre_dim: int = 32
    mood_dim: int = 32
    key_dim: int = 16
    tempo_dim: int = 16
    tag_dim: int = 32
    cond_dim: int = 128
    n_genres: int = 20
    n_moods: int = 16
    n_keys: int = 12
    n_modes: int = 2
    max_tags: int = 10
    tag_vocab_size: int = 1000


class GenreEncoder:
    def __init__(self, n_genres: int, dim: int, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.embeddings = rng.randn(n_genres, dim).astype(np.float64) * 0.02

    def encode(self, genre_ids: np.ndarray) -> np.ndarray:
        ids = np.atleast_1d(genre_ids).astype(int)
        embs = self.embeddings[ids]
        return embs.mean(axis=0) if embs.ndim > 1 else embs.flatten()

    def encode_name(self, name: str) -> np.ndarray:
        name_lower = name.lower().strip()
        idx = GENRES.index(name_lower) if name_lower in GENRES else len(GENRES) - 1
        return self.encode(np.array([idx]))


class MoodEncoder:
    def __init__(self, n_moods: int, dim: int, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.embeddings = rng.randn(n_moods, dim).astype(np.float64) * 0.02

    def encode(self, mood_ids: np.ndarray) -> np.ndarray:
        ids = np.atleast_1d(mood_ids).astype(int)
        embs = self.embeddings[ids]
        return embs.mean(axis=0) if embs.ndim > 1 else embs.flatten()

    def encode_name(self, name: str) -> np.ndarray:
        name_lower = name.lower().strip()
        idx = MOODS.index(name_lower) if name_lower in MOODS else 0
        return self.encode(np.array([idx]))

    def encode_vad(self, valence: float, arousal: float, dominance: float) -> np.ndarray:
        vad = np.array([valence, arousal, dominance])
        proj = np.zeros(self.embeddings.shape[1])
        proj[:3] = vad
        return proj


class KeyEncoder:
    def __init__(self, n_keys: int, n_modes: int, dim: int, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.key_embeddings = rng.randn(n_keys, dim).astype(np.float64) * 0.02
        self.mode_embeddings = rng.randn(n_modes, dim).astype(np.float64) * 0.02

    def encode(self, key_idx: int, mode_idx: int = 0) -> np.ndarray:
        return self.key_embeddings[key_idx] + self.mode_embeddings[mode_idx]

    def encode_name(self, key_name: str) -> np.ndarray:
        parts = key_name.strip().split()
        root = parts[0].upper() if parts else "C"
        mode = parts[1].lower() if len(parts) > 1 else "major"
        key_idx = KEYS.index(root) if root in KEYS else 0
        mode_idx = MODES.index(mode) if mode in MODES else 0
        return self.encode(key_idx, mode_idx)


class TempoEncoder:
    def __init__(self, dim: int, seed: int = 0) -> None:
        self.proj = LinearLayer(1, dim, seed)

    def encode(self, bpm: float) -> np.ndarray:
        normalized = np.array([[np.clip(bpm, 20.0, 300.0) / 300.0]])
        return self.proj.forward(normalized).flatten()


class TagEncoder:
    def __init__(self, vocab_size: int, dim: int, max_tags: int, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.embeddings = rng.randn(vocab_size, dim).astype(np.float64) * 0.02
        self.max_tags = max_tags

    def encode(self, tag_ids: np.ndarray) -> np.ndarray:
        ids = np.atleast_1d(tag_ids).astype(int)
        ids = ids[:self.max_tags]
        valid = ids[(ids >= 0) & (ids < len(self.embeddings))]
        if len(valid) == 0:
            return np.zeros(self.embeddings.shape[1])
        return self.embeddings[valid].mean(axis=0)

    def encode_hash(self, tags: list[str]) -> np.ndarray:
        ids = np.array([hash(t.lower().strip()) % len(self.embeddings) for t in tags[:self.max_tags]])
        return self.encode(ids)


class FiLMLayer:
    """Feature-wise Linear Modulation: gamma * x + beta."""

    def __init__(self, cond_dim: int, feature_dim: int, seed: int = 0) -> None:
        self.gamma_proj = LinearLayer(cond_dim, feature_dim, seed)
        self.beta_proj = LinearLayer(cond_dim, feature_dim, seed + 1)

    def modulate(self, x: np.ndarray, cond: np.ndarray) -> np.ndarray:
        if cond.ndim == 1:
            cond = cond.reshape(1, -1)
        gamma = 1.0 + self.gamma_proj.forward(cond)
        beta = self.beta_proj.forward(cond)
        if x.ndim > cond.ndim:
            gamma = gamma.reshape(*([1] * (x.ndim - gamma.ndim)), *gamma.shape)
            beta = beta.reshape(*([1] * (x.ndim - beta.ndim)), *beta.shape)
        return gamma * x + beta


class ConditioningModule:
    """Aggregates all condition encoders into a single conditioning vector."""

    def __init__(self, config: ConditioningConfig | None = None, seed: int = 0) -> None:
        self.config = config or ConditioningConfig()
        c = self.config
        self.genre_enc = GenreEncoder(c.n_genres, c.genre_dim, seed)
        self.mood_enc = MoodEncoder(c.n_moods, c.mood_dim, seed + 100)
        self.key_enc = KeyEncoder(c.n_keys, c.n_modes, c.key_dim, seed + 200)
        self.tempo_enc = TempoEncoder(c.tempo_dim, seed + 300)
        self.tag_enc = TagEncoder(c.tag_vocab_size, c.tag_dim, c.max_tags, seed + 400)

        total_dim = c.genre_dim + c.mood_dim + c.key_dim + c.tempo_dim + c.tag_dim
        self.proj = LinearLayer(total_dim, c.cond_dim, seed + 500)

    def encode(
        self,
        genre: str | int | np.ndarray | None = None,
        mood: str | np.ndarray | None = None,
        key: str | None = None,
        tempo_bpm: float | None = None,
        tags: list[str] | None = None,
        vad: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        if isinstance(genre, str):
            g = self.genre_enc.encode_name(genre)
        elif isinstance(genre, (int, np.integer)):
            g = self.genre_enc.encode(np.array([int(genre)]))
        elif isinstance(genre, np.ndarray):
            g = self.genre_enc.encode(genre)
        else:
            g = np.zeros(self.config.genre_dim)

        if isinstance(mood, str):
            m = self.mood_enc.encode_name(mood)
        elif vad is not None:
            m = self.mood_enc.encode_vad(*vad)
        elif isinstance(mood, np.ndarray):
            m = self.mood_enc.encode(mood)
        else:
            m = np.zeros(self.config.mood_dim)

        k = self.key_enc.encode_name(key) if key else np.zeros(self.config.key_dim)
        t = self.tempo_enc.encode(tempo_bpm) if tempo_bpm else np.zeros(self.config.tempo_dim)
        tg = self.tag_enc.encode_hash(tags) if tags else np.zeros(self.config.tag_dim)

        concat = np.concatenate([g, m, k, t, tg])
        return self.proj.forward(concat.reshape(1, -1)).flatten()

    def null_conditioning(self) -> np.ndarray:
        return np.zeros(self.config.cond_dim)
