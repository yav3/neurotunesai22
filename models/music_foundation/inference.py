"""Unified music inference pipeline.

Wires raw audio → FeaturePipeline → DomainIntegrator → MusicFoundationModel
into a single forward pass that produces embeddings and task predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .encoder import MusicFoundationModel, EncoderConfig, LinearLayer, gelu, layer_norm
from ..foundation.domain_integration import (
    DomainIntegrator, DomainAnalysis, DOMAIN_FEATURE_DIM,
    cognition_to_array, emotion_to_array, theory_to_array,
    acoustics_to_array, structure_to_array,
)
from ..music_pipeline.feature_pipeline import FeaturePipeline, PipelineConfig


@dataclass
class InferenceConfig:
    sr: int = 22050
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    n_genres: int = 10
    n_moods: int = 8
    enable_domain: bool = True


@dataclass
class InferenceResult:
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(0))
    audio_embedding: np.ndarray = field(default_factory=lambda: np.zeros(0))
    semantic_embedding: np.ndarray = field(default_factory=lambda: np.zeros(0))
    genre_logits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mood_logits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vad: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tempo_bpm: float = 0.0
    key_estimate: str = ""
    domain_analysis: DomainAnalysis | None = None
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    metadata: dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: "InferenceResult") -> float:
        a = self.embedding.flatten()
        b = other.embedding.flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


class GenreHead:
    def __init__(self, latent_dim: int, n_genres: int, seed: int = 200) -> None:
        self.fc1 = LinearLayer(latent_dim, latent_dim // 2, seed)
        self.fc2 = LinearLayer(latent_dim // 2, n_genres, seed + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = gelu(self.fc1.forward(x))
        return self.fc2.forward(h)


class MoodHead:
    def __init__(self, latent_dim: int, n_moods: int, seed: int = 300) -> None:
        self.fc1 = LinearLayer(latent_dim, latent_dim // 2, seed)
        self.fc2 = LinearLayer(latent_dim // 2, n_moods, seed + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = gelu(self.fc1.forward(x))
        return self.fc2.forward(h)


class VADHead:
    def __init__(self, latent_dim: int, seed: int = 400) -> None:
        self.fc = LinearLayer(latent_dim, 3, seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raw = self.fc.forward(x)
        return 1.0 / (1.0 + np.exp(-raw))


class MusicInferencePipeline:
    """End-to-end: raw audio → embeddings + predictions."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self.feature_pipeline = FeaturePipeline(self.config.pipeline)
        self.domain_integrator = DomainIntegrator(sr=self.config.sr) if self.config.enable_domain else None
        self.foundation = MusicFoundationModel(self.config.encoder)
        self.genre_head = GenreHead(self.config.encoder.latent_dim, self.config.n_genres)
        self.mood_head = MoodHead(self.config.encoder.latent_dim, self.config.n_moods)
        self.vad_head = VADHead(self.config.encoder.latent_dim)

    def _audio_to_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        feat_result = self.feature_pipeline.extract(audio, sr)
        vec = feat_result.to_vector()
        if len(vec) == 0:
            vec = np.zeros(self.config.encoder.audio_dim)
        return vec

    def _domain_to_semantic(self, domain: DomainAnalysis) -> np.ndarray:
        parts = []
        parts.append(cognition_to_array(domain.cognition))
        parts.append(emotion_to_array(domain.emotion))
        parts.append(theory_to_array(domain.theory))
        parts.append(acoustics_to_array(domain.acoustics))
        parts.append(structure_to_array(domain.structure))
        return np.concatenate(parts).astype(np.float64)

    def infer(self, audio: np.ndarray, sr: int | None = None) -> InferenceResult:
        sr = sr or self.config.sr
        result = InferenceResult()
        result.metadata["sr"] = sr
        result.metadata["n_samples"] = len(audio)

        if audio.ndim == 2:
            audio = audio.mean(axis=1) if audio.shape[0] > audio.shape[1] else audio.mean(axis=0)
        audio = audio.astype(np.float64)

        audio_features = self._audio_to_features(audio, sr)
        result.feature_vector = audio_features

        domain = None
        semantic_features = np.zeros(self.config.encoder.semantic_dim)
        if self.domain_integrator is not None:
            domain = self.domain_integrator.analyze(audio.astype(np.float32), sr)
            result.domain_analysis = domain
            sem = self._domain_to_semantic(domain)
            semantic_features[:min(len(sem), len(semantic_features))] = sem[:min(len(sem), len(semantic_features))]

            result.tempo_bpm = domain.theory.tempo_bpm
            KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            result.key_estimate = domain.theory.key

        audio_emb = self.foundation.encode_audio(audio_features)
        semantic_emb = self.foundation.encode_semantic(semantic_features)
        fused = self.foundation.bridge.fuse(audio_emb, semantic_emb)

        if fused.ndim == 2:
            emb = fused.mean(axis=0)
        else:
            emb = fused.flatten()

        result.embedding = emb
        result.audio_embedding = audio_emb.mean(axis=0) if audio_emb.ndim == 2 else audio_emb.flatten()
        result.semantic_embedding = semantic_emb.mean(axis=0) if semantic_emb.ndim == 2 else semantic_emb.flatten()

        emb_2d = emb.reshape(1, -1)
        result.genre_logits = self.genre_head.forward(emb_2d).flatten()
        result.mood_logits = self.mood_head.forward(emb_2d).flatten()
        result.vad = self.vad_head.forward(emb_2d).flatten()

        return result

    def infer_batch(self, audios: list[np.ndarray], sr: int | None = None) -> list[InferenceResult]:
        return [self.infer(a, sr) for a in audios]

    def embedding_matrix(self, audios: list[np.ndarray], sr: int | None = None) -> np.ndarray:
        results = self.infer_batch(audios, sr)
        return np.vstack([r.embedding.reshape(1, -1) for r in results])
