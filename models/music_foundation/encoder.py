"""Music Foundation Encoder: two-tower architecture for multi-modal music representation.

Tower 1 (AudioEncoder): spectral/temporal features → latent audio embedding
Tower 2 (SemanticEncoder): metadata/lyrics/tags → latent semantic embedding
GovernedBridge: aligns both towers into a joint embedding space with
    controllable mixing via a governance parameter alpha in [0,1].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EncoderConfig:
    audio_dim: int = 128
    semantic_dim: int = 64
    hidden_dim: int = 256
    latent_dim: int = 128
    n_heads: int = 4
    dropout: float = 0.1
    alpha: float = 0.5


class LinearLayer:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.randn(in_dim, out_dim).astype(np.float64) * scale
        self.b = np.zeros(out_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    def load_params(self, params: dict[str, np.ndarray]) -> None:
        self.W = params["W"]
        self.b = params["b"]


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class MultiHeadAttention:
    def __init__(self, dim: int, n_heads: int, seed: int = 0) -> None:
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.Wq = LinearLayer(dim, dim, seed)
        self.Wk = LinearLayer(dim, dim, seed + 1)
        self.Wv = LinearLayer(dim, dim, seed + 2)
        self.Wo = LinearLayer(dim, dim, seed + 3)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n, d = x.shape
        h = self.n_heads
        hd = self.head_dim

        Q = self.Wq.forward(x).reshape(n, h, hd)
        K = self.Wk.forward(x).reshape(n, h, hd)
        V = self.Wv.forward(x).reshape(n, h, hd)

        scores = np.einsum("nhd,mhd->hnm", Q, K) / np.sqrt(hd)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-12)

        context = np.einsum("hnm,mhd->nhd", attn, V).reshape(n, d)
        return self.Wo.forward(context)


class AudioEncoder:
    def __init__(self, config: EncoderConfig, seed: int = 42) -> None:
        self.proj = LinearLayer(config.audio_dim, config.hidden_dim, seed)
        self.attn = MultiHeadAttention(config.hidden_dim, config.n_heads, seed + 10)
        self.ff1 = LinearLayer(config.hidden_dim, config.hidden_dim, seed + 20)
        self.ff2 = LinearLayer(config.hidden_dim, config.latent_dim, seed + 30)
        self.out_proj = LinearLayer(config.hidden_dim, config.latent_dim, seed + 40)
        self.config = config

    def encode(self, audio_features: np.ndarray) -> np.ndarray:
        if audio_features.ndim == 1:
            audio_features = audio_features.reshape(1, -1)

        if audio_features.shape[-1] != self.config.audio_dim:
            pad_or_trunc = np.zeros((audio_features.shape[0], self.config.audio_dim))
            m = min(audio_features.shape[-1], self.config.audio_dim)
            pad_or_trunc[:, :m] = audio_features[:, :m]
            audio_features = pad_or_trunc

        h = gelu(self.proj.forward(audio_features))
        h = layer_norm(h + self.attn.forward(h))
        ff_out = gelu(self.ff2.forward(gelu(self.ff1.forward(h))))
        h = layer_norm(self.out_proj.forward(h) + ff_out)
        return h

    def params(self) -> dict[str, np.ndarray]:
        p = {}
        for name, layer in [("proj", self.proj), ("ff1", self.ff1), ("ff2", self.ff2)]:
            for k, v in layer.params.items():
                p[f"audio_{name}_{k}"] = v
        return p


class SemanticEncoder:
    def __init__(self, config: EncoderConfig, seed: int = 100) -> None:
        self.proj = LinearLayer(config.semantic_dim, config.hidden_dim, seed)
        self.ff1 = LinearLayer(config.hidden_dim, config.hidden_dim, seed + 10)
        self.ff2 = LinearLayer(config.hidden_dim, config.latent_dim, seed + 20)
        self.out_proj = LinearLayer(config.hidden_dim, config.latent_dim, seed + 30)
        self.config = config

    def encode(self, semantic_features: np.ndarray) -> np.ndarray:
        if semantic_features.ndim == 1:
            semantic_features = semantic_features.reshape(1, -1)

        if semantic_features.shape[-1] != self.config.semantic_dim:
            pad_or_trunc = np.zeros((semantic_features.shape[0], self.config.semantic_dim))
            m = min(semantic_features.shape[-1], self.config.semantic_dim)
            pad_or_trunc[:, :m] = semantic_features[:, :m]
            semantic_features = pad_or_trunc

        h = gelu(self.proj.forward(semantic_features))
        ff_out = gelu(self.ff2.forward(gelu(self.ff1.forward(h))))
        h = layer_norm(self.out_proj.forward(h) + ff_out)
        return h

    def params(self) -> dict[str, np.ndarray]:
        p = {}
        for name, layer in [("proj", self.proj), ("ff1", self.ff1), ("ff2", self.ff2)]:
            for k, v in layer.params.items():
                p[f"semantic_{name}_{k}"] = v
        return p


class GovernedBridge:
    def __init__(self, config: EncoderConfig) -> None:
        self.alpha = config.alpha
        self.latent_dim = config.latent_dim

    def fuse(self, audio_emb: np.ndarray, semantic_emb: np.ndarray) -> np.ndarray:
        a = np.clip(self.alpha, 0, 1)
        combined = a * audio_emb + (1 - a) * semantic_emb
        return layer_norm(combined)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = np.clip(alpha, 0, 1)


class MusicFoundationModel:
    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        self.audio_encoder = AudioEncoder(self.config)
        self.semantic_encoder = SemanticEncoder(self.config)
        self.bridge = GovernedBridge(self.config)

    def encode_audio(self, audio_features: np.ndarray) -> np.ndarray:
        return self.audio_encoder.encode(audio_features)

    def encode_semantic(self, semantic_features: np.ndarray) -> np.ndarray:
        return self.semantic_encoder.encode(semantic_features)

    def encode(self, audio_features: np.ndarray, semantic_features: np.ndarray) -> np.ndarray:
        audio_emb = self.encode_audio(audio_features)
        semantic_emb = self.encode_semantic(semantic_features)
        return self.bridge.fuse(audio_emb, semantic_emb)

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        a = emb_a.flatten()
        b = emb_b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def params(self) -> dict[str, np.ndarray]:
        p = self.audio_encoder.params()
        p.update(self.semantic_encoder.params())
        return p
