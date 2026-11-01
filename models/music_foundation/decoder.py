"""Decoder: masked self-attention transformer decoder and generation heads."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .encoder import LinearLayer, gelu, layer_norm, MultiHeadAttention


@dataclass
class DecoderConfig:
    latent_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2


class MaskedMultiHeadAttention:
    def __init__(self, dim: int, n_heads: int, seed: int = 0) -> None:
        self.attn = MultiHeadAttention(dim, n_heads, seed)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n, d = x.shape
        h = self.n_heads
        hd = self.head_dim

        Q = self.attn.Wq.forward(x).reshape(n, h, hd)
        K = self.attn.Wk.forward(x).reshape(n, h, hd)
        V = self.attn.Wv.forward(x).reshape(n, h, hd)

        scores = np.einsum("nhd,mhd->hnm", Q, K) / np.sqrt(hd)
        causal_mask = np.triu(np.full((n, n), -1e9), k=1)
        scores = scores + causal_mask[None, :, :]
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-12)

        context = np.einsum("hnm,mhd->nhd", attn, V).reshape(n, d)
        return self.attn.Wo.forward(context)


class CrossAttention:
    def __init__(self, dim: int, n_heads: int, seed: int = 0) -> None:
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.Wq = LinearLayer(dim, dim, seed)
        self.Wk = LinearLayer(dim, dim, seed + 1)
        self.Wv = LinearLayer(dim, dim, seed + 2)
        self.Wo = LinearLayer(dim, dim, seed + 3)

    def forward(self, x: np.ndarray, context: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if context.ndim == 1:
            context = context.reshape(1, -1)
        n, d = x.shape
        m = context.shape[0]
        h = self.n_heads
        hd = self.head_dim

        Q = self.Wq.forward(x).reshape(n, h, hd)
        K = self.Wk.forward(context).reshape(m, h, hd)
        V = self.Wv.forward(context).reshape(m, h, hd)

        scores = np.einsum("nhd,mhd->hnm", Q, K) / np.sqrt(hd)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-12)

        out = np.einsum("hnm,mhd->nhd", attn, V).reshape(n, d)
        return self.Wo.forward(out)


class TransformerDecoderLayer:
    def __init__(self, dim: int, n_heads: int, seed: int = 0) -> None:
        self.self_attn = MaskedMultiHeadAttention(dim, n_heads, seed)
        self.cross_attn = CrossAttention(dim, n_heads, seed + 100)
        self.ff1 = LinearLayer(dim, dim * 4, seed + 200)
        self.ff2 = LinearLayer(dim * 4, dim, seed + 300)

    def forward(self, x: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
        h = layer_norm(x + self.self_attn.forward(x))
        if context is not None:
            h = layer_norm(h + self.cross_attn.forward(h, context))
        ff_out = self.ff2.forward(gelu(self.ff1.forward(h)))
        return layer_norm(h + ff_out)


class MusicDecoder:
    def __init__(self, config: DecoderConfig, seed: int = 500) -> None:
        self.config = config
        self.input_proj = LinearLayer(config.latent_dim, config.hidden_dim, seed)
        self.layers = [
            TransformerDecoderLayer(config.hidden_dim, config.n_heads, seed + i * 400)
            for i in range(config.n_layers)
        ]
        self.output_proj = LinearLayer(config.hidden_dim, config.output_dim, seed + 900)

    def decode(self, z: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
        if z.ndim == 1:
            z = z.reshape(1, -1)
        h = gelu(self.input_proj.forward(z))
        for layer in self.layers:
            h = layer.forward(h, context)
        return self.output_proj.forward(h)

    def generate(
        self,
        z: np.ndarray,
        max_steps: int = 100,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if z.ndim == 1:
            z = z.reshape(1, -1)
        sequence = [z]
        current = z
        for _ in range(max_steps - 1):
            cat = np.concatenate(sequence, axis=0)
            h = gelu(self.input_proj.forward(cat))
            for layer in self.layers:
                h = layer.forward(h)
            out = self.output_proj.forward(h)
            last = out[-1:]
            if temperature != 1.0:
                last = last / temperature
            sequence.append(last)
            current = last
        return np.concatenate(sequence, axis=0)


class ReconstructionHead:
    def __init__(self, latent_dim: int, output_dim: int, seed: int = 0) -> None:
        self.proj = LinearLayer(latent_dim, output_dim, seed)

    def forward(self, z: np.ndarray) -> np.ndarray:
        return self.proj.forward(z)

    def loss(self, z: np.ndarray, target: np.ndarray) -> float:
        pred = self.forward(z)
        if pred.shape != target.shape:
            m = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :m]
            target = target[..., :m]
        return float(np.mean((pred - target) ** 2))


class GenerationHead:
    def __init__(self, latent_dim: int, vocab_size: int, seed: int = 0) -> None:
        self.proj = LinearLayer(latent_dim, vocab_size, seed)
        self.vocab_size = vocab_size

    def logits(self, z: np.ndarray) -> np.ndarray:
        return self.proj.forward(z)

    def predict(self, z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        raw = self.logits(z) / max(temperature, 1e-8)
        shifted = raw - raw.max(axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
        return probs

    def sample(self, z: np.ndarray, temperature: float = 1.0, seed: int | None = None) -> np.ndarray:
        rng = np.random.RandomState(seed)
        probs = self.predict(z, temperature)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        indices = np.array([rng.choice(self.vocab_size, p=p) for p in probs])
        return indices

    def loss(self, z: np.ndarray, target_indices: np.ndarray) -> float:
        logits = self.logits(z)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        n = len(target_indices)
        return float(-np.mean(log_probs[np.arange(n), target_indices.astype(int)]))
