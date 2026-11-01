"""Music feature tokenizer: vector quantization codebook and residual VQ."""
from __future__ import annotations

import numpy as np


class MusicCodebook:
    def __init__(self, n_codes: int = 512, dim: int = 128, seed: int = 42) -> None:
        self.n_codes = n_codes
        self.dim = dim
        rng = np.random.RandomState(seed)
        self.embeddings = rng.randn(n_codes, dim).astype(np.float64)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings /= np.maximum(norms, 1e-8)
        self._usage_counts = np.zeros(n_codes)

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        pad_x = np.zeros((x.shape[0], self.dim))
        m = min(x.shape[-1], self.dim)
        pad_x[:, :m] = x[:, :m]

        dists = (
            np.sum(pad_x ** 2, axis=1, keepdims=True)
            - 2 * pad_x @ self.embeddings.T
            + np.sum(self.embeddings ** 2, axis=1, keepdims=True).T
        )
        indices = np.argmin(dists, axis=1)
        quantized = self.embeddings[indices]
        self._usage_counts[indices] += 1
        commitment_loss = float(np.mean((pad_x - quantized) ** 2))
        return quantized, indices, commitment_loss

    def encode(self, x: np.ndarray) -> np.ndarray:
        _, indices, _ = self.quantize(x)
        return indices

    def decode(self, indices: np.ndarray) -> np.ndarray:
        return self.embeddings[indices]

    def update_codebook(self, x: np.ndarray, indices: np.ndarray, decay: float = 0.99) -> None:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        pad_x = np.zeros((x.shape[0], self.dim))
        m = min(x.shape[-1], self.dim)
        pad_x[:, :m] = x[:, :m]
        for i in range(self.n_codes):
            mask = indices == i
            if np.any(mask):
                centroid = np.mean(pad_x[mask], axis=0)
                self.embeddings[i] = decay * self.embeddings[i] + (1 - decay) * centroid

    def perplexity(self) -> float:
        total = self._usage_counts.sum()
        if total == 0:
            return 0.0
        probs = self._usage_counts / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(np.exp(entropy))


class MusicTokenizer:
    def __init__(self, n_codes: int = 512, dim: int = 128, n_quantizers: int = 4, seed: int = 42) -> None:
        self.n_quantizers = n_quantizers
        self.codebooks = [MusicCodebook(n_codes, dim, seed + i * 100) for i in range(n_quantizers)]
        self.dim = dim

    @property
    def vocab_size(self) -> int:
        return self.codebooks[0].n_codes

    def tokenize(self, features: np.ndarray) -> list[np.ndarray]:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        residual = features.copy()
        if residual.shape[-1] < self.dim:
            pad = np.zeros((residual.shape[0], self.dim))
            pad[:, :residual.shape[-1]] = residual
            residual = pad
        elif residual.shape[-1] > self.dim:
            residual = residual[:, :self.dim]

        token_ids: list[np.ndarray] = []
        for cb in self.codebooks:
            quantized, indices, _ = cb.quantize(residual)
            token_ids.append(indices)
            residual = residual - quantized
        return token_ids

    def detokenize(self, token_ids: list[np.ndarray]) -> np.ndarray:
        reconstructed = np.zeros((len(token_ids[0]), self.dim))
        for cb, ids in zip(self.codebooks, token_ids):
            reconstructed += cb.decode(ids)
        return reconstructed

    def commitment_loss(self, features: np.ndarray) -> float:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        residual = features.copy()
        if residual.shape[-1] != self.dim:
            pad = np.zeros((residual.shape[0], self.dim))
            m = min(residual.shape[-1], self.dim)
            pad[:, :m] = residual[:, :m]
            residual = pad

        total_loss = 0.0
        for cb in self.codebooks:
            quantized, _, loss = cb.quantize(residual)
            total_loss += loss
            residual = residual - quantized
        return total_loss / self.n_quantizers


def tokens_to_sequence(token_ids_list: list[np.ndarray]) -> np.ndarray:
    n_levels = len(token_ids_list)
    n_steps = len(token_ids_list[0])
    flat = np.zeros(n_levels * n_steps, dtype=int)
    for step in range(n_steps):
        for level in range(n_levels):
            flat[step * n_levels + level] = token_ids_list[level][step]
    return flat
