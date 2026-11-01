"""Positional and temporal encodings for music sequences."""
from __future__ import annotations

import numpy as np


def sinusoidal_encoding(seq_len: int, dim: int) -> np.ndarray:
    pe = np.zeros((seq_len, dim))
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    pe[:, 0::2] = np.sin(pos * div[:dim // 2 + dim % 2])
    pe[:, 1::2] = np.cos(pos * div[:dim // 2])
    return pe


def learned_positional_encoding(seq_len: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(seq_len, dim).astype(np.float64) * 0.02


class RotaryPositionalEncoding:
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        self.dim = dim
        half = dim // 2
        freqs = 1.0 / (base ** (np.arange(0, half) / half))
        positions = np.arange(max_seq_len)
        angles = np.outer(positions, freqs)
        self._cos = np.cos(angles)
        self._sin = np.sin(angles)

    def apply(self, x: np.ndarray, seq_pos: np.ndarray | None = None) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n, d = x.shape
        half = d // 2
        if seq_pos is None:
            seq_pos = np.arange(n)
        cos_vals = self._cos[seq_pos, :half]
        sin_vals = self._sin[seq_pos, :half]
        x1 = x[:, :half]
        x2 = x[:, half:2 * half]
        out = x.copy()
        out[:, :half] = x1 * cos_vals - x2 * sin_vals
        out[:, half:2 * half] = x2 * cos_vals + x1 * sin_vals
        return out


class TemporalEncoding:
    def __init__(self, dim: int, max_bars: int = 256, max_beats: int = 16) -> None:
        self.dim = dim
        self.bar_emb = sinusoidal_encoding(max_bars, dim // 2)
        beat_dim = dim - dim // 2
        self.beat_emb = sinusoidal_encoding(max_beats, beat_dim)

    def encode(self, bar_positions: np.ndarray, beat_positions: np.ndarray) -> np.ndarray:
        n = len(bar_positions)
        bar_idx = np.clip(bar_positions.astype(int), 0, self.bar_emb.shape[0] - 1)
        beat_idx = np.clip(beat_positions.astype(int), 0, self.beat_emb.shape[0] - 1)
        return np.concatenate([self.bar_emb[bar_idx], self.beat_emb[beat_idx]], axis=-1)


class RelativePositionBias:
    def __init__(self, n_heads: int, max_distance: int = 128, seed: int = 0) -> None:
        self.n_heads = n_heads
        self.max_distance = max_distance
        rng = np.random.RandomState(seed)
        self.bias_table = rng.randn(n_heads, 2 * max_distance + 1).astype(np.float64) * 0.02

    def get_bias(self, seq_len: int) -> np.ndarray:
        positions = np.arange(seq_len)
        rel = positions[:, None] - positions[None, :]
        rel = np.clip(rel, -self.max_distance, self.max_distance) + self.max_distance
        return self.bias_table[:, rel]
