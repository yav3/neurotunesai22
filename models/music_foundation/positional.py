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


# ---------------------------------------------------------------------------
# ALiBi — Attention with Linear Biases (Press et al., 2022)
# ---------------------------------------------------------------------------

def alibi_slopes(n_heads: int) -> np.ndarray:
    closest_pow2 = 2 ** int(np.floor(np.log2(n_heads)))
    base = 2.0 ** (-(2.0 ** -(np.log2(closest_pow2) - 3)))
    slopes = base ** np.arange(1, closest_pow2 + 1)
    if closest_pow2 < n_heads:
        extra_base = 2.0 ** (-(2.0 ** -(np.log2(2 * closest_pow2) - 3)))
        extra = extra_base ** np.arange(1, 2 * (n_heads - closest_pow2) + 1, 2)
        slopes = np.concatenate([slopes, extra])
    return slopes[:n_heads]


def alibi_bias(seq_len: int, n_heads: int) -> np.ndarray:
    slopes = alibi_slopes(n_heads)
    positions = np.arange(seq_len)
    rel = positions[:, None] - positions[None, :]
    return slopes[:, None, None] * rel[None, :, :]


# ---------------------------------------------------------------------------
# Conditional Positional Encoding (Wu et al., 2021 — CvT, 1-D adaptation)
# ---------------------------------------------------------------------------

class ConditionalPositionalEncoding:
    """Depth-wise 1-D conv generating position info from local context."""

    def __init__(self, dim: int, kernel_size: int = 3, seed: int = 0) -> None:
        self.dim = dim
        self.kernel_size = kernel_size
        rng = np.random.RandomState(seed)
        self.weight = rng.randn(dim, kernel_size).astype(np.float64) * 0.02
        self.bias = np.zeros(dim, dtype=np.float64)

    def encode(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        seq_len, dim = x.shape
        pad = self.kernel_size // 2
        x_padded = np.pad(x, ((pad, pad), (0, 0)), mode='reflect')
        out = np.zeros_like(x)
        for d in range(dim):
            for t in range(seq_len):
                out[t, d] = np.dot(x_padded[t:t + self.kernel_size, d], self.weight[d]) + self.bias[d]
        return x + out


# ---------------------------------------------------------------------------
# Musical Meter Encoding — hierarchical bar/beat/sub-beat/tick
# ---------------------------------------------------------------------------

class MusicalMeterEncoding:
    """Hierarchical positional encoding for musical metre structure.

    Sum of per-level sinusoidal tables (bar, beat, sub-beat, tick)
    with learnable per-level gain.
    """

    def __init__(
        self,
        dim: int,
        max_bars: int = 512,
        max_beats: int = 16,
        max_sub_beats: int = 4,
        max_ticks: int = 480,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        quarter = dim // 4
        rem = dim - 3 * quarter
        dims = [quarter, quarter, quarter, rem]
        self.tables = [
            sinusoidal_encoding(m, d)
            for m, d in zip([max_bars, max_beats, max_sub_beats, max_ticks], dims)
        ]
        self.maxes = [max_bars, max_beats, max_sub_beats, max_ticks]
        rng = np.random.RandomState(seed)
        self.gains = np.ones(4, dtype=np.float64)

    def encode(
        self,
        bars: np.ndarray,
        beats: np.ndarray,
        sub_beats: np.ndarray,
        ticks: np.ndarray,
    ) -> np.ndarray:
        indices = [bars, beats, sub_beats, ticks]
        parts = []
        for i, (idx, table) in enumerate(zip(indices, self.tables)):
            clipped = np.clip(idx.astype(int), 0, table.shape[0] - 1)
            parts.append(self.gains[i] * table[clipped])
        return np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# Pitch-Class Positional Encoding — chroma + octave decomposition
# ---------------------------------------------------------------------------

class PitchPositionalEncoding:
    """Encode MIDI pitch as chroma (mod 12) + octave components."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        half = dim // 2
        self.chroma_table = sinusoidal_encoding(12, half)
        oct_dim = dim - half
        self.octave_table = sinusoidal_encoding(11, oct_dim)  # MIDI 0-127 → octaves -1..9

    def encode(self, midi_pitches: np.ndarray) -> np.ndarray:
        midi = np.asarray(midi_pitches, dtype=int)
        chroma = midi % 12
        octave = np.clip(midi // 12, 0, 10)
        return np.concatenate([
            self.chroma_table[chroma],
            self.octave_table[octave],
        ], axis=-1)


# ---------------------------------------------------------------------------
# Multi-Scale Temporal Encoding
# ---------------------------------------------------------------------------

class MultiScaleTemporalEncoding:
    """Combines multiple time scales with learnable mixing weights."""

    def __init__(self, dim: int, n_scales: int = 4, max_len: int = 8192, seed: int = 0) -> None:
        self.dim = dim
        self.n_scales = n_scales
        rng = np.random.RandomState(seed)
        self.tables = []
        for s in range(n_scales):
            base = 10000.0 * (4.0 ** s)
            pe = np.zeros((max_len, dim))
            pos = np.arange(max_len)[:, None]
            div = np.exp(np.arange(0, dim, 2) * -(np.log(base) / dim))
            pe[:, 0::2] = np.sin(pos * div[:dim // 2 + dim % 2])
            pe[:, 1::2] = np.cos(pos * div[:dim // 2])
            self.tables.append(pe)
        self.mix_weights = np.ones(n_scales, dtype=np.float64) / n_scales

    def encode(self, positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=int)
        result = np.zeros((len(pos), self.dim))
        for s in range(self.n_scales):
            idx = np.clip(pos, 0, self.tables[s].shape[0] - 1)
            result += self.mix_weights[s] * self.tables[s][idx]
        return result


# ---------------------------------------------------------------------------
# Log-Bucketed Relative Position Bias (Raffel et al., 2020 — T5 style)
# ---------------------------------------------------------------------------

def log_relative_position_buckets(
    rel_positions: np.ndarray,
    n_buckets: int = 32,
    max_distance: int = 128,
) -> np.ndarray:
    rel = np.asarray(rel_positions, dtype=np.float64)
    is_neg = (rel < 0).astype(np.int32)
    rel = np.abs(rel)
    half_buckets = n_buckets // 2
    max_exact = half_buckets // 2
    is_small = rel < max_exact
    bucket_small = rel.astype(np.int32)
    log_ratio = np.log(np.maximum(rel, 1.0) / max_exact)
    log_max = np.log(float(max_distance) / max_exact)
    bucket_large = max_exact + (log_ratio / (log_max + 1e-12) * (half_buckets - max_exact)).astype(np.int32)
    bucket_large = np.minimum(bucket_large, half_buckets - 1)
    buckets = np.where(is_small, bucket_small, bucket_large)
    buckets = buckets + is_neg * half_buckets
    return buckets


class LogRelativePositionBias:
    """T5-style log-bucketed relative position bias."""

    def __init__(self, n_heads: int, n_buckets: int = 32, max_distance: int = 128, seed: int = 0) -> None:
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        rng = np.random.RandomState(seed)
        self.bias_table = rng.randn(n_heads, n_buckets).astype(np.float64) * 0.02

    def get_bias(self, seq_len: int) -> np.ndarray:
        positions = np.arange(seq_len)
        rel = positions[:, None] - positions[None, :]
        buckets = log_relative_position_buckets(rel, self.n_buckets, self.max_distance)
        return self.bias_table[:, buckets]


# ---------------------------------------------------------------------------
# NTK-Aware RoPE — context-length extension (Reddit/Code Llama style)
# ---------------------------------------------------------------------------

class NTKAwareRoPE:
    """Rotary PE with NTK-aware frequency scaling for context extension."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scale_factor: float = 1.0,
    ) -> None:
        self.dim = dim
        half = dim // 2
        scaled_base = base * scale_factor ** (dim / (dim - 2))
        freqs = 1.0 / (scaled_base ** (np.arange(0, half) / half))
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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    dim = 32
    seq = 16
    n_heads = 4

    # ALiBi
    s = alibi_slopes(n_heads)
    assert s.shape == (n_heads,), f"ALiBi slopes shape {s.shape}"
    b = alibi_bias(seq, n_heads)
    assert b.shape == (n_heads, seq, seq), f"ALiBi bias shape {b.shape}"
    assert np.all(b[:, 0, :] <= 0), "ALiBi bias row 0 should be ≤ 0"

    # CPE
    cpe = ConditionalPositionalEncoding(dim)
    x = np.random.randn(seq, dim)
    y = cpe.encode(x)
    assert y.shape == (seq, dim), f"CPE shape {y.shape}"

    # Musical Meter
    mme = MusicalMeterEncoding(dim)
    bars = np.arange(seq)
    beats = np.arange(seq) % 4
    subs = np.zeros(seq, dtype=int)
    ticks = np.zeros(seq, dtype=int)
    enc = mme.encode(bars, beats, subs, ticks)
    assert enc.shape == (seq, dim), f"Meter shape {enc.shape}"

    # Pitch-Class
    ppe = PitchPositionalEncoding(dim)
    pitches = np.array([60, 62, 64, 65, 67, 69, 71, 72])
    enc = ppe.encode(pitches)
    assert enc.shape == (8, dim), f"Pitch shape {enc.shape}"
    # C4 and C5 should share chroma
    assert np.allclose(enc[0, :dim // 2], enc[7, :dim // 2]), "C4/C5 same chroma"

    # Multi-Scale
    mst = MultiScaleTemporalEncoding(dim)
    pos = np.arange(seq)
    enc = mst.encode(pos)
    assert enc.shape == (seq, dim), f"MultiScale shape {enc.shape}"

    # Log Relative Position
    lrpb = LogRelativePositionBias(n_heads)
    b = lrpb.get_bias(seq)
    assert b.shape == (n_heads, seq, seq), f"LogRelBias shape {b.shape}"
    buckets = log_relative_position_buckets(np.array([0, 1, -1, 100, -100]))
    assert buckets[0] == 0, "Bucket for rel=0 should be 0"

    # NTK-Aware RoPE
    ntk = NTKAwareRoPE(dim, scale_factor=2.0)
    x = np.random.randn(seq, dim)
    y = ntk.apply(x)
    assert y.shape == (seq, dim), f"NTK RoPE shape {y.shape}"
    assert not np.allclose(x, y), "NTK RoPE should modify input"

    parts = [
        "ALiBi", "CPE", "MusicalMeter", "PitchClass",
        "MultiScale", "LogRelBias", "NTK-RoPE",
    ]
    print(f"Self-test passed: {', '.join(parts)}.")
