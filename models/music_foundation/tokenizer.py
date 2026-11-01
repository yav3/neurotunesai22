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


# ============================================================
# EMA codebook (van den Oord et al., 2017; Razavi et al., 2019)
# ============================================================

class EMACodebook:
    """Exponential moving average codebook for VQ-VAE training.

    Updates codebook entries via Laplace-smoothed EMA rather than
    backpropagation, following van den Oord et al. (2017):

        N_i ← γ N_i + (1−γ) n_i
        m_i ← γ m_i + (1−γ) Σ_j x_j   (where j → i)
        e_i ← m_i / N_i

    Also implements codebook reset for dead entries (Dhariwal et al., 2020).
    """

    def __init__(self, n_codes: int = 1024, dim: int = 256,
                 decay: float = 0.99, epsilon: float = 1e-5,
                 reset_threshold: int = 2, seed: int = 42):
        self.n_codes = n_codes
        self.dim = dim
        self.decay = decay
        self.epsilon = epsilon
        self.reset_threshold = reset_threshold

        rng = np.random.RandomState(seed)
        self.embeddings = rng.randn(n_codes, dim) * 0.1
        self._ema_count = np.zeros(n_codes)
        self._ema_weight = self.embeddings.copy()
        self._usage_count = np.zeros(n_codes, dtype=int)
        self._total_steps = 0

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Quantize x (N, dim) → (quantized, indices, commitment_loss)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        dists = (
            np.sum(x ** 2, axis=1, keepdims=True)
            - 2 * x @ self.embeddings.T
            + np.sum(self.embeddings ** 2, axis=1, keepdims=True).T
        )
        indices = np.argmin(dists, axis=1)
        quantized = self.embeddings[indices]

        commitment_loss = float(np.mean((x - quantized) ** 2))
        return quantized, indices, commitment_loss

    def ema_update(self, x: np.ndarray, indices: np.ndarray):
        """Update codebook via EMA on a batch."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        one_hot = np.zeros((len(indices), self.n_codes))
        one_hot[np.arange(len(indices)), indices] = 1.0

        batch_count = one_hot.sum(axis=0)  # (n_codes,)
        batch_sum = one_hot.T @ x          # (n_codes, dim)

        self._ema_count = self.decay * self._ema_count + (1 - self.decay) * batch_count
        self._ema_weight = self.decay * self._ema_weight + (1 - self.decay) * batch_sum

        n = self._ema_count
        laplace_n = (n + self.epsilon) / (n.sum() + self.n_codes * self.epsilon) * n.sum()
        self.embeddings = self._ema_weight / laplace_n[:, np.newaxis]

        self._usage_count += batch_count.astype(int)
        self._total_steps += 1

    def reset_dead_codes(self, x: np.ndarray, rng: np.random.RandomState):
        """Replace dead codebook entries with random training samples."""
        dead = self._usage_count < self.reset_threshold
        n_dead = int(dead.sum())
        if n_dead == 0 or len(x) == 0:
            return 0

        replace_idx = rng.choice(len(x), size=min(n_dead, len(x)), replace=False)
        dead_indices = np.where(dead)[0][:len(replace_idx)]

        self.embeddings[dead_indices] = x[replace_idx] + rng.randn(len(dead_indices), self.dim) * 0.01
        self._ema_weight[dead_indices] = self.embeddings[dead_indices]
        self._ema_count[dead_indices] = 1.0
        self._usage_count[dead_indices] = 1
        return int(len(dead_indices))

    def perplexity(self) -> float:
        total = self._usage_count.sum()
        if total == 0:
            return 0.0
        probs = self._usage_count / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(np.exp(entropy))

    def utilisation(self) -> float:
        """Fraction of codes used at least once."""
        return float(np.mean(self._usage_count > 0))


# ============================================================
# Gumbel-softmax quantizer (Jang et al., 2017)
# ============================================================

def _gumbel_sample(shape: tuple, rng: np.random.RandomState, eps: float = 1e-10) -> np.ndarray:
    u = rng.uniform(0, 1, shape)
    return -np.log(-np.log(u + eps) + eps)


def gumbel_softmax_quantize(
    logits: np.ndarray,
    codebook: np.ndarray,
    temperature: float = 1.0,
    hard: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Gumbel-softmax quantization (straight-through in forward pass).

    Args:
        logits: (N, n_codes) unnormalised log-probs over codebook entries.
        codebook: (n_codes, dim) codebook embeddings.
        temperature: Gumbel temperature τ (lower → harder selection).
        hard: If True, argmax in forward, soft gradients in backward (STE).

    Returns:
        (quantized, soft_probs) where quantized is (N, dim).
    """
    rng = np.random.RandomState(seed)
    gumbel = _gumbel_sample(logits.shape, rng)
    y = (logits + gumbel) / temperature

    y_exp = np.exp(y - y.max(axis=-1, keepdims=True))
    soft_probs = y_exp / (y_exp.sum(axis=-1, keepdims=True) + 1e-10)

    if hard:
        indices = np.argmax(soft_probs, axis=-1)
        hard_probs = np.zeros_like(soft_probs)
        hard_probs[np.arange(len(indices)), indices] = 1.0
        probs = hard_probs - soft_probs + soft_probs  # STE
    else:
        probs = soft_probs

    quantized = probs @ codebook  # (N, dim)
    return quantized, soft_probs


# ============================================================
# Finite Scalar Quantization (Mentzer et al., 2023)
# ============================================================

class FiniteScalarQuantizer:
    """Finite Scalar Quantization: per-dimension uniform binning.

    Each dimension is independently quantized to one of L levels in [-1, 1].
    Total codebook size = L^d (implicit, no stored codebook needed).
    Straight-through estimator for gradient flow.
    """

    def __init__(self, dim: int = 128, levels: int = 8):
        self.dim = dim
        self.levels = levels
        self.step = 2.0 / levels
        self.centers = np.linspace(-1 + self.step / 2, 1 - self.step / 2, levels)

    @property
    def codebook_size(self) -> int:
        return self.levels ** self.dim

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantize x (N, dim) → (quantized, indices_per_dim).

        indices_per_dim: (N, dim) integer indices in [0, levels).
        """
        x_clipped = np.clip(x, -1, 1)
        bucket = np.floor((x_clipped + 1) / self.step).astype(int)
        bucket = np.clip(bucket, 0, self.levels - 1)
        quantized = self.centers[bucket]
        return quantized, bucket

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Dequantize indices (N, dim) → continuous (N, dim)."""
        return self.centers[np.clip(indices, 0, self.levels - 1)]

    def commitment_loss(self, x: np.ndarray) -> float:
        quantized, _ = self.quantize(x)
        return float(np.mean((x - quantized) ** 2))

    def entropy_per_dim(self, indices: np.ndarray) -> np.ndarray:
        """Per-dimension entropy of the binned distribution (dim,)."""
        entropies = np.zeros(self.dim)
        for d in range(min(self.dim, indices.shape[1])):
            counts = np.bincount(indices[:, d], minlength=self.levels)
            probs = counts / (counts.sum() + 1e-10)
            probs = probs[probs > 0]
            entropies[d] = -np.sum(probs * np.log(probs))
        return entropies


# ============================================================
# Product Quantization (Jégou et al., 2011)
# ============================================================

class ProductQuantizer:
    """Product quantization: split dim into G groups, VQ each independently.

    Total implicit codebook size = K^G where K = codes per group.
    """

    def __init__(self, dim: int = 256, n_groups: int = 8,
                 codes_per_group: int = 256, seed: int = 42):
        self.dim = dim
        self.n_groups = n_groups
        self.codes_per_group = codes_per_group
        self.group_dim = dim // n_groups

        rng = np.random.RandomState(seed)
        self.codebooks = rng.randn(n_groups, codes_per_group, self.group_dim) * 0.1

    @property
    def codebook_size(self) -> int:
        return self.codes_per_group ** self.n_groups

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Quantize x (N, dim) → (quantized, group_indices (N, G), loss)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        N = x.shape[0]
        quantized = np.zeros_like(x[:, :self.group_dim * self.n_groups])
        indices = np.zeros((N, self.n_groups), dtype=int)
        total_loss = 0.0

        for g in range(self.n_groups):
            start = g * self.group_dim
            end = start + self.group_dim
            x_g = x[:, start:end]
            cb = self.codebooks[g]  # (K, group_dim)

            dists = (
                np.sum(x_g ** 2, axis=1, keepdims=True)
                - 2 * x_g @ cb.T
                + np.sum(cb ** 2, axis=1, keepdims=True).T
            )
            idx = np.argmin(dists, axis=1)
            indices[:, g] = idx
            quantized[:, start:end] = cb[idx]
            total_loss += float(np.mean((x_g - cb[idx]) ** 2))

        return quantized, indices, total_loss / self.n_groups

    def update_centroids(self, x: np.ndarray, indices: np.ndarray,
                         lr: float = 0.01):
        """Online centroid update for each group."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        for g in range(self.n_groups):
            start = g * self.group_dim
            end = start + self.group_dim
            x_g = x[:, start:end]
            for k in range(self.codes_per_group):
                mask = indices[:, g] == k
                if mask.any():
                    centroid = np.mean(x_g[mask], axis=0)
                    self.codebooks[g, k] += lr * (centroid - self.codebooks[g, k])


# ============================================================
# Residual VQ with configurable quantizer backend
# ============================================================

class ResidualVQ:
    """Multi-level residual vector quantization (Zeghidour et al., 2021).

    Cascades Q levels: each quantizer encodes the residual left by the previous.
    Supports both standard codebook and EMA codebook backends.
    """

    def __init__(self, n_levels: int = 8, n_codes: int = 1024,
                 dim: int = 256, use_ema: bool = True,
                 decay: float = 0.99, seed: int = 42):
        self.n_levels = n_levels
        self.dim = dim
        if use_ema:
            self.quantizers = [
                EMACodebook(n_codes, dim, decay=decay, seed=seed + i * 137)
                for i in range(n_levels)
            ]
        else:
            self.quantizers = [
                MusicCodebook(n_codes, dim, seed=seed + i * 137)
                for i in range(n_levels)
            ]
        self._use_ema = use_ema

    def encode(self, x: np.ndarray) -> list[np.ndarray]:
        """Encode x (N, dim) → list of Q index arrays, each (N,)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        residual = x.copy()
        if residual.shape[-1] < self.dim:
            pad = np.zeros((residual.shape[0], self.dim))
            pad[:, :residual.shape[-1]] = residual
            residual = pad
        elif residual.shape[-1] > self.dim:
            residual = residual[:, :self.dim]

        all_indices = []
        for q in self.quantizers:
            quantized, indices, _ = q.quantize(residual)
            all_indices.append(indices)
            residual = residual - quantized
        return all_indices

    def decode(self, all_indices: list[np.ndarray]) -> np.ndarray:
        """Decode list of Q index arrays → reconstructed (N, dim)."""
        N = len(all_indices[0])
        recon = np.zeros((N, self.dim))
        for q, indices in zip(self.quantizers, all_indices):
            if hasattr(q, 'decode'):
                recon += q.decode(indices)
            else:
                recon += q.embeddings[indices]
        return recon

    def encode_decode(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], float]:
        """Full forward pass: quantize → dequantize, return (recon, indices, loss)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        residual = x.copy()
        if residual.shape[-1] != self.dim:
            pad = np.zeros((residual.shape[0], self.dim))
            m = min(residual.shape[-1], self.dim)
            pad[:, :m] = residual[:, :m]
            residual = pad

        recon = np.zeros_like(residual)
        all_indices = []
        total_loss = 0.0

        for q in self.quantizers:
            quantized, indices, loss = q.quantize(residual)
            all_indices.append(indices)
            recon += quantized
            total_loss += loss
            residual = residual - quantized

        return recon, all_indices, total_loss / self.n_levels

    def train_step(self, x: np.ndarray, rng: Optional[np.random.RandomState] = None):
        """One EMA update step across all levels."""
        if not self._use_ema:
            return
        if x.ndim == 1:
            x = x.reshape(1, -1)
        residual = x.copy()
        if residual.shape[-1] != self.dim:
            pad = np.zeros((residual.shape[0], self.dim))
            m = min(residual.shape[-1], self.dim)
            pad[:, :m] = residual[:, :m]
            residual = pad

        for q in self.quantizers:
            quantized, indices, _ = q.quantize(residual)
            q.ema_update(residual, indices)
            if rng is not None:
                q.reset_dead_codes(residual, rng)
            residual = residual - quantized

    def bandwidth(self, sr: int = 22050, hop: int = 256) -> float:
        """Estimated bandwidth in kbps for the RVQ codec."""
        frame_rate = sr / hop
        bits_per_frame = sum(
            np.ceil(np.log2(q.n_codes)) for q in self.quantizers
        )
        return float(bits_per_frame * frame_rate / 1000)

    def perplexity_per_level(self) -> list[float]:
        return [q.perplexity() for q in self.quantizers]


# ============================================================
# Self-test
# ============================================================

def _self_test():
    ema = EMACodebook(n_codes=64, dim=32, seed=0)
    x = np.random.randn(100, 32)
    q, idx, loss = ema.quantize(x)
    assert q.shape == (100, 32) and loss > 0
    ema.ema_update(x, idx)
    assert ema.utilisation() > 0

    fsq = FiniteScalarQuantizer(dim=16, levels=4)
    q2, bidx = fsq.quantize(np.random.randn(50, 16) * 0.5)
    assert q2.shape == (50, 16)
    assert bidx.max() < 4
    assert fsq.commitment_loss(np.random.randn(50, 16) * 0.5) > 0

    pq = ProductQuantizer(dim=64, n_groups=4, codes_per_group=32)
    q3, gidx, ploss = pq.quantize(np.random.randn(30, 64))
    assert gidx.shape == (30, 4) and ploss > 0

    rvq = ResidualVQ(n_levels=4, n_codes=64, dim=32, use_ema=True, seed=0)
    recon, indices, rloss = rvq.encode_decode(np.random.randn(20, 32))
    assert recon.shape == (20, 32) and len(indices) == 4
    rvq.train_step(np.random.randn(20, 32), np.random.RandomState(0))
    bw = rvq.bandwidth()
    assert bw > 0

    logits = np.random.randn(10, 64)
    cb = np.random.randn(64, 32)
    gq, sp = gumbel_softmax_quantize(logits, cb, temperature=0.5)
    assert gq.shape == (10, 32) and sp.shape == (10, 64)

    print(f"Self-test passed: EMA codebook, FSQ, PQ, RVQ({rvq.n_levels}L, "
          f"bw={bw:.1f}kbps), Gumbel-softmax.")


if __name__ == "__main__":
    _self_test()
