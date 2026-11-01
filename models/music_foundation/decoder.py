"""Decoder: masked self-attention transformer decoder, generation heads, and
advanced sampling strategies (top-k, nucleus/top-p, beam search, KV-cache).

References:
  - Holtzman et al. (2020) "The Curious Case of Neural Text Degeneration" (nucleus)
  - Fan et al. (2018) "Hierarchical Neural Story Generation" (top-k)
  - Keskar et al. (2019) repetition penalty
  - Vaswani et al. (2017) beam search, KV-cache
  - Ho & Salimans (2022) classifier-free guidance
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .encoder import LinearLayer, gelu, layer_norm, MultiHeadAttention


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    latent_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2


@dataclass
class SamplingConfig:
    """Controls token-level sampling behaviour."""
    temperature: float = 1.0
    top_k: int = 0          # 0 = disabled
    top_p: float = 1.0      # 1.0 = disabled (nucleus)
    repetition_penalty: float = 1.0   # >1 penalises repeats
    frequency_penalty: float = 0.0    # additive penalty per occurrence
    presence_penalty: float = 0.0     # additive penalty if token appeared at all
    min_p: float = 0.0      # Gemma-style min-p filtering
    typical_p: float = 1.0  # locally-typical sampling (1.0 = disabled)
    seed: int | None = None


# ---------------------------------------------------------------------------
# KV-cache
# ---------------------------------------------------------------------------

class KVCache:
    """Stores key/value tensors per layer for efficient autoregressive decode."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.keys: List[Optional[np.ndarray]] = [None] * n_layers
        self.values: List[Optional[np.ndarray]] = [None] * n_layers

    def update(self, layer_idx: int, new_k: np.ndarray, new_v: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_k
            self.values[layer_idx] = new_v
        else:
            self.keys[layer_idx] = np.concatenate(
                [self.keys[layer_idx], new_k], axis=0)
            self.values[layer_idx] = np.concatenate(
                [self.values[layer_idx], new_v], axis=0)
        return self.keys[layer_idx], self.values[layer_idx]

    @property
    def seq_len(self) -> int:
        if self.keys[0] is None:
            return 0
        return self.keys[0].shape[0]

    def reset(self) -> None:
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers


# ---------------------------------------------------------------------------
# Attention layers
# ---------------------------------------------------------------------------

class MaskedMultiHeadAttention:
    def __init__(self, dim: int, n_heads: int, seed: int = 0) -> None:
        self.attn = MultiHeadAttention(dim, n_heads, seed)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def forward(self, x: np.ndarray, kv_cache: KVCache | None = None,
                layer_idx: int = 0) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n, d = x.shape
        h = self.n_heads
        hd = self.head_dim

        Q = self.attn.Wq.forward(x).reshape(n, h, hd)
        K = self.attn.Wk.forward(x).reshape(n, h, hd)
        V = self.attn.Wv.forward(x).reshape(n, h, hd)

        if kv_cache is not None:
            K, V = kv_cache.update(layer_idx, K, V)

        m = K.shape[0]
        scores = np.einsum("nhd,mhd->hnm", Q, K) / np.sqrt(hd)

        # causal mask: each query position can attend only to positions <= its
        # absolute index.  When using KV-cache, the query tokens correspond to
        # positions [m-n .. m-1], so the mask shape is (n, m).
        offset = m - n
        causal_mask = np.zeros((n, m))
        for i in range(n):
            causal_mask[i, offset + i + 1:] = -1e9
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


# ---------------------------------------------------------------------------
# Decoder layer and full decoder
# ---------------------------------------------------------------------------

class TransformerDecoderLayer:
    def __init__(self, dim: int, n_heads: int, seed: int = 0) -> None:
        self.self_attn = MaskedMultiHeadAttention(dim, n_heads, seed)
        self.cross_attn = CrossAttention(dim, n_heads, seed + 100)
        self.ff1 = LinearLayer(dim, dim * 4, seed + 200)
        self.ff2 = LinearLayer(dim * 4, dim, seed + 300)

    def forward(self, x: np.ndarray, context: np.ndarray | None = None,
                kv_cache: KVCache | None = None, layer_idx: int = 0
                ) -> np.ndarray:
        h = layer_norm(x + self.self_attn.forward(x, kv_cache, layer_idx))
        if context is not None:
            h = layer_norm(h + self.cross_attn.forward(h, context))
        ff_out = self.ff2.forward(gelu(self.ff1.forward(h)))
        return layer_norm(h + ff_out)


class MusicDecoder:
    def __init__(self, config: DecoderConfig, seed: int = 500) -> None:
        self.config = config
        self.input_proj = LinearLayer(config.latent_dim, config.hidden_dim, seed)
        self.layers = [
            TransformerDecoderLayer(config.hidden_dim, config.n_heads,
                                   seed + i * 400)
            for i in range(config.n_layers)
        ]
        self.output_proj = LinearLayer(config.hidden_dim, config.output_dim,
                                       seed + 900)

    def decode(self, z: np.ndarray, context: np.ndarray | None = None,
               kv_cache: KVCache | None = None) -> np.ndarray:
        if z.ndim == 1:
            z = z.reshape(1, -1)
        h = gelu(self.input_proj.forward(z))
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, context, kv_cache, layer_idx=i)
        return self.output_proj.forward(h)

    # --- legacy generate (no KV-cache, simple temperature) ----------------

    def generate(self, z: np.ndarray, max_steps: int = 100,
                 temperature: float = 1.0) -> np.ndarray:
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

    # --- KV-cached generate -----------------------------------------------

    def generate_cached(self, z: np.ndarray, max_steps: int = 100,
                        temperature: float = 1.0,
                        context: np.ndarray | None = None) -> np.ndarray:
        """Autoregressive generation with KV-cache (O(n) per step instead of
        O(n^2) from re-encoding the full prefix)."""
        if z.ndim == 1:
            z = z.reshape(1, -1)
        cache = KVCache(len(self.layers))

        first_out = self.decode(z, context, cache)
        outputs = [first_out]
        current = first_out[-1:]

        for _ in range(max_steps - 1):
            step_out = self.decode(current, context, cache)
            last = step_out[-1:]
            if temperature != 1.0:
                last = last / max(temperature, 1e-8)
            outputs.append(last)
            current = last

        return np.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Zero out all but the top-k logits (set to -inf)."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    out = logits.copy()
    threshold = np.partition(out, -k, axis=-1)[..., -k]
    if out.ndim == 1:
        out[out < threshold] = -1e9
    else:
        mask = out < threshold[..., None]
        out[mask] = -1e9
    return out


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability mass >= p (Holtzman et al., 2020)."""
    if p >= 1.0:
        return logits
    probs = _softmax(logits)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        logits_2d = logits.reshape(1, -1)
        squeeze = True
    else:
        logits_2d = logits
        squeeze = False

    out = logits_2d.copy()
    for i in range(out.shape[0]):
        sorted_idx = np.argsort(-probs[i])
        cum = np.cumsum(probs[i][sorted_idx])
        cutoff = np.searchsorted(cum, p) + 1
        remove = sorted_idx[cutoff:]
        out[i, remove] = -1e9

    return out.squeeze(0) if squeeze else out


def min_p_filter(logits: np.ndarray, min_p: float) -> np.ndarray:
    """Min-p filtering: remove tokens with prob < min_p * max_prob."""
    if min_p <= 0.0:
        return logits
    probs = _softmax(logits)
    max_prob = probs.max(axis=-1, keepdims=True)
    out = logits.copy()
    out[probs < min_p * max_prob] = -1e9
    return out


def typical_p_filter(logits: np.ndarray, typical_p: float) -> np.ndarray:
    """Locally typical sampling (Meister et al., 2023): keep tokens whose
    information content is close to the expected information content."""
    if typical_p >= 1.0:
        return logits
    probs = _softmax(logits)
    log_probs = np.log(probs + 1e-12)
    entropy = -np.sum(probs * log_probs, axis=-1, keepdims=True)
    surprise = -log_probs
    deviation = np.abs(surprise - entropy)

    out = logits.copy()
    if out.ndim == 1:
        sorted_idx = np.argsort(deviation)
        cum = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cum, typical_p) + 1
        out[sorted_idx[cutoff:]] = -1e9
    else:
        for i in range(out.shape[0]):
            sorted_idx = np.argsort(deviation[i])
            cum = np.cumsum(probs[i][sorted_idx])
            cutoff = np.searchsorted(cum, typical_p) + 1
            out[i, sorted_idx[cutoff:]] = -1e9
    return out


def apply_repetition_penalty(logits: np.ndarray,
                             token_ids: np.ndarray,
                             penalty: float = 1.0,
                             frequency_penalty: float = 0.0,
                             presence_penalty: float = 0.0) -> np.ndarray:
    """Multiplicative + additive repetition penalty (Keskar et al., 2019).

    - `penalty` > 1: divide positive logits / multiply negative logits by penalty
      for each previously generated token.
    - `frequency_penalty`: subtract count(token) * freq_pen from logit.
    - `presence_penalty`: subtract presence_pen if token appeared at all.
    """
    if penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits
    if len(token_ids) == 0:
        return logits

    out = logits.copy()
    unique, counts = np.unique(token_ids, return_counts=True)
    unique = unique.astype(int)

    if out.ndim == 1:
        for tok, cnt in zip(unique, counts):
            if 0 <= tok < out.shape[-1]:
                if out[tok] > 0:
                    out[tok] /= penalty
                else:
                    out[tok] *= penalty
                out[tok] -= frequency_penalty * cnt
                out[tok] -= presence_penalty
    else:
        for tok, cnt in zip(unique, counts):
            if 0 <= tok < out.shape[-1]:
                pos = out[:, tok] > 0
                out[pos, tok] /= penalty
                out[~pos, tok] *= penalty
                out[:, tok] -= frequency_penalty * cnt
                out[:, tok] -= presence_penalty
    return out


def sample_logits(logits: np.ndarray, cfg: SamplingConfig,
                  past_tokens: np.ndarray | None = None) -> np.ndarray:
    """Full sampling pipeline: temperature → repetition penalty → top-k →
    top-p → min-p → typical-p → categorical sample."""
    rng = np.random.RandomState(cfg.seed)
    l = logits.copy()

    # temperature
    if cfg.temperature > 0 and cfg.temperature != 1.0:
        l = l / cfg.temperature
    elif cfg.temperature <= 0:
        # greedy
        if l.ndim == 1:
            return np.array([np.argmax(l)])
        return np.argmax(l, axis=-1)

    # repetition penalty
    if past_tokens is not None and len(past_tokens) > 0:
        l = apply_repetition_penalty(l, past_tokens, cfg.repetition_penalty,
                                     cfg.frequency_penalty,
                                     cfg.presence_penalty)

    # filtering
    l = top_k_filter(l, cfg.top_k)
    l = top_p_filter(l, cfg.top_p)
    l = min_p_filter(l, cfg.min_p)
    l = typical_p_filter(l, cfg.typical_p)

    probs = _softmax(l)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    indices = np.array([rng.choice(probs.shape[-1], p=p / (p.sum() + 1e-12))
                        for p in probs])
    return indices[0] if squeeze else indices


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

@dataclass
class BeamHypothesis:
    tokens: List[int] = field(default_factory=list)
    log_prob: float = 0.0
    cache: KVCache | None = None


def beam_search(gen_head: 'GenerationHead',
                decoder: MusicDecoder,
                z: np.ndarray,
                max_steps: int = 50,
                beam_width: int = 4,
                length_penalty: float = 0.6,
                context: np.ndarray | None = None,
                eos_token: int | None = None,
                ) -> List[BeamHypothesis]:
    """Beam search over a GenerationHead attached to a MusicDecoder.

    Returns `beam_width` hypotheses sorted by length-normalised log-prob.
    """
    if z.ndim == 1:
        z = z.reshape(1, -1)

    def _make_cache() -> KVCache:
        return KVCache(len(decoder.layers))

    # initialise
    init_cache = _make_cache()
    init_out = decoder.decode(z, context, init_cache)
    init_logits = gen_head.logits(init_out[-1:])
    init_log_probs = np.log(_softmax(init_logits.squeeze(0)) + 1e-12)

    top_idx = np.argsort(-init_log_probs)[:beam_width]
    beams: List[BeamHypothesis] = []
    for idx in top_idx:
        beams.append(BeamHypothesis(
            tokens=[int(idx)],
            log_prob=float(init_log_probs[idx]),
            cache=None,  # caches diverge after first step; re-encode
        ))

    finished: List[BeamHypothesis] = []

    for step in range(1, max_steps):
        candidates: List[BeamHypothesis] = []
        for beam in beams:
            if eos_token is not None and len(beam.tokens) > 0 and beam.tokens[-1] == eos_token:
                finished.append(beam)
                continue

            # re-encode from scratch (small model; production would keep caches)
            # map token back to latent via the transpose of the projection
            seq = z.copy()
            W = gen_head.proj.W  # (latent_dim, vocab_size)
            for tok in beam.tokens:
                tok_emb = W[:, tok].reshape(1, -1)
                seq = np.concatenate([seq, tok_emb], axis=0)

            cache = _make_cache()
            out = decoder.decode(seq, context, cache)
            logits = gen_head.logits(out[-1:]).squeeze(0)
            log_probs = np.log(_softmax(logits) + 1e-12)

            top_idx_step = np.argsort(-log_probs)[:beam_width]
            for idx in top_idx_step:
                candidates.append(BeamHypothesis(
                    tokens=beam.tokens + [int(idx)],
                    log_prob=beam.log_prob + float(log_probs[idx]),
                ))

        if not candidates:
            break

        def _score(h: BeamHypothesis) -> float:
            lp = ((5 + len(h.tokens)) / 6) ** length_penalty
            return h.log_prob / lp

        candidates.sort(key=_score, reverse=True)
        beams = candidates[:beam_width]

    all_hyps = finished + beams
    all_hyps.sort(
        key=lambda h: h.log_prob / ((5 + len(h.tokens)) / 6) ** length_penalty,
        reverse=True)
    return all_hyps[:beam_width]


# ---------------------------------------------------------------------------
# Classifier-free guidance wrapper
# ---------------------------------------------------------------------------

def classifier_free_guidance(decoder: MusicDecoder,
                             z: np.ndarray,
                             context: np.ndarray,
                             guidance_scale: float = 3.0,
                             kv_cache_cond: KVCache | None = None,
                             kv_cache_uncond: KVCache | None = None,
                             ) -> np.ndarray:
    """Ho & Salimans (2022): output = uncond + w * (cond - uncond).

    Returns the guided decoder output (hidden dim). Caller applies the
    generation head on top.
    """
    out_cond = decoder.decode(z, context, kv_cache_cond)
    out_uncond = decoder.decode(z, None, kv_cache_uncond)
    return out_uncond + guidance_scale * (out_cond - out_uncond)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

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
        return _softmax(raw)

    def sample(self, z: np.ndarray, temperature: float = 1.0,
               seed: int | None = None) -> np.ndarray:
        rng = np.random.RandomState(seed)
        probs = self.predict(z, temperature)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        indices = np.array([rng.choice(self.vocab_size, p=p) for p in probs])
        return indices

    def sample_advanced(self, z: np.ndarray, cfg: SamplingConfig,
                        past_tokens: np.ndarray | None = None) -> np.ndarray:
        """Sample using the full SamplingConfig pipeline."""
        logits = self.logits(z)
        return sample_logits(logits, cfg, past_tokens)

    def loss(self, z: np.ndarray, target_indices: np.ndarray) -> float:
        logits = self.logits(z)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(
            np.sum(np.exp(shifted), axis=-1, keepdims=True))
        n = len(target_indices)
        return float(-np.mean(
            log_probs[np.arange(n), target_indices.astype(int)]))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    print("decoder self-test …")
    rng = np.random.RandomState(42)

    cfg = DecoderConfig(latent_dim=32, hidden_dim=64, output_dim=32,
                        n_heads=2, n_layers=2)
    dec = MusicDecoder(cfg, seed=0)

    # basic decode
    z = rng.randn(4, 32).astype(np.float32)
    ctx = rng.randn(3, 64).astype(np.float32)
    out = dec.decode(z, ctx)
    assert out.shape == (4, 32), f"decode shape {out.shape}"

    # legacy generate
    seq = dec.generate(z[:1], max_steps=5)
    assert seq.shape == (5, 32), f"generate shape {seq.shape}"

    # KV-cached generate
    cached = dec.generate_cached(z[:1], max_steps=5, context=ctx)
    assert cached.shape[0] == 5, f"cached steps {cached.shape[0]}"
    assert cached.shape[1] == 32

    # KV-cache object
    cache = KVCache(2)
    assert cache.seq_len == 0
    cache.update(0, rng.randn(3, 2, 16), rng.randn(3, 2, 16))
    assert cache.seq_len == 3
    cache.update(0, rng.randn(1, 2, 16), rng.randn(1, 2, 16))
    assert cache.seq_len == 4
    cache.reset()
    assert cache.seq_len == 0

    # top-k filter
    logits = np.array([1.0, 5.0, 3.0, 0.5, 4.0])
    filt = top_k_filter(logits, 2)
    assert np.sum(filt > -1e8) == 2

    # top-p filter
    logits2 = np.array([10.0, 1.0, 0.1, 0.01, -5.0])
    filt2 = top_p_filter(logits2, 0.9)
    probs = _softmax(filt2)
    assert probs[0] > 0.5

    # min-p filter
    logits3 = np.array([5.0, 4.9, 0.1, -1.0, -2.0])
    filt3 = min_p_filter(logits3, 0.1)
    assert np.sum(filt3 > -1e8) >= 2

    # typical-p filter
    filt4 = typical_p_filter(logits2, 0.5)
    assert np.sum(filt4 > -1e8) >= 1

    # repetition penalty
    logits_rep = np.array([2.0, 3.0, 1.0, 4.0])
    past = np.array([1, 1, 3])
    pen = apply_repetition_penalty(logits_rep, past, penalty=2.0,
                                   frequency_penalty=0.5)
    assert pen[1] < logits_rep[1], "rep penalty should reduce repeated token"
    assert pen[3] < logits_rep[3]

    # full sampling pipeline
    big_logits = rng.randn(20).astype(np.float32)
    sc = SamplingConfig(temperature=0.8, top_k=5, top_p=0.9,
                        repetition_penalty=1.2, seed=42)
    tok = sample_logits(big_logits, sc, past_tokens=np.array([0, 1, 2]))
    assert 0 <= tok < 20

    # greedy sampling
    sc_greedy = SamplingConfig(temperature=0.0)
    tok_greedy = sample_logits(big_logits, sc_greedy)
    assert tok_greedy == np.argmax(big_logits)

    # generation head
    gh = GenerationHead(32, 20, seed=0)
    z_single = rng.randn(1, 32).astype(np.float32)
    probs_gh = gh.predict(z_single)
    assert abs(probs_gh.sum() - 1.0) < 1e-5

    tok_basic = gh.sample(z_single, seed=42)
    assert 0 <= tok_basic[0] < 20

    tok_adv = gh.sample_advanced(z_single, sc)
    assert 0 <= tok_adv < 20

    loss_val = gh.loss(z_single, np.array([5]))
    assert loss_val > 0

    # reconstruction head
    rh = ReconstructionHead(32, 16, seed=0)
    out_rh = rh.forward(z_single)
    assert out_rh.shape == (1, 16)
    loss_rh = rh.loss(z_single, rng.randn(1, 16).astype(np.float32))
    assert loss_rh > 0

    # beam search
    hyps = beam_search(gh, dec, z[:1], max_steps=4, beam_width=3)
    assert len(hyps) <= 3
    assert len(hyps[0].tokens) > 0

    # classifier-free guidance
    guided = classifier_free_guidance(dec, z[:1], ctx, guidance_scale=2.0)
    assert guided.shape == (1, 32)

    print("  all decoder tests passed ✓")


if __name__ == "__main__":
    _self_test()
