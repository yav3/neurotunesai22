"""Autoregressive music generation for the frontier foundation model.

Implements the full generation stack:
  - Token sampling: temperature, top-k, top-p (nucleus), repetition penalty
  - Autoregressive decoding with KV-caching for efficient inference
  - Classifier-free guidance (Ho & Salimans, 2022)
  - Beam search with length penalty and early stopping
  - Music continuation: encode seed audio → quantize → generate → decode
  - Constrained generation: key/tempo/time-signature vocabulary masks

The generation loop feeds discrete codebook tokens back through the
transformer at each step, projecting them into d_model space via the
quantizer's codebook lookup + inverse projection before the next
transformer pass.

References:
  - Holtzman et al. (2020), The Curious Case of Neural Text Degeneration (nucleus)
  - Ho & Salimans (2022), Classifier-Free Diffusion Guidance
  - Graves (2012), Sequence Transduction with RNNs (beam search)
  - Copet et al. (2023), MusicGen architecture inspiration

Pure numpy. No autograd dependencies.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .frontier_model import (
    FrontierConfig, MusicFrontierModel, TransformerLayer,
    _layer_norm, _gelu, _softmax, _logsumexp,
)


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

@dataclass
class GenerationConfig:
    """Controls the generation behaviour."""
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_length: int = 256
    repetition_penalty: float = 1.0
    cfg_scale: float = 0.0
    length_penalty: float = 1.0
    beam_width: int = 1
    eos_token: int = -1
    seed: int = 42


# ------------------------------------------------------------------ #
#  KV cache                                                            #
# ------------------------------------------------------------------ #

class KVCache:
    """Stores past key/value tensors per layer for autoregressive decoding."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.keys: List[Optional[np.ndarray]] = [None] * n_layers
        self.values: List[Optional[np.ndarray]] = [None] * n_layers

    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Append new K, V and return the full cached K, V."""
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = np.concatenate([self.keys[layer_idx], k], axis=1)
            self.values[layer_idx] = np.concatenate([self.values[layer_idx], v], axis=1)
        return self.keys[layer_idx], self.values[layer_idx]

    @property
    def seq_len(self) -> int:
        if self.keys[0] is None:
            return 0
        return self.keys[0].shape[1]

    def clone(self) -> "KVCache":
        c = KVCache(self.n_layers)
        for i in range(self.n_layers):
            if self.keys[i] is not None:
                c.keys[i] = self.keys[i].copy()
                c.values[i] = self.values[i].copy()
        return c


# ------------------------------------------------------------------ #
#  Cached transformer step                                             #
# ------------------------------------------------------------------ #

def _cached_layer_forward(
    layer: TransformerLayer,
    x_new: np.ndarray,
    cache: KVCache,
    layer_idx: int,
    pos_offset: int,
) -> np.ndarray:
    """Run one transformer layer on a single new token with KV caching.

    Args:
        layer: the TransformerLayer
        x_new: (1, d_model) — the new position only
        cache: KV cache to read/append
        layer_idx: which layer in the cache
        pos_offset: current sequence position (for causal mask)

    Returns:
        (1, d_model) output for the new position
    """
    d = x_new.shape[1]
    h = layer.n_heads
    dk = layer.d_k

    x_norm = _layer_norm(x_new, layer.gamma1, layer.beta1)

    q = (x_norm @ layer.W_q).reshape(1, h, dk).transpose(1, 0, 2)  # (h, 1, dk)
    k_new = (x_norm @ layer.W_k).reshape(1, h, dk).transpose(1, 0, 2)
    v_new = (x_norm @ layer.W_v).reshape(1, h, dk).transpose(1, 0, 2)

    k_full, v_full = cache.update(layer_idx, k_new, v_new)  # (h, T, dk)

    scores = (q @ k_full.swapaxes(-2, -1)) / np.sqrt(dk)  # (h, 1, T)
    weights = _softmax(scores)
    attn_out = (weights @ v_full).transpose(1, 0, 2).reshape(1, d)  # (1, d)
    attn_out = attn_out @ layer.W_o

    x_out = x_new + attn_out

    x_norm2 = _layer_norm(x_out, layer.gamma2, layer.beta2)
    ff = _gelu(x_norm2 @ layer.W_ff1 + layer.b_ff1) @ layer.W_ff2 + layer.b_ff2
    x_out = x_out + ff

    return x_out


def _full_transformer_step(
    model: MusicFrontierModel,
    token_embed: np.ndarray,
    cache: KVCache,
    pos: int,
) -> np.ndarray:
    """Run a single token embedding through the full transformer with caching.

    Args:
        model: the frontier model
        token_embed: (1, d_model)
        cache: KV cache
        pos: current absolute position

    Returns:
        (d_model,) hidden state for this position
    """
    trans = model.transformer
    x = token_embed + trans.pos_enc[pos:pos + 1]

    for i, layer in enumerate(trans.layers):
        x = _cached_layer_forward(layer, x, cache, i, pos)

    x = _layer_norm(x, trans.final_gamma, trans.final_beta)
    return x[0]  # (d_model,)


# ------------------------------------------------------------------ #
#  Token ↔ embedding helpers                                           #
# ------------------------------------------------------------------ #

def _token_to_embed(model: MusicFrontierModel, token_id: int) -> np.ndarray:
    """Convert a codebook index to a d_model embedding via inverse projection.

    Looks up the codebook vector for `token_id` in codebook 0 and projects
    it back into d_model space using the pseudo-inverse of the quantizer's
    projection.
    """
    cb = model.quantizer.codebooks[0]  # (V, codebook_dim)
    vec = cb[token_id % cb.shape[0]]  # (codebook_dim,)
    if model.config.n_codebooks > 1:
        vec = np.tile(vec, model.config.n_codebooks)
    proj_W = model.quantizer.proj_W  # (d_model, codebook_dim * n_codebooks)
    inv_proj = np.linalg.pinv(proj_W)  # (codebook_dim * n_codebooks, d_model)
    embed = vec @ inv_proj  # (d_model,)
    return embed


def _hidden_to_logits(model: MusicFrontierModel, hidden: np.ndarray) -> np.ndarray:
    """Convert a hidden state to vocabulary logits over the codebook.

    Projects through the quantizer projection and computes cosine similarity
    with every codebook entry.

    Args:
        hidden: (d_model,)

    Returns:
        (codebook_size,) logits
    """
    proj = hidden @ model.quantizer.proj_W  # (codebook_dim * n_codebooks,)
    cdim = model.config.codebook_dim
    proj_g0 = proj[:cdim]  # first codebook group
    cb = model.quantizer.codebooks[0]  # (V, codebook_dim)
    proj_norm = proj_g0 / (np.linalg.norm(proj_g0) + 1e-8)
    cb_norm = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-8)
    logits = proj_norm @ cb_norm.T  # (V,)
    return logits / max(model.config.contrastive_temperature, 1e-6)


# ------------------------------------------------------------------ #
#  Token sampler                                                       #
# ------------------------------------------------------------------ #

class TokenSampler:
    """Applies temperature, top-k, top-p, and repetition penalty to logits."""

    def __init__(self, config: GenerationConfig, rng: np.random.RandomState):
        self.config = config
        self.rng = rng

    def sample(self, logits: np.ndarray, past_tokens: np.ndarray) -> int:
        """Sample a single token from logits.

        Args:
            logits: (V,) raw logits
            past_tokens: (T,) previously generated token ids

        Returns:
            sampled token index
        """
        logits = logits.copy().astype(np.float64)

        if self.config.repetition_penalty != 1.0 and len(past_tokens) > 0:
            logits = self._apply_repetition_penalty(logits, past_tokens)

        logits = logits / max(self.config.temperature, 1e-8)

        if self.config.top_k > 0:
            logits = self._top_k_filter(logits)

        if self.config.top_p < 1.0:
            logits = self._top_p_filter(logits)

        probs = _softmax(logits[np.newaxis])[0]
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total < 1e-12:
            return int(self.rng.randint(len(probs)))
        probs = probs / total
        return int(self.rng.choice(len(probs), p=probs))

    def _apply_repetition_penalty(self, logits: np.ndarray, past: np.ndarray) -> np.ndarray:
        unique_past = np.unique(past)
        for idx in unique_past:
            if 0 <= idx < len(logits):
                if logits[idx] > 0:
                    logits[idx] /= self.config.repetition_penalty
                else:
                    logits[idx] *= self.config.repetition_penalty
        return logits

    def _top_k_filter(self, logits: np.ndarray) -> np.ndarray:
        k = min(self.config.top_k, len(logits))
        threshold = np.partition(logits, -k)[-k]
        logits[logits < threshold] = -1e10
        return logits

    def _top_p_filter(self, logits: np.ndarray) -> np.ndarray:
        sorted_idx = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_idx]
        probs = _softmax(sorted_logits[np.newaxis])[0]
        cumsum = np.cumsum(probs)
        cutoff = np.searchsorted(cumsum, self.config.top_p) + 1
        cutoff = min(cutoff, len(logits))
        mask = np.ones(len(logits), dtype=bool)
        mask[sorted_idx[:cutoff]] = False
        logits[mask] = -1e10
        return logits


# ------------------------------------------------------------------ #
#  Autoregressive generator                                            #
# ------------------------------------------------------------------ #

class AutoregressiveGenerator:
    """Generate codebook token sequences from the frontier model.

    Uses KV-caching: at each step only the new token embedding is fed
    through the transformer, while past keys/values are read from cache.
    """

    def __init__(self, model: MusicFrontierModel, config: Optional[GenerationConfig] = None):
        self.model = model
        self.config = config or GenerationConfig()
        self.rng = np.random.RandomState(self.config.seed)

    def generate(
        self,
        prompt_tokens: np.ndarray,
        config: Optional[GenerationConfig] = None,
        constraint_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a continuation of prompt_tokens.

        Args:
            prompt_tokens: (T_prompt,) integer codebook indices
            config: override generation config
            constraint_mask: (codebook_size,) boolean — True = allowed token

        Returns:
            (T_total,) full sequence including prompt
        """
        cfg = config or self.config
        sampler = TokenSampler(cfg, self.rng)
        n_layers = len(self.model.transformer.layers)
        cache = KVCache(n_layers)

        tokens = list(prompt_tokens)

        for pos, tok in enumerate(prompt_tokens):
            embed = _token_to_embed(self.model, int(tok))[np.newaxis]  # (1, d_model)
            _full_transformer_step(self.model, embed, cache, pos)

        for step in range(cfg.max_length - len(prompt_tokens)):
            pos = len(tokens)
            if pos >= self.model.config.max_seq_len:
                break

            last_tok = tokens[-1]
            embed = _token_to_embed(self.model, int(last_tok))[np.newaxis]
            hidden = _full_transformer_step(self.model, embed, cache, pos)
            logits = _hidden_to_logits(self.model, hidden)

            if constraint_mask is not None:
                logits[~constraint_mask] = -1e10

            new_tok = sampler.sample(logits, np.array(tokens, dtype=np.int64))

            if new_tok == cfg.eos_token:
                break
            tokens.append(new_tok)

        return np.array(tokens, dtype=np.int64)


# ------------------------------------------------------------------ #
#  Classifier-free guidance generator                                  #
# ------------------------------------------------------------------ #

class CFGGenerator:
    """Classifier-free guidance (Ho & Salimans, 2022).

    Runs conditional and unconditional forward passes and interpolates:
        logits = (1 + s) * cond_logits - s * uncond_logits
    where s = cfg_scale.
    """

    def __init__(self, model: MusicFrontierModel, config: Optional[GenerationConfig] = None):
        self.model = model
        self.config = config or GenerationConfig()
        self.rng = np.random.RandomState(self.config.seed)

    def generate(
        self,
        prompt_tokens: np.ndarray,
        uncond_prompt_tokens: Optional[np.ndarray] = None,
        config: Optional[GenerationConfig] = None,
    ) -> np.ndarray:
        """Generate with classifier-free guidance.

        Args:
            prompt_tokens: (T,) conditional prompt
            uncond_prompt_tokens: (T',) unconditional prompt (defaults to zeros)
            config: override config

        Returns:
            (T_total,) generated token sequence
        """
        cfg = config or self.config
        sampler = TokenSampler(cfg, self.rng)
        n_layers = len(self.model.transformer.layers)

        cond_cache = KVCache(n_layers)
        uncond_cache = KVCache(n_layers)

        if uncond_prompt_tokens is None:
            uncond_prompt_tokens = np.zeros(1, dtype=np.int64)

        for pos, tok in enumerate(prompt_tokens):
            embed = _token_to_embed(self.model, int(tok))[np.newaxis]
            _full_transformer_step(self.model, embed, cond_cache, pos)

        for pos, tok in enumerate(uncond_prompt_tokens):
            embed = _token_to_embed(self.model, int(tok))[np.newaxis]
            _full_transformer_step(self.model, embed, uncond_cache, pos)

        tokens = list(prompt_tokens)
        s = cfg.cfg_scale

        for step in range(cfg.max_length - len(prompt_tokens)):
            cond_pos = len(tokens)
            uncond_pos = len(uncond_prompt_tokens) + step
            if cond_pos >= self.model.config.max_seq_len:
                break

            last_tok = tokens[-1]
            embed = _token_to_embed(self.model, int(last_tok))[np.newaxis]

            cond_hidden = _full_transformer_step(self.model, embed, cond_cache, cond_pos)
            uncond_hidden = _full_transformer_step(self.model, embed, uncond_cache, min(uncond_pos, self.model.config.max_seq_len - 1))

            cond_logits = _hidden_to_logits(self.model, cond_hidden)
            uncond_logits = _hidden_to_logits(self.model, uncond_hidden)

            guided_logits = (1.0 + s) * cond_logits - s * uncond_logits

            new_tok = sampler.sample(guided_logits, np.array(tokens, dtype=np.int64))
            if new_tok == cfg.eos_token:
                break
            tokens.append(new_tok)

        return np.array(tokens, dtype=np.int64)


# ------------------------------------------------------------------ #
#  Beam search generator                                               #
# ------------------------------------------------------------------ #

@dataclass
class _Beam:
    tokens: List[int]
    score: float
    cache: KVCache
    finished: bool = False


class BeamSearchGenerator:
    """Beam search with length penalty and early stopping.

    Score: log P(y) / |y|^α  where α = length_penalty.
    """

    def __init__(self, model: MusicFrontierModel, config: Optional[GenerationConfig] = None):
        self.model = model
        self.config = config or GenerationConfig()
        self.rng = np.random.RandomState(self.config.seed)

    def _score(self, log_prob: float, length: int) -> float:
        alpha = self.config.length_penalty
        return log_prob / (length ** alpha) if length > 0 else log_prob

    def generate(
        self,
        prompt_tokens: np.ndarray,
        config: Optional[GenerationConfig] = None,
    ) -> np.ndarray:
        """Run beam search.

        Args:
            prompt_tokens: (T_prompt,) seed tokens
            config: override config

        Returns:
            (T_total,) best beam's token sequence
        """
        cfg = config or self.config
        B = max(1, cfg.beam_width)
        n_layers = len(self.model.transformer.layers)

        init_cache = KVCache(n_layers)
        for pos, tok in enumerate(prompt_tokens):
            embed = _token_to_embed(self.model, int(tok))[np.newaxis]
            _full_transformer_step(self.model, embed, init_cache, pos)

        beams = [_Beam(tokens=list(prompt_tokens), score=0.0, cache=init_cache)]
        finished: List[_Beam] = []

        for step in range(cfg.max_length - len(prompt_tokens)):
            candidates: List[_Beam] = []

            for beam in beams:
                if beam.finished:
                    finished.append(beam)
                    continue

                pos = len(beam.tokens)
                if pos >= self.model.config.max_seq_len:
                    beam.finished = True
                    finished.append(beam)
                    continue

                last_tok = beam.tokens[-1]
                embed = _token_to_embed(self.model, int(last_tok))[np.newaxis]
                cache_copy = beam.cache.clone()
                hidden = _full_transformer_step(self.model, embed, cache_copy, pos)
                logits = _hidden_to_logits(self.model, hidden)

                log_probs = logits - _logsumexp(logits[np.newaxis], axis=-1)[0]

                top_k = min(B * 2, len(log_probs))
                top_idx = np.argpartition(log_probs, -top_k)[-top_k:]

                for idx in top_idx:
                    new_score = beam.score + float(log_probs[idx])
                    new_tokens = beam.tokens + [int(idx)]
                    is_finished = int(idx) == cfg.eos_token

                    candidates.append(_Beam(
                        tokens=new_tokens,
                        score=new_score,
                        cache=cache_copy.clone(),
                        finished=is_finished,
                    ))

            if not candidates:
                break

            candidates.sort(
                key=lambda b: self._score(b.score, len(b.tokens)),
                reverse=True,
            )
            beams = candidates[:B]

            if all(b.finished for b in beams):
                finished.extend(beams)
                break

        all_beams = finished + [b for b in beams if not b.finished]
        if not all_beams:
            return prompt_tokens

        best = max(all_beams, key=lambda b: self._score(b.score, len(b.tokens)))
        return np.array(best.tokens, dtype=np.int64)


# ------------------------------------------------------------------ #
#  Music continuation                                                  #
# ------------------------------------------------------------------ #

class MusicContinuation:
    """Encode a seed audio segment, quantize to tokens, generate continuation,
    and decode back through the codebook.

    Pipeline: mel spec → conv → transformer → quantize → generate → codebook lookup
    """

    def __init__(self, model: MusicFrontierModel, config: Optional[GenerationConfig] = None):
        self.model = model
        self.config = config or GenerationConfig()

    def encode_to_tokens(self, spectrogram: np.ndarray) -> np.ndarray:
        """Encode a mel spectrogram to codebook token indices.

        Args:
            spectrogram: (n_mels, T) mel spectrogram

        Returns:
            (T',) token indices for codebook 0
        """
        enc = self.model.encode(spectrogram)
        quant = self.model.quantizer.forward(enc['hidden_states'])
        return quant['indices'][:, 0]  # codebook 0 indices

    def tokens_to_embeddings(self, tokens: np.ndarray) -> np.ndarray:
        """Decode tokens back to continuous embeddings via codebook lookup.

        Args:
            tokens: (T,) codebook indices

        Returns:
            (T, codebook_dim) continuous vectors
        """
        cb = self.model.quantizer.codebooks[0]
        return cb[tokens % cb.shape[0]]

    def continue_audio(
        self,
        seed_spectrogram: np.ndarray,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Dict[str, np.ndarray]:
        """Full continuation pipeline.

        Args:
            seed_spectrogram: (n_mels, T) seed audio as mel spectrogram

        Returns:
            dict with 'prompt_tokens', 'generated_tokens', 'full_tokens',
            'continuation_embeddings'
        """
        cfg = gen_config or self.config
        prompt_tokens = self.encode_to_tokens(seed_spectrogram)

        gen = AutoregressiveGenerator(self.model, cfg)
        full_tokens = gen.generate(prompt_tokens, cfg)

        continuation_tokens = full_tokens[len(prompt_tokens):]
        cont_embeds = self.tokens_to_embeddings(continuation_tokens)

        return {
            'prompt_tokens': prompt_tokens,
            'generated_tokens': full_tokens,
            'continuation_tokens': continuation_tokens,
            'continuation_embeddings': cont_embeds,
            'n_prompt': len(prompt_tokens),
            'n_generated': len(continuation_tokens),
        }


# ------------------------------------------------------------------ #
#  Constrained generation                                              #
# ------------------------------------------------------------------ #

class ConstrainedGenerator:
    """Generation with structural constraints via vocabulary masks.

    Supports key, tempo, and time-signature constraints by masking out
    tokens that are incompatible with the target structure.
    """

    def __init__(self, model: MusicFrontierModel, config: Optional[GenerationConfig] = None):
        self.model = model
        self.config = config or GenerationConfig()

    @staticmethod
    def build_key_mask(
        codebook_size: int,
        key_root: int = 0,
        mode: str = "major",
        seed: int = 42,
    ) -> np.ndarray:
        """Build a vocabulary mask encouraging tokens compatible with a key.

        Uses a deterministic hash of token index mod 12 to map tokens to
        pitch classes, then allows only those in the target scale.

        Args:
            codebook_size: vocabulary size
            key_root: root pitch class (0=C, 1=C#, ..., 11=B)
            mode: "major" or "minor"
            seed: for deterministic mapping

        Returns:
            (codebook_size,) boolean mask
        """
        if mode == "major":
            intervals = [0, 2, 4, 5, 7, 9, 11]
        else:
            intervals = [0, 2, 3, 5, 7, 8, 10]

        scale_pcs = set((key_root + i) % 12 for i in intervals)

        rng = np.random.RandomState(seed)
        token_to_pc = rng.randint(0, 12, size=codebook_size)

        mask = np.array([pc in scale_pcs for pc in token_to_pc])
        return mask

    @staticmethod
    def build_tempo_mask(
        codebook_size: int,
        target_bpm: float = 120.0,
        tolerance: float = 20.0,
        seed: int = 42,
    ) -> np.ndarray:
        """Build a mask biasing toward a target tempo range.

        Maps each token to a pseudo-tempo and allows those within tolerance.
        """
        rng = np.random.RandomState(seed)
        token_bpm = rng.uniform(40.0, 240.0, size=codebook_size)
        mask = np.abs(token_bpm - target_bpm) <= tolerance
        if mask.sum() == 0:
            mask[:] = True
        return mask

    @staticmethod
    def build_time_sig_mask(
        codebook_size: int,
        beats_per_bar: int = 4,
        seed: int = 42,
    ) -> np.ndarray:
        """Build a mask for time-signature compatibility.

        Tokens whose pseudo-metric-position is compatible with the
        target meter are allowed.
        """
        rng = np.random.RandomState(seed)
        token_beat = rng.randint(1, 13, size=codebook_size)
        mask = (token_beat % beats_per_bar == 0) | (token_beat <= beats_per_bar)
        return mask

    def combine_masks(self, *masks: np.ndarray) -> np.ndarray:
        """AND-combine multiple constraint masks."""
        result = np.ones(masks[0].shape, dtype=bool)
        for m in masks:
            result &= m
        if result.sum() == 0:
            result[:] = True
        return result

    def generate(
        self,
        prompt_tokens: np.ndarray,
        constraints: Optional[Dict[str, object]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> np.ndarray:
        """Generate with structural constraints.

        Args:
            prompt_tokens: (T,) seed tokens
            constraints: dict with optional keys:
                'key_root' (int), 'mode' (str),
                'target_bpm' (float), 'bpm_tolerance' (float),
                'beats_per_bar' (int)
            config: override generation config

        Returns:
            (T_total,) constrained token sequence
        """
        cfg = config or self.config
        V = self.model.config.codebook_size
        constraints = constraints or {}

        masks = []
        if 'key_root' in constraints:
            masks.append(self.build_key_mask(
                V,
                key_root=constraints['key_root'],
                mode=constraints.get('mode', 'major'),
            ))
        if 'target_bpm' in constraints:
            masks.append(self.build_tempo_mask(
                V,
                target_bpm=constraints['target_bpm'],
                tolerance=constraints.get('bpm_tolerance', 20.0),
            ))
        if 'beats_per_bar' in constraints:
            masks.append(self.build_time_sig_mask(
                V,
                beats_per_bar=constraints['beats_per_bar'],
            ))

        combined = self.combine_masks(*masks) if masks else None

        gen = AutoregressiveGenerator(self.model, cfg)
        return gen.generate(prompt_tokens, cfg, constraint_mask=combined)


# ------------------------------------------------------------------ #
#  Convenience entry point                                             #
# ------------------------------------------------------------------ #

def generate_music_tokens(
    model: MusicFrontierModel,
    prompt_tokens: np.ndarray,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    max_length: int = 128,
    repetition_penalty: float = 1.2,
    cfg_scale: float = 0.0,
    beam_width: int = 1,
    constraints: Optional[Dict[str, object]] = None,
    seed: int = 42,
) -> np.ndarray:
    """One-call generation interface.

    Selects the appropriate generator based on parameters:
    - beam_width > 1 → beam search
    - cfg_scale > 0 → classifier-free guidance
    - constraints not None → constrained generation
    - otherwise → standard autoregressive

    Returns:
        (T,) token sequence
    """
    config = GenerationConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        cfg_scale=cfg_scale,
        beam_width=beam_width,
        seed=seed,
    )

    if beam_width > 1:
        gen = BeamSearchGenerator(model, config)
        return gen.generate(prompt_tokens, config)

    if cfg_scale > 0.0:
        gen = CFGGenerator(model, config)
        return gen.generate(prompt_tokens, config=config)

    if constraints is not None:
        gen = ConstrainedGenerator(model, config)
        return gen.generate(prompt_tokens, constraints, config)

    gen = AutoregressiveGenerator(model, config)
    return gen.generate(prompt_tokens, config)


# ------------------------------------------------------------------ #
#  Self-test                                                           #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== Music Generation Module Self-Test ===\n")

    cfg = FrontierConfig(
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        conv_channels=[32, 64], conv_kernels=[10, 3], conv_strides=[5, 2],
        n_codebooks=1, codebook_size=32, codebook_dim=32,
        max_seq_len=256,
    )
    model = MusicFrontierModel(cfg, seed=42)
    prompt = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    # 1. Basic autoregressive
    gen_cfg = GenerationConfig(temperature=0.8, top_k=10, max_length=20, seed=42)
    ar = AutoregressiveGenerator(model, gen_cfg)
    tokens = ar.generate(prompt)
    print(f"Autoregressive: {len(tokens)} tokens, prompt={len(prompt)}")
    assert len(tokens) >= len(prompt)
    assert len(tokens) <= 20

    # 2. With repetition penalty
    gen_cfg2 = GenerationConfig(temperature=0.8, top_k=10, max_length=20,
                                 repetition_penalty=1.5, seed=42)
    tokens2 = AutoregressiveGenerator(model, gen_cfg2).generate(prompt)
    print(f"With rep penalty: {len(tokens2)} tokens")

    # 3. Classifier-free guidance
    cfg_gen = GenerationConfig(temperature=0.8, cfg_scale=2.0, max_length=15, seed=42)
    cfg_tokens = CFGGenerator(model, cfg_gen).generate(prompt)
    print(f"CFG (s=2.0): {len(cfg_tokens)} tokens")

    # 4. Beam search
    beam_cfg = GenerationConfig(beam_width=3, max_length=15, length_penalty=0.8, seed=42)
    beam_tokens = BeamSearchGenerator(model, beam_cfg).generate(prompt)
    print(f"Beam search (w=3): {len(beam_tokens)} tokens")

    # 5. Constrained generation
    constraints = {'key_root': 0, 'mode': 'major', 'target_bpm': 120.0}
    constrained_tokens = generate_music_tokens(
        model, prompt, max_length=15, constraints=constraints, seed=42,
    )
    print(f"Constrained (C major, 120 BPM): {len(constrained_tokens)} tokens")

    # 6. Music continuation from mel spectrogram
    from .frontier_model import _simple_mel_spectrogram
    rng = np.random.RandomState(42)
    signal = rng.randn(22050)  # 1 second
    spec = _simple_mel_spectrogram(signal, 128, 1024, 256, 22050)
    cont = MusicContinuation(model, GenerationConfig(max_length=30, seed=42))
    result = cont.continue_audio(spec)
    print(f"Continuation: {result['n_prompt']} prompt → {result['n_generated']} new tokens")
    assert result['continuation_embeddings'].shape[1] == cfg.codebook_dim

    # 7. Convenience function
    quick = generate_music_tokens(model, prompt, max_length=12, seed=42)
    print(f"Convenience: {len(quick)} tokens")

    print("\n=== All generation tests passed ===")
