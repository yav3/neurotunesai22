"""Frontier music foundation model: deep transformer for learned music representations.

Composes the encoder, multi-head attention, representation learning, and
generation into a single forward-pass pipeline suitable for pretraining
on large-scale audio corpora.

Architecture (loosely following wav2vec 2.0 / HuBERT / MERT):
  1. Feature extraction: convolutional front-end (1-D temporal conv stack)
  2. Positional encoding: sinusoidal + optional relative bias
  3. Transformer encoder: L layers of pre-norm multi-head self-attention + FFN
  4. Quantization head: VQ-VAE codebook for discrete targets
  5. Projection heads: contrastive, masked-prediction, downstream task heads
  6. Full forward pass with configurable pretraining objectives

References:
  - Baevski et al. (2020), wav2vec 2.0
  - Hsu et al. (2021), HuBERT
  - Li et al. (2023), MERT
  - Vaswani et al. (2017), Transformer
  - Ba et al. (2016), Layer Normalization
  - He et al. (2016), residual connections

Pure numpy. Forward-pass only (no autograd).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FrontierConfig:
    # audio input
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128

    # convolutional feature extractor
    conv_channels: List[int] = field(default_factory=lambda: [512, 512, 512, 512, 512])
    conv_kernels: List[int] = field(default_factory=lambda: [10, 3, 3, 3, 3])
    conv_strides: List[int] = field(default_factory=lambda: [5, 2, 2, 2, 2])

    # transformer
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 1024

    # quantization
    n_codebooks: int = 1
    codebook_size: int = 320
    codebook_dim: int = 256

    # pretraining
    mask_prob: float = 0.065
    mask_length: int = 10
    contrastive_temperature: float = 0.1
    n_negatives: int = 100
    diversity_weight: float = 0.1

    # downstream
    n_classes: int = 10


# ---------- convolutional feature extractor ----------

class ConvFeatureExtractor:
    """1-D temporal convolution stack (wav2vec 2.0 style front-end).

    Converts raw waveform or mel spectrogram frames into a sequence of
    feature vectors at a reduced temporal resolution.
    """

    def __init__(self, config: FrontierConfig, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.layers = []
        in_ch = config.n_mels

        for i, (out_ch, k, s) in enumerate(zip(
            config.conv_channels, config.conv_kernels, config.conv_strides
        )):
            scale = np.sqrt(2.0 / (in_ch * k))
            W = rng.randn(out_ch, in_ch, k) * scale
            b = np.zeros(out_ch)
            self.layers.append((W, b, s))
            in_ch = out_ch

        self.out_dim = config.conv_channels[-1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: (C_in, T) input spectrogram or (T,) raw signal reshaped

        Returns:
            (T', C_out) feature sequence
        """
        h = x
        if h.ndim == 1:
            h = h.reshape(1, -1)

        for W, b, stride in self.layers:
            h = self._conv1d(h, W, b, stride)
            h = _gelu(h)
            h = _group_norm(h, n_groups=min(32, h.shape[0]))

        return h.T  # (T', C_out)

    @staticmethod
    def _conv1d(
        x: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int
    ) -> np.ndarray:
        """1-D convolution: (C_in, T) * (C_out, C_in, K) -> (C_out, T')."""
        C_out, C_in, K = W.shape
        T = x.shape[1]
        T_out = max(1, (T - K) // stride + 1)
        out = np.zeros((C_out, T_out))

        for t in range(T_out):
            start = t * stride
            patch = x[:, start:start + K]  # (C_in, K)
            if patch.shape[1] < K:
                padded = np.zeros((C_in, K))
                padded[:, :patch.shape[1]] = patch
                patch = padded
            for c in range(C_out):
                out[c, t] = np.sum(W[c] * patch) + b[c]

        return out

    def receptive_field(self) -> int:
        """Total receptive field in input samples."""
        rf = 1
        stride_product = 1
        for _, _, s in reversed(self.layers):
            rf += (self.layers[0][0].shape[2] - 1) * stride_product
            stride_product *= s
        return rf


# ---------- transformer building blocks ----------

class TransformerLayer:
    """Pre-norm transformer encoder layer (Ba et al., 2016).

    Self-attention → residual → layer-norm → FFN → residual → layer-norm.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        scale_attn = np.sqrt(2.0 / (d_model + d_model))
        scale_ff = np.sqrt(2.0 / (d_model + d_ff))

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = rng.randn(d_model, d_model) * scale_attn
        self.W_k = rng.randn(d_model, d_model) * scale_attn
        self.W_v = rng.randn(d_model, d_model) * scale_attn
        self.W_o = rng.randn(d_model, d_model) * scale_attn

        self.W_ff1 = rng.randn(d_model, d_ff) * scale_ff
        self.b_ff1 = np.zeros(d_ff)
        self.W_ff2 = rng.randn(d_ff, d_model) * scale_ff
        self.b_ff2 = np.zeros(d_model)

        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def forward(
        self, x: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            x: (T, d_model)
            mask: (T, T) boolean, True = attend

        Returns:
            (output, attention_weights)
        """
        T, d = x.shape
        h = self.n_heads
        dk = self.d_k

        # pre-norm
        x_norm = _layer_norm(x, self.gamma1, self.beta1)

        # multi-head self-attention
        Q = (x_norm @ self.W_q).reshape(T, h, dk).transpose(1, 0, 2)
        K = (x_norm @ self.W_k).reshape(T, h, dk).transpose(1, 0, 2)
        V = (x_norm @ self.W_v).reshape(T, h, dk).transpose(1, 0, 2)

        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(dk)

        if mask is not None:
            scores = np.where(mask[np.newaxis], scores, -1e9)

        weights = _softmax(scores)
        attn_out = (weights @ V).transpose(1, 0, 2).reshape(T, d)
        attn_out = attn_out @ self.W_o

        x = x + attn_out

        # pre-norm FFN
        x_norm = _layer_norm(x, self.gamma2, self.beta2)
        ff = _gelu(x_norm @ self.W_ff1 + self.b_ff1) @ self.W_ff2 + self.b_ff2
        x = x + ff

        avg_weights = weights.mean(axis=0)
        return x, avg_weights


class TransformerEncoder:
    """Stack of L transformer layers with sinusoidal position embeddings."""

    def __init__(self, config: FrontierConfig, seed: int = 42):
        self.config = config
        rng = np.random.RandomState(seed)

        # linear projection from conv output to d_model
        in_dim = config.conv_channels[-1] if config.conv_channels else config.n_mels
        scale = np.sqrt(2.0 / (in_dim + config.d_model))
        self.proj_W = rng.randn(in_dim, config.d_model) * scale
        self.proj_b = np.zeros(config.d_model)

        # positional encoding
        self.pos_enc = _sinusoidal_encoding(config.max_seq_len, config.d_model)

        # transformer layers
        self.layers = [
            TransformerLayer(config.d_model, config.n_heads, config.d_ff, seed + i * 100)
            for i in range(config.n_layers)
        ]

        # final layer norm
        self.final_gamma = np.ones(config.d_model)
        self.final_beta = np.zeros(config.d_model)

    def forward(
        self,
        features: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_all_layers: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Encode a feature sequence through the transformer stack.

        Args:
            features: (T, C) from conv extractor
            mask: (T, T) attention mask
            return_all_layers: if True, collect hidden states from each layer

        Returns:
            dict with 'hidden_states', 'attention_weights', optionally 'all_layers'
        """
        T = features.shape[0]
        x = features @ self.proj_W + self.proj_b
        x = x + self.pos_enc[:T]

        all_layers = []
        all_weights = []

        for layer in self.layers:
            x, w = layer.forward(x, mask)
            if return_all_layers:
                all_layers.append(x.copy())
            all_weights.append(w)

        x = _layer_norm(x, self.final_gamma, self.final_beta)

        result = {
            'hidden_states': x,
            'attention_weights': np.stack(all_weights),
        }
        if return_all_layers:
            result['all_layers'] = np.stack(all_layers)

        return result


# ---------- quantization module ----------

class QuantizationModule:
    """Gumbel-softmax product quantization for discrete representation targets.

    Following wav2vec 2.0: G codebook groups, each with V entries of dim d/G.
    """

    def __init__(self, config: FrontierConfig, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n_codebooks = config.n_codebooks
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim

        self.codebooks = rng.randn(
            config.n_codebooks, config.codebook_size, config.codebook_dim
        ) * 0.01

        scale = np.sqrt(2.0 / (config.d_model + config.codebook_dim * config.n_codebooks))
        self.proj_W = rng.randn(config.d_model, config.codebook_dim * config.n_codebooks) * scale

    def forward(
        self, x: np.ndarray, temperature: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Quantize hidden states.

        Args:
            x: (T, d_model) continuous representations
            temperature: Gumbel-softmax temperature

        Returns:
            dict with 'quantized', 'indices', 'perplexity'
        """
        T = x.shape[0]
        projected = x @ self.proj_W  # (T, n_codebooks * codebook_dim)
        projected = projected.reshape(T, self.n_codebooks, self.codebook_dim)

        all_indices = []
        all_quantized = []

        for g in range(self.n_codebooks):
            logits = projected[:, g] @ self.codebooks[g].T  # (T, V)
            logits = logits / max(temperature, 1e-6)

            probs = _softmax(logits)
            indices = np.argmax(probs, axis=-1)  # (T,)
            quantized = self.codebooks[g][indices]  # (T, codebook_dim)

            all_indices.append(indices)
            all_quantized.append(quantized)

        quantized = np.concatenate(all_quantized, axis=-1)  # (T, n_codebooks * codebook_dim)
        indices = np.stack(all_indices, axis=-1)  # (T, n_codebooks)

        # codebook perplexity (diversity measure)
        flat_indices = indices.reshape(-1)
        counts = np.bincount(flat_indices, minlength=self.codebook_size * self.n_codebooks)
        probs = counts / max(counts.sum(), 1)
        probs = probs[probs > 0]
        perplexity = float(np.exp(-np.sum(probs * np.log(probs))))

        return {
            'quantized': quantized,
            'indices': indices,
            'perplexity': perplexity,
        }


# ---------- projection heads ----------

class ContrastiveHead:
    """Linear projection for contrastive pretraining loss (wav2vec 2.0 style)."""

    def __init__(self, d_model: int, proj_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (d_model + proj_dim))
        self.W = rng.randn(d_model, proj_dim) * scale
        self.b = np.zeros(proj_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project and L2-normalize.

        Args:
            x: (T, d_model)

        Returns:
            (T, proj_dim) L2-normalized projections
        """
        z = x @ self.W + self.b
        norms = np.linalg.norm(z, axis=-1, keepdims=True)
        return z / (norms + 1e-8)


class ClassificationHead:
    """Two-layer MLP for downstream classification tasks."""

    def __init__(self, d_model: int, n_classes: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        hidden = d_model // 2
        scale1 = np.sqrt(2.0 / (d_model + hidden))
        scale2 = np.sqrt(2.0 / (hidden + n_classes))
        self.W1 = rng.randn(d_model, hidden) * scale1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, n_classes) * scale2
        self.b2 = np.zeros(n_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Classify from pooled representation.

        Args:
            x: (d_model,) or (B, d_model)

        Returns:
            (n_classes,) or (B, n_classes) logits
        """
        h = _gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


# ---------- masking ----------

def generate_span_mask(
    T: int, mask_prob: float = 0.065, mask_length: int = 10, seed: int = 42
) -> np.ndarray:
    """Generate span masking pattern (wav2vec 2.0 / HuBERT style).

    Randomly selects starting positions and masks contiguous spans.

    Args:
        T: sequence length
        mask_prob: fraction of timesteps to mask
        mask_length: length of each masked span

    Returns:
        (T,) boolean mask, True = masked
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros(T, dtype=bool)
    n_masked = max(1, int(T * mask_prob))
    n_spans = max(1, n_masked // mask_length)

    starts = rng.choice(max(1, T - mask_length), size=n_spans, replace=False)
    for s in starts:
        mask[s:s + mask_length] = True

    return mask


# ---------- loss functions ----------

def contrastive_loss(
    context: np.ndarray,
    targets: np.ndarray,
    negatives: np.ndarray,
    temperature: float = 0.1,
) -> float:
    """InfoNCE contrastive loss (van den Oord et al., 2018).

    Args:
        context: (T_masked, D) context vectors at masked positions
        targets: (T_masked, D) positive targets (quantized)
        negatives: (T_masked, K, D) sampled negative targets
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    T, D = context.shape
    K = negatives.shape[1]

    pos_sim = np.sum(context * targets, axis=-1) / temperature  # (T,)

    neg_sim = np.einsum('td,tkd->tk', context, negatives) / temperature  # (T, K)

    logits = np.concatenate([pos_sim[:, np.newaxis], neg_sim], axis=-1)  # (T, 1+K)
    log_probs = logits - _logsumexp(logits, axis=-1, keepdims=True)

    loss = -log_probs[:, 0].mean()
    return float(loss)


def diversity_loss(perplexity: float, codebook_size: int) -> float:
    """Encourage uniform codebook usage.

    Maximizes perplexity toward codebook_size (uniform distribution).
    """
    return float(codebook_size - perplexity) / codebook_size


def masked_prediction_loss(
    predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> float:
    """L1 loss on masked positions for feature prediction.

    Args:
        predictions: (T, D) predicted features
        targets: (T, D) original features
        mask: (T,) boolean mask

    Returns:
        scalar loss
    """
    if mask.sum() == 0:
        return 0.0
    diff = np.abs(predictions[mask] - targets[mask])
    return float(diff.mean())


# ---------- the frontier model ----------

class MusicFrontierModel:
    """Full frontier music foundation model.

    Combines convolutional feature extraction, transformer encoding,
    quantization, and multiple pretraining heads into a single
    forward-pass pipeline.
    """

    def __init__(self, config: Optional[FrontierConfig] = None, seed: int = 42):
        self.config = config or FrontierConfig()
        self.seed = seed

        self.feature_extractor = ConvFeatureExtractor(self.config, seed)
        self.transformer = TransformerEncoder(self.config, seed + 1000)
        self.quantizer = QuantizationModule(self.config, seed + 2000)
        self.contrastive_head = ContrastiveHead(
            self.config.d_model, self.config.codebook_dim, seed + 3000
        )
        self.classifier = ClassificationHead(
            self.config.d_model, self.config.n_classes, seed + 4000
        )

    def extract_features(self, spectrogram: np.ndarray) -> np.ndarray:
        """Run the convolutional front-end.

        Args:
            spectrogram: (n_mels, T) mel spectrogram

        Returns:
            (T', conv_channels[-1]) feature sequence
        """
        return self.feature_extractor.forward(spectrogram)

    def encode(
        self,
        spectrogram: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_all_layers: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Full encoding pipeline: conv → transformer.

        Args:
            spectrogram: (n_mels, T) mel spectrogram
            mask: optional attention mask
            return_all_layers: collect all intermediate representations

        Returns:
            dict with 'hidden_states', 'features', 'attention_weights'
        """
        features = self.extract_features(spectrogram)
        result = self.transformer.forward(features, mask, return_all_layers)
        result['features'] = features
        return result

    def pretrain_forward(
        self,
        spectrogram: np.ndarray,
        mask_seed: int = 42,
    ) -> Dict[str, object]:
        """Full pretraining forward pass.

        1. Extract features
        2. Quantize unmasked features → targets
        3. Replace masked positions with learned mask embedding
        4. Run transformer on partially-masked sequence
        5. Compute contrastive + diversity + prediction losses

        Args:
            spectrogram: (n_mels, T) mel spectrogram
            mask_seed: seed for span mask generation

        Returns:
            dict with losses, representations, and diagnostics
        """
        features = self.extract_features(spectrogram)
        T = features.shape[0]

        # quantize original (unmasked) features
        quant_result = self.quantizer.forward(
            features @ self.transformer.proj_W + self.transformer.proj_b
        )

        # generate span mask
        span_mask = generate_span_mask(
            T, self.config.mask_prob, self.config.mask_length, mask_seed
        )

        # mask features (replace with zeros — learned mask embedding in real impl)
        masked_features = features.copy()
        masked_features[span_mask] = 0.0

        # encode masked sequence
        enc_result = self.transformer.forward(masked_features)
        hidden = enc_result['hidden_states']

        # contrastive loss on masked positions
        masked_indices = np.where(span_mask)[0]
        if len(masked_indices) > 0:
            context_vecs = self.contrastive_head.forward(hidden[masked_indices])
            target_vecs = self.contrastive_head.forward(
                quant_result['quantized'][masked_indices]
                @ np.linalg.pinv(self.transformer.proj_W.T)[:self.config.codebook_dim * self.config.n_codebooks, :self.config.d_model]
                if self.config.codebook_dim * self.config.n_codebooks != self.config.d_model
                else quant_result['quantized'][masked_indices]
            )

            # sample negatives from unmasked positions
            unmasked_indices = np.where(~span_mask)[0]
            rng = np.random.RandomState(mask_seed + 1)
            K = min(self.config.n_negatives, len(unmasked_indices))
            if K > 0 and len(masked_indices) > 0:
                neg_idx = rng.choice(
                    unmasked_indices, size=(len(masked_indices), K), replace=True
                )
                neg_vecs = self.contrastive_head.forward(
                    hidden[neg_idx.reshape(-1)]
                ).reshape(len(masked_indices), K, -1)

                c_loss = contrastive_loss(
                    context_vecs, target_vecs, neg_vecs, self.config.contrastive_temperature
                )
            else:
                c_loss = 0.0

            # prediction loss
            pred_loss = masked_prediction_loss(
                hidden, features @ self.transformer.proj_W + self.transformer.proj_b,
                span_mask
            )
        else:
            c_loss = 0.0
            pred_loss = 0.0

        # diversity loss
        d_loss = diversity_loss(quant_result['perplexity'], self.config.codebook_size)

        total_loss = (
            c_loss
            + self.config.diversity_weight * d_loss
            + pred_loss
        )

        return {
            'total_loss': total_loss,
            'contrastive_loss': c_loss,
            'diversity_loss': d_loss,
            'prediction_loss': pred_loss,
            'perplexity': quant_result['perplexity'],
            'hidden_states': hidden,
            'mask': span_mask,
            'n_masked': int(span_mask.sum()),
        }

    def classify(self, spectrogram: np.ndarray) -> np.ndarray:
        """Downstream classification from mean-pooled representation.

        Args:
            spectrogram: (n_mels, T)

        Returns:
            (n_classes,) logits
        """
        result = self.encode(spectrogram)
        pooled = result['hidden_states'].mean(axis=0)
        return self.classifier.forward(pooled)

    def get_representation(
        self, spectrogram: np.ndarray, layer: int = -1
    ) -> np.ndarray:
        """Extract representation from a specific transformer layer.

        Args:
            spectrogram: (n_mels, T)
            layer: which layer (-1 = final, 0 = first)

        Returns:
            (T', d_model) hidden states from specified layer
        """
        result = self.encode(spectrogram, return_all_layers=True)
        if 'all_layers' in result:
            return result['all_layers'][layer]
        return result['hidden_states']

    def similarity(
        self, spec_a: np.ndarray, spec_b: np.ndarray
    ) -> float:
        """Cosine similarity between two spectrograms in representation space."""
        rep_a = self.encode(spec_a)['hidden_states'].mean(axis=0)
        rep_b = self.encode(spec_b)['hidden_states'].mean(axis=0)
        dot = np.dot(rep_a, rep_b)
        norm = np.linalg.norm(rep_a) * np.linalg.norm(rep_b)
        return float(dot / (norm + 1e-8))

    def param_count(self) -> Dict[str, int]:
        """Count parameters in each component."""
        counts = {}

        # conv extractor
        conv_params = sum(W.size + b.size for W, b, _ in self.feature_extractor.layers)
        counts['conv_extractor'] = conv_params

        # transformer projection
        counts['transformer_proj'] = (
            self.transformer.proj_W.size + self.transformer.proj_b.size
        )

        # transformer layers
        layer_params = 0
        for layer in self.transformer.layers:
            layer_params += (
                layer.W_q.size + layer.W_k.size + layer.W_v.size + layer.W_o.size
                + layer.W_ff1.size + layer.b_ff1.size
                + layer.W_ff2.size + layer.b_ff2.size
                + layer.gamma1.size + layer.beta1.size
                + layer.gamma2.size + layer.beta2.size
            )
        counts['transformer_layers'] = layer_params

        # quantizer
        counts['quantizer'] = self.quantizer.codebooks.size + self.quantizer.proj_W.size

        # heads
        counts['contrastive_head'] = self.contrastive_head.W.size + self.contrastive_head.b.size
        counts['classifier'] = (
            self.classifier.W1.size + self.classifier.b1.size
            + self.classifier.W2.size + self.classifier.b2.size
        )

        counts['total'] = sum(counts.values())
        return counts


# ---------- training utilities ----------

@dataclass
class TrainingState:
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    loss_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)


def cosine_warmup_schedule(
    step: int,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    base_lr: float = 3e-4,
    min_lr: float = 1e-6,
) -> float:
    """Cosine annealing with linear warmup (Loshchilov & Hutter, 2017)."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def batch_spectrograms(
    signals: List[np.ndarray],
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    sr: int = 22050,
) -> List[np.ndarray]:
    """Convert raw signals to mel spectrograms for model input.

    Returns list of (n_mels, T_i) spectrograms (variable length).
    """
    specs = []
    for sig in signals:
        spec = _simple_mel_spectrogram(sig, n_mels, n_fft, hop_length, sr)
        specs.append(spec)
    return specs


# ---------- helpers ----------

def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return m.squeeze(axis=axis) if not keepdims else m + np.log(
        np.sum(np.exp(x - m), axis=axis, keepdims=keepdims) + 1e-10
    )


def _layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _group_norm(x: np.ndarray, n_groups: int = 32, eps: float = 1e-6) -> np.ndarray:
    """Group normalization (Wu & He, 2018). x: (C, T)."""
    C, T = x.shape
    G = min(n_groups, C)
    x_grouped = x.reshape(G, C // G, T)
    mean = x_grouped.mean(axis=(1, 2), keepdims=True)
    var = x_grouped.var(axis=(1, 2), keepdims=True)
    x_norm = (x_grouped - mean) / np.sqrt(var + eps)
    return x_norm.reshape(C, T)


def _sinusoidal_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    pos = np.arange(max_len, dtype=np.float64)[:, np.newaxis]
    dim = np.arange(d_model, dtype=np.float64)[np.newaxis, :]
    angles = pos / (10000 ** (2 * (dim // 2) / d_model))
    encoding = np.zeros((max_len, d_model))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


def _simple_mel_spectrogram(
    signal: np.ndarray,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    sr: int = 22050,
) -> np.ndarray:
    """Minimal mel spectrogram for model input. Returns (n_mels, T)."""
    n_frames = max(1, (len(signal) - n_fft) // hop_length + 1)
    spectrogram = np.zeros((n_fft // 2 + 1, n_frames))

    window = np.hanning(n_fft)
    for t in range(n_frames):
        start = t * hop_length
        frame = signal[start:start + n_fft]
        if len(frame) < n_fft:
            padded = np.zeros(n_fft)
            padded[:len(frame)] = frame
            frame = padded
        fft = np.fft.rfft(frame * window)
        spectrogram[:, t] = np.abs(fft) ** 2

    # mel filterbank
    fmin, fmax = 0.0, sr / 2
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(left, center):
            if center > left:
                filterbank[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                filterbank[m, k] = (right - k) / (right - center)

    mel_spec = filterbank @ spectrogram
    return np.log1p(mel_spec)
