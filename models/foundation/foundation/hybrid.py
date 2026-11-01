"""NeuroHybrid — foundational-frontier hybrid model.

Architecture
------------
Unifies the multimodal clinical transformer (NeuroFoundation) with the
gene-expression cVAE (PerturbationResponseModel) through a shared
cross-modal backbone, and routes to frontier LLMs when narrow-model
confidence is insufficient.

Three operating modes:
    foundation:   backbone + task head, no frontier call
    augmented:    backbone output enriched by frontier model reasoning
    governed:     data-sufficiency gating selects mode automatically

The gene-expression branch compresses n_genes scalars into K latent
tokens via a learned projection + self-attention bottleneck (a
lightweight Perceiver-style compressor), then feeds them into the
same cross-modal transformer that processes clinical modality tokens.
The [CLS] representation thus captures both clinical state and
molecular context.

Task heads:
    perturbation: cVAE decoder conditioned on [CLS] + perturbation spec
    disease:      14-class neurological state classifier
    expression:   gene-expression reconstruction (pretraining)

Follows the governed evidence discipline: species firewall enforced,
regime declared before method, hash-chained lineage on every output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import (
    DISEASE_CODES,
    MODALITY_DIMS,
    NUM_DISEASE_STATES,
    FoundationConfig,
)
from models.foundation.cross_attention import CrossModalTransformer
from models.foundation.encoders import MultiModalEncoder
from models.foundation.perturbation_response import (
    PerturbationEncoder,
    PerturbationModelConfig,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class HybridMode(Enum):
    FOUNDATION = "foundation"
    AUGMENTED = "augmented"
    GOVERNED = "governed"


@dataclass
class HybridConfig:
    """Unified configuration for the foundational-frontier hybrid."""

    # Shared backbone
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    use_cls_token: bool = True

    # Gene expression branch
    n_genes: int = 5000
    gene_latent_tokens: int = 8
    gene_bottleneck_heads: int = 4
    gene_bottleneck_layers: int = 2

    # Perturbation head (cVAE)
    latent_dim: int = 128
    perturbation_dim: int = 64
    n_drugs: int = 64
    max_gene_idx: int = 25000
    beta: float = 1.0
    consistency_lambda: float = 0.1

    # Disease classification head
    n_disease_states: int = NUM_DISEASE_STATES
    head_hidden_dim: int = 128
    head_dropout: float = 0.2

    # Frontier bridge
    confidence_threshold: float = 0.8
    min_modalities_for_foundation: int = 2

    # Evidence discipline
    seed: int = 2026

    def to_foundation_config(self) -> FoundationConfig:
        return FoundationConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            activation=self.activation,
            use_cls_token=self.use_cls_token,
        )

    def to_perturbation_config(self) -> PerturbationModelConfig:
        return PerturbationModelConfig(
            n_genes=self.n_genes,
            latent_dim=self.latent_dim,
            perturbation_dim=self.perturbation_dim,
            n_drugs=self.n_drugs,
            max_gene_idx=self.max_gene_idx,
            beta=self.beta,
            consistency_lambda=self.consistency_lambda,
        )


# ---------------------------------------------------------------------------
# Gene expression encoder
# ---------------------------------------------------------------------------

class GeneBottleneckBlock(nn.Module):
    """Self-attention + FFN over K latent gene tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class GeneExpressionEncoder(nn.Module):
    """Compress n_genes expression scalars into K latent tokens in R^d.

    Step 1: Linear projection from n_genes → K * d, reshape to (B, K, d).
    Step 2: Self-attention bottleneck (2 layers) refines the K tokens.
    Step 3: Add a learned gene-modality type embedding.

    This is a lightweight Perceiver-style compressor. The linear projection
    acts as the cross-attention (gene space → latent space) in a single
    step; the self-attention bottleneck refines inter-token dependencies.

    Parameter count: n_genes * K * d + bottleneck params.
    For n_genes=5000, K=8, d=256: ~10.5M (projection) + ~2.1M (bottleneck).
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        n_latent_tokens: int = 8,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.n_latent_tokens = n_latent_tokens

        self.input_norm = nn.LayerNorm(n_genes)
        self.proj = nn.Linear(n_genes, n_latent_tokens * hidden_dim)

        self.bottleneck = nn.Sequential(*[
            GeneBottleneckBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.token_pos = nn.Embedding(n_latent_tokens, hidden_dim)
        self.type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """expression: (B, n_genes) → (B, K, d) latent gene tokens."""
        B = expression.shape[0]

        x = self.input_norm(expression)
        x = self.proj(x)  # (B, K * d)
        x = x.view(B, self.n_latent_tokens, self.hidden_dim)

        pos = self.token_pos(
            torch.arange(self.n_latent_tokens, device=expression.device)
        )
        x = x + pos.unsqueeze(0) + self.type_embed

        x = self.bottleneck(x)
        return x


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class PerturbationHead(nn.Module):
    """cVAE perturbation-prediction head conditioned on cross-modal [CLS].

    Uses the proven cVAE architecture (NT-001 through NT-004) but replaces
    the standalone expression encoder with the hybrid backbone's [CLS]
    representation, giving the VAE access to clinical + molecular context.

    Encoder: ([CLS] || h_perturb) → (μ, logvar)
    Decoder: (z || h_perturb || [CLS]) → x̂_target
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        d = config.hidden_dim
        p = config.perturbation_dim
        lat = config.latent_dim
        n_genes = config.n_genes

        self.perturb_enc = PerturbationEncoder(config.to_perturbation_config())

        # Encoder: [CLS] + perturbation → (μ, logvar)
        enc_in = d + p
        self.enc_net = nn.Sequential(
            nn.Linear(enc_in, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(d, lat)
        self.logvar_head = nn.Linear(d, lat)

        # Decoder: z + perturbation + [CLS] → predicted expression
        dec_in = lat + p + d
        self.dec_net = nn.Sequential(
            nn.Linear(dec_in, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d * 2, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d * 2, n_genes),
        )

    def encode(
        self,
        cls_repr: torch.Tensor,
        h_perturb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([cls_repr, h_perturb], dim=-1)
        h = self.enc_net(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(
        self,
        z: torch.Tensor,
        h_perturb: torch.Tensor,
        cls_repr: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z, h_perturb, cls_repr], dim=-1)
        return self.dec_net(x)

    def forward(
        self,
        cls_repr: torch.Tensor,
        ptype_ids: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        dose: torch.Tensor,
        duration: torch.Tensor,
        time_point: torch.Tensor,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        h_p = self.perturb_enc(
            ptype_ids, gene_idx, drug_idx, dose, duration, time_point,
        )
        mu, logvar = self.encode(cls_repr, h_p)

        if training:
            std = (0.5 * logvar).exp()
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        predicted = self.decode(z, h_p, cls_repr)
        return {"mu": mu, "logvar": logvar, "z": z, "predicted": predicted}


class DiseaseClassificationHead(nn.Module):
    """14-class neurological disease state classifier on [CLS]."""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_dim, config.n_disease_states),
        )

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        return self.net(cls_repr)


class ExpressionReconstructionHead(nn.Module):
    """Reconstruct gene expression from gene tokens (pretraining)."""

    def __init__(self, config: HybridConfig):
        super().__init__()
        d = config.hidden_dim
        K = config.gene_latent_tokens
        self.proj = nn.Sequential(
            nn.Linear(K * d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, config.n_genes),
        )
        self.K = K
        self.d = d

    def forward(self, gene_tokens: torch.Tensor) -> torch.Tensor:
        """gene_tokens: (B, K, d) → (B, n_genes) reconstructed expression."""
        B = gene_tokens.shape[0]
        flat = gene_tokens.view(B, self.K * self.d)
        return self.proj(flat)


# ---------------------------------------------------------------------------
# Hybrid model
# ---------------------------------------------------------------------------

class NeuroHybrid(nn.Module):
    """Foundational-frontier hybrid model.

    Unifies clinical multimodal features and gene expression in a shared
    cross-modal transformer backbone, with task-specific heads and a
    confidence-gated frontier bridge.

    Input modalities (any subset):
        - Clinical: dict of (modality_name → (B, d_m)) tensors
        - Gene expression: (B, n_genes) tensor

    Outputs (depending on head):
        - cls_repr: (B, d) cross-modal patient representation
        - perturbation: {mu, logvar, z, predicted} from cVAE head
        - disease: (B, 14) logits
        - expression_recon: (B, n_genes) reconstructed expression
        - confidence: scalar data-sufficiency estimate
    """

    def __init__(self, config: HybridConfig | None = None):
        super().__init__()
        self.config = config or HybridConfig()
        c = self.config

        # Clinical modality encoder (reuses existing architecture)
        self.modality_encoder = MultiModalEncoder(c.hidden_dim, c.dropout)

        # Gene expression encoder (new)
        self.gene_encoder = GeneExpressionEncoder(
            n_genes=c.n_genes,
            hidden_dim=c.hidden_dim,
            n_latent_tokens=c.gene_latent_tokens,
            n_heads=c.gene_bottleneck_heads,
            n_layers=c.gene_bottleneck_layers,
            dropout=c.dropout,
        )

        # Shared cross-modal transformer
        n_modalities = len(MODALITY_DIMS)
        max_seq = 1 + n_modalities + c.gene_latent_tokens  # [CLS] + modalities + gene tokens
        self.transformer = CrossModalTransformer(
            hidden_dim=c.hidden_dim,
            num_layers=c.num_layers,
            num_heads=c.num_heads,
            ffn_dim=c.ffn_dim,
            dropout=c.dropout,
            activation=c.activation,
            use_cls=c.use_cls_token,
        )

        # Task heads
        self.perturbation_head = PerturbationHead(c)
        self.disease_head = DiseaseClassificationHead(c)
        self.expression_recon_head = ExpressionReconstructionHead(c)

        # Clinical reconstruction heads (for pretraining)
        self.clinical_recon_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(c.hidden_dim, c.hidden_dim),
                nn.GELU(),
                nn.Linear(c.hidden_dim, dim),
            )
            for name, dim in MODALITY_DIMS.items()
        })

        # Confidence estimator (data sufficiency gate)
        self.confidence_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.head_hidden_dim),
            nn.GELU(),
            nn.Linear(c.head_hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Projection head for contrastive pretraining
        self.projection_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_sequence(
        self,
        clinical_dict: dict[str, torch.Tensor] | None = None,
        mask_dict: dict[str, torch.Tensor] | None = None,
        expression: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Assemble modality tokens + gene tokens into a single sequence.

        Returns:
            tokens: (B, M_total, d)
            modality_mask: (B, M_total) bool — True where present
            n_clinical: number of clinical token slots
        """
        parts_tokens = []
        parts_mask = []

        if clinical_dict is not None and len(clinical_dict) > 0:
            clin_tokens, clin_mask = self.modality_encoder(clinical_dict, mask_dict)
            parts_tokens.append(clin_tokens)
            parts_mask.append(clin_mask)

        n_clinical = parts_tokens[0].shape[1] if parts_tokens else 0

        if expression is not None:
            B = expression.shape[0]
            gene_tokens = self.gene_encoder(expression)  # (B, K, d)
            K = gene_tokens.shape[1]
            gene_mask = torch.ones(B, K, dtype=torch.bool, device=expression.device)
            parts_tokens.append(gene_tokens)
            parts_mask.append(gene_mask)

        if not parts_tokens:
            raise ValueError("At least one input modality required")

        # Ensure consistent batch dim if only one branch
        tokens = torch.cat(parts_tokens, dim=1)
        mask = torch.cat(parts_mask, dim=1)

        return tokens, mask, n_clinical

    def encode(
        self,
        clinical_dict: dict[str, torch.Tensor] | None = None,
        mask_dict: dict[str, torch.Tensor] | None = None,
        expression: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode inputs → [CLS] representation (B, d)."""
        tokens, mask, _ = self._build_sequence(clinical_dict, mask_dict, expression)
        cls_repr, _ = self.transformer(tokens, mask)
        return cls_repr

    def forward(
        self,
        clinical_dict: dict[str, torch.Tensor] | None = None,
        mask_dict: dict[str, torch.Tensor] | None = None,
        expression: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full forward pass: backbone + all reconstructions + confidence.

        Returns dict with keys:
            cls_repr, seq_out, confidence, projection,
            clinical_recon (dict), expression_recon (if expression given),
            disease_logits
        """
        tokens, mask, n_clinical = self._build_sequence(
            clinical_dict, mask_dict, expression,
        )
        cls_repr, seq_out = self.transformer(tokens, mask)

        result: dict[str, Any] = {
            "cls_repr": cls_repr,
            "seq_out": seq_out,
        }

        # Confidence estimate
        result["confidence"] = self.confidence_head(cls_repr).squeeze(-1)

        # Projection for contrastive learning
        proj = self.projection_head(cls_repr)
        result["projection"] = F.normalize(proj, dim=-1)

        # Disease classification
        result["disease_logits"] = self.disease_head(cls_repr)

        # Clinical reconstruction (from modality tokens)
        offset = 1 if self.config.use_cls_token else 0
        modality_names = list(MODALITY_DIMS.keys())
        clinical_recon: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(modality_names):
            if idx < n_clinical:
                tok = seq_out[:, idx + offset]
                clinical_recon[name] = self.clinical_recon_heads[name](tok)
        result["clinical_recon"] = clinical_recon

        # Expression reconstruction (from gene tokens)
        if expression is not None:
            K = self.config.gene_latent_tokens
            gene_start = offset + n_clinical
            gene_tokens = seq_out[:, gene_start:gene_start + K]
            result["expression_recon"] = self.expression_recon_head(gene_tokens)

        return result

    def predict_perturbation(
        self,
        expression: torch.Tensor,
        ptype_ids: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        dose: torch.Tensor,
        duration: torch.Tensor,
        time_point: torch.Tensor,
        clinical_dict: dict[str, torch.Tensor] | None = None,
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict post-perturbation expression via the cVAE head.

        The [CLS] token carries both clinical and molecular context,
        giving the cVAE access to the patient's full state.
        """
        cls_repr = self.encode(clinical_dict, mask_dict, expression)
        return self.perturbation_head(
            cls_repr, ptype_ids, gene_idx, drug_idx,
            dose, duration, time_point,
            training=self.training,
        )

    def compute_perturbation_loss(
        self,
        expression: torch.Tensor,
        target: torch.Tensor,
        ptype_ids: torch.Tensor,
        gene_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        dose: torch.Tensor,
        duration: torch.Tensor,
        time_point: torch.Tensor,
        clinical_dict: dict[str, torch.Tensor] | None = None,
        mask_dict: dict[str, torch.Tensor] | None = None,
        beta: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full β-VAE loss for the perturbation head."""
        beta = beta if beta is not None else self.config.beta
        out = self.predict_perturbation(
            expression, ptype_ids, gene_idx, drug_idx,
            dose, duration, time_point, clinical_dict, mask_dict,
        )

        recon = F.mse_loss(out["predicted"], target)

        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        total = recon + beta * kl
        return {
            "total": total,
            "reconstruction": recon,
            "kl": kl,
            "mu": mu,
            "logvar": logvar,
            "predicted": out["predicted"],
        }

    def should_defer_to_frontier(
        self,
        clinical_dict: dict[str, torch.Tensor] | None = None,
        mask_dict: dict[str, torch.Tensor] | None = None,
        expression: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Data-sufficiency gate: should we call a frontier model?

        Returns (defer_mask (B,) bool, diagnostics dict).
        True = confidence below threshold, defer to frontier.
        """
        with torch.no_grad():
            out = self.forward(clinical_dict, mask_dict, expression)

        confidence = out["confidence"]
        defer = confidence < self.config.confidence_threshold

        n_modalities = 0
        if clinical_dict:
            n_modalities += sum(1 for v in clinical_dict.values() if v is not None)
        if expression is not None:
            n_modalities += 1

        diagnostics = {
            "confidence": confidence,
            "defer_mask": defer,
            "n_modalities": n_modalities,
            "below_min_modalities": n_modalities < self.config.min_modalities_for_foundation,
        }
        return defer, diagnostics

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
