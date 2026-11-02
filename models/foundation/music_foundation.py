"""MusicFoundation — audio-feature transformer foundation model.

Mirrors GeneFoundation but operates on audio feature vectors (~16 dims)
extracted by AudioSolver / AudioVADEstimator:

1. Learnable feature embeddings (per audio-feature token)
2. Binned value encoding (rank-based, robust to loudness normalisation)
3. Transformer encoder with feature-feature attention
4. [CLS] token → track-level representation

Input: (B, F) feature matrix + (B, F) or (F,) feature indices
       where F ≈ 16 (spectral_centroid, rolloff, bandwidth, zcr,
       rms_energy, dynamic_range, bass/mid/treble_energy, tempo_bpm,
       key, scale, camelot, valence, arousal, dominance)

Pretraining: MaskedFeatureModeling — mask feature tokens, reconstruct values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MusicFoundationConfig:
    """Configuration for the audio-feature foundation encoder."""
    max_features: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    n_value_bins: int = 32
    use_cls_token: bool = True
    feature_embed_dim: int = 128
    mask_ratio: float = 0.15
    seed: int = 2026


class ValueBinner(nn.Module):
    """Bin continuous audio feature values into discrete tokens.

    Rank-based binning within each sample, identical to ExpressionBinner
    but for audio features.  Robust to loudness normalisation and
    platform-specific scaling.
    """

    def __init__(self, n_bins: int, hidden_dim: int):
        super().__init__()
        self.n_bins = n_bins
        self.bin_embed = nn.Embedding(n_bins, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, F) audio feature values
        Returns:
            (B, F, d) bin embeddings per feature
        """
        B, F_dim = features.shape
        ranks = features.argsort(dim=-1).argsort(dim=-1).float()
        bin_ids = (ranks / max(F_dim, 1) * self.n_bins).long().clamp(0, self.n_bins - 1)
        return self.bin_embed(bin_ids)


class TrackFeatureEmbedding(nn.Module):
    """Per-feature learnable embedding + value bin embedding.

    Each audio feature dimension gets a unique learnable vector.
    Final token = feature_embed + bin_embed.
    """

    def __init__(self, config: MusicFoundationConfig):
        super().__init__()
        d = config.hidden_dim
        self.feature_embed = nn.Embedding(config.max_features, d)
        self.binner = ValueBinner(config.n_value_bins, d)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        features: torch.Tensor,
        feature_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, F) audio feature values
            feature_indices: (B, F) or (F,) integer feature IDs in [0, max_features)
        Returns:
            (B, F, d) token embeddings
        """
        f_emb = self.feature_embed(feature_indices)
        v_emb = self.binner(features)
        return self.norm(f_emb + v_emb)


class MusicTransformerBlock(nn.Module):
    """Pre-norm transformer block (shared architecture with gene model)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class MusicFoundationEncoder(nn.Module):
    """Audio-feature transformer encoder.

    Input:  (B, F) feature matrix + (B, F) or (F,) feature indices
    Output: (B, d) track representation  [CLS]
            (B, F, d) per-feature representations  [full sequence]
    """

    def __init__(self, config: MusicFoundationConfig | None = None):
        super().__init__()
        self.config = config or MusicFoundationConfig()
        c = self.config

        self.embedding = TrackFeatureEmbedding(c)

        if c.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, c.hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            MusicTransformerBlock(c.hidden_dim, c.num_heads, c.ffn_dim, c.dropout)
            for _ in range(c.num_layers)
        ])
        self.final_norm = nn.LayerNorm(c.hidden_dim)

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

    def forward(
        self,
        features: torch.Tensor,
        feature_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, F) audio feature values
            feature_indices: (B, F) or (F,) feature ID integers
            padding_mask: (B, F) bool — True where feature is present
        Returns:
            cls_repr: (B, d)
            seq_out: (B, F', d) full sequence (F' = F+1 if CLS)
        """
        tokens = self.embedding(features, feature_indices)
        B, F_dim, d = tokens.shape

        if self.config.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            if padding_mask is not None:
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        attn_mask = ~padding_mask if padding_mask is not None else None

        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=attn_mask)

        tokens = self.final_norm(tokens)

        if self.config.use_cls_token:
            cls_repr = tokens[:, 0]
            seq_out = tokens[:, 1:]
        else:
            if padding_mask is not None:
                lengths = padding_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                cls_repr = (tokens * padding_mask.unsqueeze(-1).float()).sum(1) / lengths
            else:
                cls_repr = tokens.mean(dim=1)
            seq_out = tokens

        return cls_repr, seq_out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MaskedFeatureModeling(nn.Module):
    """Pretraining objective: mask feature tokens, reconstruct values.

    Analogous to MaskedGeneModeling:
      1. Randomly mask `mask_ratio` of feature tokens
      2. Replace with a learnable [MASK] embedding
      3. Predict the original feature value at masked positions
    """

    def __init__(self, encoder: MusicFoundationEncoder):
        super().__init__()
        self.encoder = encoder
        c = encoder.config
        self.mask_token = nn.Parameter(torch.randn(1, 1, c.hidden_dim) * 0.02)
        self.prediction_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, F_dim = features.shape
        c = self.encoder.config

        mask = torch.rand(B, F_dim, device=features.device) < c.mask_ratio
        if padding_mask is not None:
            mask = mask & padding_mask

        masked_feats = features.clone()
        masked_feats[mask] = 0.0

        tokens = self.encoder.embedding(masked_feats, feature_indices)
        tokens[mask] = self.mask_token.squeeze(0).squeeze(0)

        if c.use_cls_token:
            cls = self.encoder.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            if padding_mask is not None:
                cls_m = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)
                full_mask = torch.cat([cls_m, padding_mask], dim=1)
            else:
                full_mask = None
        else:
            full_mask = padding_mask

        attn_mask = ~full_mask if full_mask is not None else None
        for layer in self.encoder.layers:
            tokens = layer(tokens, key_padding_mask=attn_mask)
        tokens = self.encoder.final_norm(tokens)

        if c.use_cls_token:
            feat_tokens = tokens[:, 1:]
        else:
            feat_tokens = tokens

        pred = self.prediction_head(feat_tokens).squeeze(-1)

        loss = F.mse_loss(pred[mask], features[mask])

        return {
            "loss": loss,
            "n_masked": mask.sum().item(),
            "pred": pred,
            "mask": mask,
        }
