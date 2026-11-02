"""GeneFoundation — gene-level transformer foundation model.

Unlike NeuroFoundation (40 clinical features, 9 modality groups), this model
operates directly on gene expression vectors (5K–20K genes) with:

1. Learnable gene embeddings (per-gene token)
2. Binned expression encoding (rank-based, robust to batch effects)
3. Transformer encoder with gene-gene attention
4. [CLS] token → sample-level representation

Architecture matches scGPT/Geneformer patterns scaled to the data regime
the NT trials have shown works: 100–1000 samples, 2K–10K genes, governed
by the 4-criterion protocol.

Operates in log1p space.  Species firewall enforced at data loading, not here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeneFoundationConfig:
    """Configuration for the gene-level foundation encoder."""
    max_genes: int = 10_000
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    n_expression_bins: int = 64
    use_cls_token: bool = True
    gene_embed_dim: int = 256
    max_position: int = 20_000
    mask_ratio: float = 0.15
    seed: int = 2026


class ExpressionBinner(nn.Module):
    """Bin continuous expression values into discrete tokens.

    Rank-based binning is robust to batch effects and platform differences.
    For each sample independently:
      1. Rank genes by expression value
      2. Map rank → bin index in [0, n_bins)
      3. Look up learned bin embedding
    """

    def __init__(self, n_bins: int, hidden_dim: int):
        super().__init__()
        self.n_bins = n_bins
        self.bin_embed = nn.Embedding(n_bins, hidden_dim)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expression: (B, G) log1p expression values
        Returns:
            (B, G, d) bin embeddings per gene
        """
        B, G = expression.shape
        ranks = expression.argsort(dim=-1).argsort(dim=-1).float()
        bin_ids = (ranks / G * self.n_bins).long().clamp(0, self.n_bins - 1)
        return self.bin_embed(bin_ids)


class GeneEmbedding(nn.Module):
    """Per-gene learnable embedding + expression bin embedding.

    Each gene gets a unique learnable vector (like a word embedding in NLP).
    The final token = gene_embed + bin_embed + position_embed.
    """

    def __init__(self, config: GeneFoundationConfig):
        super().__init__()
        d = config.hidden_dim
        self.gene_embed = nn.Embedding(config.max_position, d)
        self.binner = ExpressionBinner(config.n_expression_bins, d)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        expression: torch.Tensor,
        gene_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            expression: (B, G) log1p expression values
            gene_indices: (B, G) or (G,) integer gene IDs in [0, max_position)
        Returns:
            (B, G, d) token embeddings
        """
        g_emb = self.gene_embed(gene_indices)
        b_emb = self.binner(expression)
        return self.norm(g_emb + b_emb)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

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


class GeneFoundationEncoder(nn.Module):
    """Gene-level transformer encoder.

    Input: (B, G) expression matrix + (B, G) or (G,) gene indices
    Output: (B, d) sample representation  [CLS]
            (B, G, d) per-gene representations  [full sequence]
    """

    def __init__(self, config: GeneFoundationConfig | None = None):
        super().__init__()
        self.config = config or GeneFoundationConfig()
        c = self.config

        self.embedding = GeneEmbedding(c)

        if c.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, c.hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(c.hidden_dim, c.num_heads, c.ffn_dim, c.dropout)
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
        expression: torch.Tensor,
        gene_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            expression: (B, G) log1p expression
            gene_indices: (B, G) or (G,) gene ID integers
            padding_mask: (B, G) bool — True where gene is present
        Returns:
            cls_repr: (B, d)
            seq_out: (B, G', d) full sequence (G' = G+1 if CLS)
        """
        tokens = self.embedding(expression, gene_indices)
        B, G, d = tokens.shape

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


class MaskedGeneModeling(nn.Module):
    """Pretraining objective: mask gene tokens, reconstruct expression.

    Like MLM in BERT, but for gene expression values:
      1. Randomly mask `mask_ratio` of gene tokens
      2. Replace with a learnable [MASK] embedding
      3. Predict the original expression value at masked positions
    """

    def __init__(self, encoder: GeneFoundationEncoder):
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
        expression: torch.Tensor,
        gene_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, G = expression.shape
        c = self.encoder.config

        mask = torch.rand(B, G, device=expression.device) < c.mask_ratio
        if padding_mask is not None:
            mask = mask & padding_mask

        masked_expr = expression.clone()
        masked_expr[mask] = 0.0

        tokens = self.encoder.embedding(masked_expr, gene_indices)
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
            gene_tokens = tokens[:, 1:]
        else:
            gene_tokens = tokens

        pred = self.prediction_head(gene_tokens).squeeze(-1)

        loss = F.mse_loss(pred[mask], expression[mask])

        return {
            "loss": loss,
            "n_masked": mask.sum().item(),
            "pred": pred,
            "mask": mask,
        }
