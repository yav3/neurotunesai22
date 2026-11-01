"""Cross-modal transformer encoder block.

Standard pre-norm transformer encoder that operates on the sequence of
modality tokens.  Supports a key-padding mask so that missing modalities
are ignored in the attention computation.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            act,
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


class CrossModalTransformer(nn.Module):
    """Stack of ``TransformerBlock`` layers with a prepended [CLS] token."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1, activation: str = "gelu",
                 use_cls: bool = True):
        super().__init__()
        self.use_cls = use_cls
        if use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout, activation)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor,
                modality_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, M, d) modality token embeddings.
            modality_mask: (B, M) bool — True where modality is present.
        Returns:
            cls_repr: (B, d) patient-level representation (CLS pooling or mean).
            seq_out:  (B, M', d) full sequence output including CLS if used.
        """
        B = tokens.shape[0]

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)
            modality_mask = torch.cat([cls_mask, modality_mask], dim=1)

        # MultiheadAttention key_padding_mask: True → ignore
        attn_mask = ~modality_mask

        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=attn_mask)

        tokens = self.final_norm(tokens)

        if self.use_cls:
            cls_repr = tokens[:, 0]
        else:
            lengths = modality_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            cls_repr = (tokens * modality_mask.unsqueeze(-1).float()).sum(dim=1) / lengths

        return cls_repr, tokens
