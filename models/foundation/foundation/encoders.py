"""Modality-specific encoders: project each biomarker group into the shared
embedding space d_model and add a learnable modality-type embedding."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.config import MODALITY_DIMS


class ModalityEncoder(nn.Module):
    """Linear projection → LayerNorm → GELU → residual-ready token for one
    modality group.  Input dim is the raw feature count of that modality;
    output dim is ``hidden_dim``."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultiModalEncoder(nn.Module):
    """One ``ModalityEncoder`` per modality group plus learnable modality-type
    embeddings that are added to the projected tokens.

    Forward pass: ``x_dict`` maps modality names to (B, d_m) tensors.
    Returns (B, M, d_model) where M = number of observed modalities.
    Also returns a boolean mask (B, M) — True where the modality is present.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modality_names = list(MODALITY_DIMS.keys())

        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dim, hidden_dim, dropout)
            for name, dim in MODALITY_DIMS.items()
        })
        self.type_embeddings = nn.Embedding(len(self.modality_names), hidden_dim)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = next(iter(x_dict.values())).shape[0]
        device = next(iter(x_dict.values())).device

        tokens = torch.zeros(B, len(self.modality_names), self.hidden_dim, device=device)
        modality_mask = torch.zeros(B, len(self.modality_names), dtype=torch.bool, device=device)

        for idx, name in enumerate(self.modality_names):
            if name not in x_dict:
                continue
            h = self.encoders[name](x_dict[name])
            h = h + self.type_embeddings(torch.tensor(idx, device=device))
            tokens[:, idx] = h

            if mask_dict is not None and name in mask_dict:
                modality_mask[:, idx] = mask_dict[name]
            else:
                modality_mask[:, idx] = True

        return tokens, modality_mask
