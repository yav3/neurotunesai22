"""NeuroFoundation — multimodal transformer foundation model for
neurological disease state modeling.

Architecture
------------
1. Per-modality linear encoders → shared d-dimensional tokens
2. Cross-modal transformer with [CLS] aggregation
3. Reconstruction heads for each modality (pretraining)
4. Patient-level CLS embedding for downstream tasks
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.config import FoundationConfig, MODALITY_DIMS
from models.foundation.encoders import MultiModalEncoder
from models.foundation.cross_attention import CrossModalTransformer


class ReconstructionHead(nn.Module):
    """Decodes a modality token back to original feature space."""
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class NeuroFoundation(nn.Module):
    """
    Forward pass modes:
        encode:      x_dict → cls_repr (B, d)
        pretrain:    x_dict, mask_dict → loss terms + reconstructions
        full:        x_dict → cls_repr, seq_out, reconstructions
    """

    def __init__(self, config: FoundationConfig | None = None):
        super().__init__()
        self.config = config or FoundationConfig()
        c = self.config

        self.encoder = MultiModalEncoder(c.hidden_dim, c.dropout)
        self.transformer = CrossModalTransformer(
            c.hidden_dim, c.num_layers, c.num_heads, c.ffn_dim,
            c.dropout, c.activation, c.use_cls_token,
        )

        self.recon_heads = nn.ModuleDict({
            name: ReconstructionHead(c.hidden_dim, dim)
            for name, dim in MODALITY_DIMS.items()
        })

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

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        tokens, modality_mask = self.encoder(x_dict, mask_dict)
        cls_repr, _ = self.transformer(tokens, modality_mask)
        return cls_repr

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens, modality_mask = self.encoder(x_dict, mask_dict)
        cls_repr, seq_out = self.transformer(tokens, modality_mask)

        offset = 1 if self.config.use_cls_token else 0
        modality_names = list(MODALITY_DIMS.keys())

        reconstructions: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(modality_names):
            tok = seq_out[:, idx + offset]
            reconstructions[name] = self.recon_heads[name](tok)

        proj = self.projection_head(cls_repr)
        proj = nn.functional.normalize(proj, dim=-1)

        return {
            "cls_repr": cls_repr,
            "seq_out": seq_out,
            "projection": proj,
            "reconstructions": reconstructions,
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
