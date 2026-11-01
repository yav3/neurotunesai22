"""Disease progression prediction: given baseline biomarkers + time horizon,
predict future biomarker values.

Uses the CLS embedding concatenated with a time embedding as input to an MLP
that outputs predicted biomarker deltas.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import TOTAL_FEATURES, FinetuneConfig
from models.foundation.neurofoundation import NeuroFoundation


class TimeEmbedding(nn.Module):
    """Sinusoidal + learned time encoding (months → d-dim vector)."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(1, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.norm(self.linear(t.float()))


class ProgressionPredictor(nn.Module):
    """Predicts Δ-biomarker vector at a future time t given baseline CLS."""

    def __init__(
        self,
        foundation: NeuroFoundation,
        config: FinetuneConfig | None = None,
    ):
        super().__init__()
        self.foundation = foundation
        c = config or FinetuneConfig()
        d = foundation.config.hidden_dim

        self.time_emb = TimeEmbedding(d)
        self.head = nn.Sequential(
            nn.Linear(2 * d, c.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(c.head_dropout),
            nn.Linear(c.head_hidden_dim, c.head_hidden_dim),
            nn.GELU(),
            nn.Linear(c.head_hidden_dim, TOTAL_FEATURES),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        time_months: torch.Tensor,
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cls_repr = self.foundation.encode(x_dict, mask_dict)
        t_emb = self.time_emb(time_months)
        combined = torch.cat([cls_repr, t_emb], dim=-1)
        return self.head(combined)

    def loss(
        self,
        x_dict: dict[str, torch.Tensor],
        time_months: torch.Tensor,
        targets: torch.Tensor,
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        preds = self.forward(x_dict, time_months, mask_dict)
        valid = ~torch.isnan(targets)
        if valid.any():
            mse = F.mse_loss(preds[valid], targets[valid])
        else:
            mse = torch.tensor(0.0, device=preds.device)
        return {"loss": mse, "predictions": preds}
