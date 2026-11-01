"""Disease state classification head: maps CLS embedding → 14-class logits."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import NUM_DISEASE_STATES, FinetuneConfig
from models.foundation.neurofoundation import NeuroFoundation


class DiseaseClassifier(nn.Module):
    """Two-layer MLP on top of the frozen/finetuned foundation encoder."""

    def __init__(
        self,
        foundation: NeuroFoundation,
        config: FinetuneConfig | None = None,
    ):
        super().__init__()
        self.foundation = foundation
        c = config or FinetuneConfig()
        d = foundation.config.hidden_dim

        self.head = nn.Sequential(
            nn.Linear(d, c.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(c.head_dropout),
            nn.Linear(c.head_hidden_dim, NUM_DISEASE_STATES),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cls_repr = self.foundation.encode(x_dict, mask_dict)
        return self.head(cls_repr)

    def loss(
        self,
        x_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.forward(x_dict, mask_dict)
        ce = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        return {"loss": ce, "accuracy": acc, "logits": logits}
