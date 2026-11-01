"""Treatment response prediction: given patient state + drug features,
predict the expected treatment effect on target biomarkers.

Architecture: CLS ⊕ drug_embedding → MLP → predicted effect size.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import FinetuneConfig
from models.foundation.neurofoundation import NeuroFoundation

NUM_DRUG_FEATURES = 8  # PK/PD summary features per drug


class TreatmentResponsePredictor(nn.Module):
    def __init__(
        self,
        foundation: NeuroFoundation,
        num_drugs: int = 11,
        config: FinetuneConfig | None = None,
    ):
        super().__init__()
        self.foundation = foundation
        c = config or FinetuneConfig()
        d = foundation.config.hidden_dim

        self.drug_encoder = nn.Sequential(
            nn.Linear(NUM_DRUG_FEATURES, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d, c.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(c.head_dropout),
            nn.Linear(c.head_hidden_dim, 1),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        drug_features: torch.Tensor,
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cls_repr = self.foundation.encode(x_dict, mask_dict)
        drug_emb = self.drug_encoder(drug_features)
        combined = torch.cat([cls_repr, drug_emb], dim=-1)
        return self.head(combined).squeeze(-1)

    def loss(
        self,
        x_dict: dict[str, torch.Tensor],
        drug_features: torch.Tensor,
        targets: torch.Tensor,
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        preds = self.forward(x_dict, drug_features, mask_dict)
        mse = F.mse_loss(preds, targets)
        return {"loss": mse, "predictions": preds}
