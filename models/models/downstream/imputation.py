"""Missing modality imputation via cross-attention decoding.

Given observed modalities, reconstruct the missing ones using the
transformer's cross-modal representations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import MODALITY_DIMS, FinetuneConfig
from models.foundation.neurofoundation import NeuroFoundation


class ModalityImputer(nn.Module):
    """Uses the full sequence output (not just CLS) to impute each missing
    modality via a per-modality decoder head."""

    def __init__(
        self,
        foundation: NeuroFoundation,
        config: FinetuneConfig | None = None,
    ):
        super().__init__()
        self.foundation = foundation
        c = config or FinetuneConfig()
        d = foundation.config.hidden_dim
        self.modality_names = list(MODALITY_DIMS.keys())

        self.decoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d, c.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(c.head_dropout),
                nn.Linear(c.head_hidden_dim, dim),
            )
            for name, dim in MODALITY_DIMS.items()
        })

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.foundation(x_dict, mask_dict)
        seq = out["seq_out"]
        offset = 1 if self.foundation.config.use_cls_token else 0

        imputed = {}
        for idx, name in enumerate(self.modality_names):
            tok = seq[:, idx + offset]
            imputed[name] = self.decoders[name](tok)
        return imputed

    def loss(
        self,
        x_dict: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        imputed = self.forward(x_dict, mask_dict)
        total = torch.tensor(0.0, device=next(iter(imputed.values())).device)
        count = 0
        per_modality: dict[str, torch.Tensor] = {}
        for name in imputed:
            if name not in targets:
                continue
            if mask_dict and name in mask_dict:
                absent = ~mask_dict[name]
                if absent.any():
                    mod_loss = F.mse_loss(imputed[name][absent], targets[name][absent])
                    total = total + mod_loss
                    per_modality[name] = mod_loss.detach()
                    count += 1
        total = total / max(count, 1)
        return {"loss": total, "per_modality": per_modality, "imputed": imputed}
