"""Pretraining objectives for NeuroFoundation.

Three self-supervised losses:
1. Masked Modality Modeling (MMM) — mask entire modality groups, reconstruct
2. Masked Feature Modeling (MFM) — mask random individual features, reconstruct
3. Cross-Modal Contrastive — align representations of the same patient under
   different modality subsets (InfoNCE)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import PretrainConfig, MODALITY_DIMS
from models.foundation.neurofoundation import NeuroFoundation


def _mask_modalities(
    x_dict: dict[str, torch.Tensor],
    prob: float,
    rng: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Zero-out entire modalities with probability ``prob``.
    Returns (masked_x, mask_dict, targets)."""
    masked = {}
    masks = {}
    targets = {}
    for name, x in x_dict.items():
        B = x.shape[0]
        keep = torch.rand(B, 1, generator=rng, device=x.device) > prob
        masked[name] = x * keep.float()
        masks[name] = keep.squeeze(-1)
        targets[name] = x
    return masked, masks, targets


def _mask_features(
    x_dict: dict[str, torch.Tensor],
    prob: float,
    noise_std: float = 0.0,
    rng: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Per-feature masking: replace masked features with Gaussian noise."""
    masked = {}
    feature_masks = {}
    for name, x in x_dict.items():
        mask = torch.rand_like(x) > prob
        noise = torch.randn_like(x) * noise_std
        masked[name] = torch.where(mask, x, noise)
        feature_masks[name] = mask
    return masked, feature_masks


class PretrainingObjective(nn.Module):
    """Wraps NeuroFoundation with the three pretraining losses."""

    def __init__(self, model: NeuroFoundation, config: PretrainConfig | None = None):
        super().__init__()
        self.model = model
        self.config = config or PretrainConfig()

    def _reconstruction_loss(
        self,
        recon: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(recon.values())).device)
        count = 0
        for name in recon:
            if name not in targets:
                continue
            # Only penalize reconstructions of masked (absent) modalities
            absent = ~masks.get(name, torch.ones(recon[name].shape[0],
                                dtype=torch.bool, device=recon[name].device))
            if absent.any():
                loss = loss + F.mse_loss(
                    recon[name][absent], targets[name][absent],
                )
                count += 1
        return loss / max(count, 1)

    def _contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Symmetric InfoNCE between two views of the same patients."""
        B = z1.shape[0]
        logits = z1 @ z2.T / temperature
        labels = torch.arange(B, device=z1.device)
        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.T, labels)
        return (loss_12 + loss_21) / 2

    def forward(
        self, x_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        c = self.config

        # === MMM ===
        masked_x, mod_masks, targets = _mask_modalities(x_dict, c.mask_modality_prob)
        out_mmm = self.model(masked_x, mod_masks)
        mmm_loss = self._reconstruction_loss(out_mmm["reconstructions"], targets, mod_masks)

        # === MFM ===
        feat_masked, feat_masks = _mask_features(x_dict, c.mask_feature_prob, c.noise_std)
        out_mfm = self.model(feat_masked)
        mfm_loss = torch.tensor(0.0, device=mmm_loss.device)
        mfm_count = 0
        for name in out_mfm["reconstructions"]:
            if name not in feat_masks:
                continue
            absent_feats = ~feat_masks[name]
            if absent_feats.any():
                mfm_loss = mfm_loss + F.mse_loss(
                    out_mfm["reconstructions"][name][absent_feats],
                    x_dict[name][absent_feats],
                )
                mfm_count += 1
        mfm_loss = mfm_loss / max(mfm_count, 1)

        # === Contrastive ===
        view1_x, view1_masks, _ = _mask_modalities(x_dict, c.mask_modality_prob)
        view2_x, view2_masks, _ = _mask_modalities(x_dict, c.mask_modality_prob)
        z1 = self.model(view1_x, view1_masks)["projection"]
        z2 = self.model(view2_x, view2_masks)["projection"]
        contrastive_loss = self._contrastive_loss(z1, z2, c.contrastive_temperature)

        w = c.loss_weights
        total = (w["mmm"] * mmm_loss
                 + w["mfm"] * mfm_loss
                 + w["contrastive"] * contrastive_loss)

        return {
            "loss": total,
            "mmm_loss": mmm_loss.detach(),
            "mfm_loss": mfm_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
        }
