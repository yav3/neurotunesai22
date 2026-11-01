"""Batch collation for the multimodal NeuroFoundation datasets."""
from __future__ import annotations

import torch
from models.config import MODALITY_DIMS


class ModalityMaskingCollator:
    """Collates a list of sample dicts into batched tensors.

    Produces ``x_dict`` (modality name → (B, d_m) tensor) and
    ``mask_dict`` (modality name → (B,) bool tensor).
    """

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        B = len(batch)
        modality_names = list(MODALITY_DIMS.keys())

        x_dict: dict[str, torch.Tensor] = {}
        mask_dict: dict[str, torch.Tensor] = {}

        for name in modality_names:
            dim = MODALITY_DIMS[name]
            stacked = torch.zeros(B, dim)
            m = torch.zeros(B, dtype=torch.bool)
            for i, sample in enumerate(batch):
                if name in sample:
                    stacked[i] = sample[name]
                    if "_mask" in sample and name in sample["_mask"]:
                        m[i] = sample["_mask"][name]
                    else:
                        m[i] = True
            x_dict[name] = stacked
            mask_dict[name] = m

        result: dict[str, torch.Tensor] = {"x_dict": x_dict, "mask_dict": mask_dict}

        if "label" in batch[0]:
            result["labels"] = torch.stack([s["label"] for s in batch])

        return result
