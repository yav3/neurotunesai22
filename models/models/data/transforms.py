"""Data augmentation and preprocessing transforms for neurological data."""
from __future__ import annotations

import torch
import numpy as np


class GaussianNoise:
    """Add Gaussian noise to all modality tensors."""
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for k, v in sample.items():
            if k.startswith("_") or k == "label" or not isinstance(v, torch.Tensor):
                out[k] = v
                continue
            if v.is_floating_point():
                out[k] = v + torch.randn_like(v) * self.std
            else:
                out[k] = v
        return out


class RandomModalityDropout:
    """Randomly zero entire modalities during training to improve robustness
    to missing data at inference."""
    def __init__(self, prob: float = 0.15):
        self.prob = prob

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(sample)
        mask = dict(out.get("_mask", {}))
        for k in list(out.keys()):
            if k.startswith("_") or k == "label":
                continue
            if isinstance(out[k], torch.Tensor) and out[k].is_floating_point():
                if torch.rand(1).item() < self.prob:
                    out[k] = torch.zeros_like(out[k])
                    mask[k] = torch.tensor(False)
        out["_mask"] = mask
        return out


class FeatureScaling:
    """Per-feature z-score normalization using precomputed statistics."""
    def __init__(self, stats: dict[str, tuple[np.ndarray, np.ndarray]]):
        self.means = {k: torch.from_numpy(v[0]) for k, v in stats.items()}
        self.stds = {k: torch.from_numpy(v[1]) for k, v in stats.items()}

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(sample)
        for k in self.means:
            if k in out and isinstance(out[k], torch.Tensor) and out[k].is_floating_point():
                out[k] = (out[k] - self.means[k]) / self.stds[k]
        return out


class Compose:
    """Chain multiple transforms."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for t in self.transforms:
            sample = t(sample)
        return sample
