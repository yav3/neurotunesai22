"""Helper functions for model loading, device management, and reproducibility."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from models.config import FoundationConfig
from models.foundation.neurofoundation import NeuroFoundation


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_foundation(
    checkpoint_path: str | Path,
    config: FoundationConfig | None = None,
    device: str | torch.device | None = None,
) -> NeuroFoundation:
    config = config or FoundationConfig()
    model = NeuroFoundation(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cleaned = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
               for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    device = device or get_device()
    return model.to(device)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
