"""Model checkpoint: save/load MusicFoundationModel parameters to disk."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .encoder import MusicFoundationModel, EncoderConfig


def save_checkpoint(
    model: MusicFoundationModel,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params = model.params()
    arrays = {k: v.tolist() for k, v in params.items()}

    checkpoint = {
        "config": {
            "audio_dim": model.config.audio_dim,
            "semantic_dim": model.config.semantic_dim,
            "hidden_dim": model.config.hidden_dim,
            "latent_dim": model.config.latent_dim,
            "n_heads": model.config.n_heads,
            "dropout": model.config.dropout,
            "alpha": model.config.alpha,
        },
        "params": arrays,
        "metadata": metadata or {},
        "timestamp": time.time(),
    }

    with open(path, "w") as f:
        json.dump(checkpoint, f)

    return path


def load_checkpoint(path: str | Path) -> tuple[MusicFoundationModel, dict[str, Any]]:
    path = Path(path)
    with open(path) as f:
        checkpoint = json.load(f)

    cfg = EncoderConfig(**checkpoint["config"])
    model = MusicFoundationModel(cfg)

    params = model.params()
    for k, v in checkpoint["params"].items():
        if k in params:
            arr = np.array(v)
            if arr.shape == params[k].shape:
                params[k][:] = arr

    return model, checkpoint.get("metadata", {})


def save_npz(model: MusicFoundationModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = model.params()
    np.savez_compressed(str(path), **params)
    return path


def load_npz(model: MusicFoundationModel, path: str | Path) -> None:
    path = Path(path)
    data = np.load(str(path))
    params = model.params()
    for k in data.files:
        if k in params and data[k].shape == params[k].shape:
            params[k][:] = data[k]
