"""Checkpoint management for the hybrid model.

Save and load model weights, training config, and governed trial records
with hash-chained integrity verification.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from models.training.hybrid_trainer import TrialRecord


CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"


def _hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """SHA-256 of the serialised model weights."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode())
        h.update(state_dict[key].cpu().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(
    model: torch.nn.Module,
    record: TrialRecord,
    config: dict[str, Any] | None = None,
    tag: str = "",
    checkpoint_dir: Path | None = None,
) -> Path:
    """Save model weights + trial record atomically.

    Returns the checkpoint path.
    """
    out_dir = checkpoint_dir or CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    name = record.study_id
    if tag:
        name = f"{name}_{tag}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{name}_{ts}"

    ckpt_path = out_dir / f"{name}.pt"
    record_path = out_dir / f"{name}_record.json"

    state = model.state_dict()
    weights_hash = _hash_state_dict(state)

    payload = {
        "model_state_dict": state,
        "weights_hash": weights_hash,
        "study_id": record.study_id,
        "content_hash": record.content_hash,
        "config": config or {},
    }
    torch.save(payload, ckpt_path)

    record_dict = asdict(record)
    record_dict["weights_hash"] = weights_hash
    record_dict["checkpoint_path"] = str(ckpt_path)
    with open(record_path, "w") as f:
        json.dump(record_dict, f, indent=2, default=str)

    return ckpt_path


def load_checkpoint(
    ckpt_path: str | Path,
    model: torch.nn.Module,
    verify: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load model weights from checkpoint.

    Returns (model, metadata) where metadata includes weights_hash,
    study_id, content_hash, and config.
    """
    ckpt_path = Path(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model.load_state_dict(payload["model_state_dict"])

    if verify:
        actual_hash = _hash_state_dict(model.state_dict())
        if actual_hash != payload["weights_hash"]:
            raise ValueError(
                f"Weights hash mismatch: expected {payload['weights_hash']}, "
                f"got {actual_hash}"
            )

    metadata = {
        "weights_hash": payload["weights_hash"],
        "study_id": payload.get("study_id", ""),
        "content_hash": payload.get("content_hash", ""),
        "config": payload.get("config", {}),
    }
    return model, metadata


def load_trial_record(record_path: str | Path) -> TrialRecord:
    """Load a trial record from JSON."""
    with open(record_path) as f:
        data = json.load(f)

    data.pop("weights_hash", None)
    data.pop("checkpoint_path", None)
    return TrialRecord(**data)
