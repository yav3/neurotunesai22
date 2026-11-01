"""Cross-repo synchronization: push NeuroTunesAI results into Merlin's evidence system.

This module writes study records and training artifacts in Merlin's format
so they can be consumed by Merlin's evidence discipline pipeline. It never
modifies Merlin's code — only writes to designated output directories.

Sync targets:
  1. Study records → merlinaineuralplus2/docs/partnerships/records/NT-*.json
  2. Model checkpoints → merlinaineuralplus2/backend/var/neurotunes/
  3. Data registry entries → proposed additions (not auto-committed)

All writes are content-addressed and idempotent.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from models.interface.merlin_bridge import (
    MERLIN_ROOT,
    MERLIN_RECORDS,
    Species,
    format_study_record,
)

MERLIN_VAR = MERLIN_ROOT / "backend" / "var" / "neurotunes"
NT_RECORDS_PREFIX = "NT-"


@dataclass(frozen=True)
class SyncResult:
    target: str
    path: str
    content_hash: str
    written: bool
    reason: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def push_study_record(
    study_id: str,
    title: str,
    question: str,
    data_kind: str,
    species: Species,
    regime: str,
    arms: dict[str, dict[str, float]],
    criteria_results: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> SyncResult:
    """Write a content-addressed study record to Merlin's records directory."""
    if not study_id.startswith(NT_RECORDS_PREFIX):
        study_id = f"{NT_RECORDS_PREFIX}{study_id}"

    record = format_study_record(
        study_id=study_id,
        title=title,
        question=question,
        data_kind=data_kind,
        species=species,
        regime=regime,
        arms=arms,
        criteria_results=criteria_results,
        parameters=parameters,
    )
    record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["source_repo"] = "yav3/NeuroTunesAI"

    content = json.dumps(record, indent=2, sort_keys=True)
    content_hash = _content_hash(content.encode())

    target_path = MERLIN_RECORDS / f"{study_id}.json"

    if target_path.exists():
        existing = json.loads(target_path.read_text())
        if existing.get("content_hash") == record.get("content_hash"):
            return SyncResult(
                target="study_record",
                path=str(target_path),
                content_hash=content_hash,
                written=False,
                reason="identical record already exists",
            )

    _ensure_dir(MERLIN_RECORDS)
    target_path.write_text(content)

    return SyncResult(
        target="study_record",
        path=str(target_path),
        content_hash=content_hash,
        written=True,
        reason="record written",
    )


def push_checkpoint(
    checkpoint_path: Path,
    model_id: str,
    metadata: dict[str, Any] | None = None,
) -> SyncResult:
    """Copy a content-addressed checkpoint to Merlin's var directory."""
    if not checkpoint_path.exists():
        return SyncResult(
            target="checkpoint",
            path=str(checkpoint_path),
            content_hash="",
            written=False,
            reason="source checkpoint not found",
        )

    content_hash = _content_hash(checkpoint_path.read_bytes())
    target_dir = MERLIN_VAR / model_id
    _ensure_dir(target_dir)

    target_path = target_dir / f"{content_hash[:16]}.pt"
    if target_path.exists():
        return SyncResult(
            target="checkpoint",
            path=str(target_path),
            content_hash=content_hash,
            written=False,
            reason="identical checkpoint already exists",
        )

    shutil.copy2(checkpoint_path, target_path)

    if metadata:
        meta_path = target_dir / f"{content_hash[:16]}.meta.json"
        meta = {**metadata, "content_hash": content_hash, "model_id": model_id}
        meta_path.write_text(json.dumps(meta, indent=2))

    return SyncResult(
        target="checkpoint",
        path=str(target_path),
        content_hash=content_hash,
        written=True,
        reason="checkpoint synced",
    )


def push_training_summary(
    study_id: str,
    model_id: str,
    epochs: int,
    train_loss_history: list[float],
    val_loss_history: list[float],
    final_metrics: dict[str, float],
    criteria_met: list[str],
    criteria_failed: list[str],
    species: Species,
    regime: str,
    seed: int,
) -> SyncResult:
    """Write a complete training summary as a study record."""
    arms = {
        "model": final_metrics,
    }
    criteria_results = [
        {"criterion": c, "met": True} for c in criteria_met
    ] + [
        {"criterion": c, "met": False} for c in criteria_failed
    ]
    parameters = {
        "model_id": model_id,
        "epochs": epochs,
        "seed": seed,
        "train_loss_first": train_loss_history[0] if train_loss_history else None,
        "train_loss_last": train_loss_history[-1] if train_loss_history else None,
        "val_loss_first": val_loss_history[0] if val_loss_history else None,
        "val_loss_last": val_loss_history[-1] if val_loss_history else None,
    }

    return push_study_record(
        study_id=study_id,
        title=f"NeuroTunesAI training run: {model_id}",
        question=f"Does {model_id} meet pre-registered criteria?",
        data_kind="real" if regime == "data-driven" else "synthetic",
        species=species,
        regime=regime,
        arms=arms,
        criteria_results=criteria_results,
        parameters=parameters,
    )


def list_synced_records() -> list[str]:
    """List all NT-prefixed records in Merlin's records directory."""
    if not MERLIN_RECORDS.exists():
        return []
    return [
        f.stem for f in MERLIN_RECORDS.glob(f"{NT_RECORDS_PREFIX}*.json")
    ]


def list_synced_checkpoints(model_id: str) -> list[Path]:
    """List all checkpoints for a model in Merlin's var directory."""
    target_dir = MERLIN_VAR / model_id
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("*.pt"))
