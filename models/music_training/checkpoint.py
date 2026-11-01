"""Checkpoint manager: save/load model weights as numpy .npz, track best metrics."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CheckpointInfo:
    path: str
    epoch: int
    metrics: dict[str, float] = field(default_factory=dict)
    config_hash: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckpointInfo:
        return cls(
            path=d["path"],
            epoch=d["epoch"],
            metrics=d.get("metrics", {}),
            config_hash=d.get("config_hash", ""),
            timestamp=d.get("timestamp", 0.0),
        )


class CheckpointManager:
    def __init__(
        self,
        directory: str | Path,
        metric_name: str = "loss",
        mode: str = "min",
        top_k: int = 3,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        self.top_k = top_k
        self.checkpoints: list[CheckpointInfo] = []
        self._load_manifest()

    def _manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    def _load_manifest(self) -> None:
        mp = self._manifest_path()
        if mp.exists():
            with open(mp) as f:
                data = json.load(f)
            self.checkpoints = [CheckpointInfo.from_dict(d) for d in data.get("checkpoints", [])]

    def _save_manifest(self) -> None:
        data = {"checkpoints": [c.to_dict() for c in self.checkpoints]}
        with open(self._manifest_path(), "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def config_hash(config: dict[str, Any]) -> str:
        s = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    def save(
        self,
        weights: dict[str, np.ndarray],
        epoch: int,
        metrics: dict[str, float],
        config: dict[str, Any] | None = None,
    ) -> CheckpointInfo:
        filename = f"checkpoint_epoch{epoch:04d}.npz"
        filepath = self.directory / filename
        np.savez(filepath, **weights)

        info = CheckpointInfo(
            path=str(filepath),
            epoch=epoch,
            metrics=metrics,
            config_hash=self.config_hash(config) if config else "",
            timestamp=time.time(),
        )
        self.checkpoints.append(info)
        self._prune()
        self._save_manifest()
        return info

    def _metric_value(self, info: CheckpointInfo) -> float:
        return info.metrics.get(self.metric_name, float("inf") if self.mode == "min" else float("-inf"))

    def _prune(self) -> None:
        if len(self.checkpoints) <= self.top_k:
            return
        reverse = self.mode == "max"
        self.checkpoints.sort(key=lambda c: self._metric_value(c), reverse=reverse)
        to_remove = self.checkpoints[self.top_k:]
        self.checkpoints = self.checkpoints[:self.top_k]
        for c in to_remove:
            p = Path(c.path)
            if p.exists():
                p.unlink()

    def load(self, checkpoint: CheckpointInfo | None = None) -> dict[str, np.ndarray]:
        if checkpoint is None:
            checkpoint = self.best()
        if checkpoint is None:
            raise FileNotFoundError("No checkpoints available")
        data = np.load(checkpoint.path)
        return dict(data)

    def best(self) -> CheckpointInfo | None:
        if not self.checkpoints:
            return None
        reverse = self.mode == "max"
        return sorted(self.checkpoints, key=lambda c: self._metric_value(c), reverse=reverse)[0]

    def latest(self) -> CheckpointInfo | None:
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda c: c.epoch)
