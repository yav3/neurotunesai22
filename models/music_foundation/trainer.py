"""Training loop: ties model, optimizer, scheduler, objectives, evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .encoder import MusicFoundationModel, EncoderConfig
from .pretraining import PretrainingObjective
from .optimizer import Adam, GradientClipper
from .gradient import numerical_gradient


@dataclass
class TrainerConfig:
    lr: float = 1e-4
    epochs: int = 100
    batch_size: int = 32
    warmup_steps: int = 100
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 50
    patience: int = 10
    min_delta: float = 1e-4


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.counter = 0

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False
        improved = (
            metric < self.best - self.min_delta if self.mode == "min"
            else metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def reset(self) -> None:
        self.best = None
        self.counter = 0


class TrainingHistory:
    def __init__(self) -> None:
        self.step_losses: list[dict[str, float]] = []
        self.epoch_losses: list[dict[str, float]] = []
        self.val_metrics: list[dict[str, float]] = []

    def log_step(self, losses: dict[str, float]) -> None:
        self.step_losses.append(losses)

    def log_epoch(self, losses: dict[str, float]) -> None:
        self.epoch_losses.append(losses)

    def log_val(self, metrics: dict[str, float]) -> None:
        self.val_metrics.append(metrics)

    def summary(self) -> dict[str, Any]:
        return {
            "total_steps": len(self.step_losses),
            "total_epochs": len(self.epoch_losses),
            "final_train_loss": self.epoch_losses[-1].get("total", 0.0) if self.epoch_losses else None,
            "final_val_loss": self.val_metrics[-1].get("total", 0.0) if self.val_metrics else None,
            "best_epoch": self.best_epoch(),
        }

    def best_epoch(self, metric: str = "total") -> int:
        if not self.val_metrics:
            return 0
        vals = [m.get(metric, float("inf")) for m in self.val_metrics]
        return int(np.argmin(vals))


class MusicTrainer:
    def __init__(
        self,
        model: MusicFoundationModel | None = None,
        config: TrainerConfig | None = None,
        objective: PretrainingObjective | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.model = model or MusicFoundationModel()
        self.objective = objective or PretrainingObjective()
        self.optimizer = Adam(lr=self.config.lr)
        self.clipper = GradientClipper(max_norm=self.config.grad_clip)
        self.early_stopping = EarlyStopping(
            patience=self.config.patience, min_delta=self.config.min_delta
        )
        self.history = TrainingHistory()
        self._global_step = 0

    def _compute_loss(
        self,
        params: dict[str, np.ndarray],
        audio_batch: np.ndarray,
        semantic_batch: np.ndarray,
    ) -> float:
        audio_embs = self.model.encode_audio(audio_batch)
        semantic_embs = self.model.encode_semantic(semantic_batch)
        fused = self.model.bridge.fuse(audio_embs, semantic_embs)
        losses = self.objective(audio_embs, semantic_embs, fused)
        return losses["total"]

    def train_step(
        self,
        audio_batch: np.ndarray,
        semantic_batch: np.ndarray,
    ) -> dict[str, float]:
        audio_embs = self.model.encode_audio(audio_batch)
        semantic_embs = self.model.encode_semantic(semantic_batch)
        fused = self.model.bridge.fuse(audio_embs, semantic_embs)
        losses = self.objective(audio_embs, semantic_embs, fused)

        params = self.model.params()

        def loss_fn(p: dict[str, np.ndarray]) -> float:
            return self._compute_loss(p, audio_batch, semantic_batch)

        grads = numerical_gradient(loss_fn, params, eps=1e-4)
        grads = self.clipper.clip(grads)
        params = self.optimizer.step(params, grads)

        self._global_step += 1
        self.history.log_step(losses)
        return losses

    def eval_step(
        self,
        audio_batch: np.ndarray,
        semantic_batch: np.ndarray,
    ) -> dict[str, float]:
        audio_embs = self.model.encode_audio(audio_batch)
        semantic_embs = self.model.encode_semantic(semantic_batch)
        fused = self.model.bridge.fuse(audio_embs, semantic_embs)
        return self.objective(audio_embs, semantic_embs, fused)

    def train_epoch(
        self,
        audio_data: np.ndarray,
        semantic_data: np.ndarray,
    ) -> dict[str, float]:
        n = len(audio_data)
        bs = self.config.batch_size
        epoch_losses: list[dict[str, float]] = []
        indices = np.random.permutation(n)
        for start in range(0, n, bs):
            idx = indices[start : start + bs]
            losses = self.train_step(audio_data[idx], semantic_data[idx])
            epoch_losses.append(losses)
        avg = {}
        if epoch_losses:
            for key in epoch_losses[0]:
                avg[key] = float(np.mean([l[key] for l in epoch_losses]))
        self.history.log_epoch(avg)
        return avg

    def evaluate(
        self,
        audio_data: np.ndarray,
        semantic_data: np.ndarray,
    ) -> dict[str, float]:
        n = len(audio_data)
        bs = self.config.batch_size
        all_metrics: list[dict[str, float]] = []
        for start in range(0, n, bs):
            metrics = self.eval_step(audio_data[start : start + bs], semantic_data[start : start + bs])
            all_metrics.append(metrics)
        avg = {}
        if all_metrics:
            for key in all_metrics[0]:
                avg[key] = float(np.mean([m[key] for m in all_metrics]))
        self.history.log_val(avg)
        return avg

    def fit(
        self,
        train_audio: np.ndarray,
        train_semantic: np.ndarray,
        val_audio: np.ndarray | None = None,
        val_semantic: np.ndarray | None = None,
        epochs: int | None = None,
    ) -> TrainingHistory:
        n_epochs = epochs or self.config.epochs
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_audio, train_semantic)
            if val_audio is not None and val_semantic is not None:
                if (epoch + 1) % max(self.config.eval_every, 1) == 0 or epoch == n_epochs - 1:
                    val_metrics = self.evaluate(val_audio, val_semantic)
                    if self.early_stopping(val_metrics.get("total", 0.0)):
                        break
        return self.history

    def training_state(self) -> dict[str, Any]:
        return {
            "global_step": self._global_step,
            "optimizer": self.optimizer.state_dict(),
            "history_summary": self.history.summary(),
        }
