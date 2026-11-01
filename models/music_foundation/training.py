"""Training loop for the music foundation model.

Implements SGD-based training with the PretrainingObjective, learning rate
scheduling, gradient clipping, and metric tracking. Pure numpy — no torch dep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .encoder import MusicFoundationModel, EncoderConfig, LinearLayer
from .pretraining import PretrainingObjective, MaskedFeaturePrediction
from .inference import MusicInferencePipeline, InferenceConfig


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    batch_size: int = 32
    warmup_steps: int = 100
    total_steps: int = 1000
    clip_norm: float = 1.0
    contrastive_weight: float = 1.0
    masked_weight: float = 0.5
    alignment_weight: float = 0.5
    mask_ratio: float = 0.15
    log_interval: int = 10
    eval_interval: int = 50
    seed: int = 42


@dataclass
class TrainingMetrics:
    step: int = 0
    epoch: int = 0
    loss_history: list[float] = field(default_factory=list)
    contrastive_history: list[float] = field(default_factory=list)
    masked_history: list[float] = field(default_factory=list)
    alignment_history: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    eval_losses: list[tuple[int, float]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.step,
            "epochs": self.epoch,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "best_loss": min(self.loss_history) if self.loss_history else None,
            "mean_loss_last_10": float(np.mean(self.loss_history[-10:])) if self.loss_history else None,
        }


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def numerical_gradient(
    model: MusicFoundationModel,
    audio_batch: np.ndarray,
    semantic_batch: np.ndarray,
    objective: PretrainingObjective,
    eps: float = 1e-5,
) -> dict[str, np.ndarray]:
    grads = {}
    params = model.params()

    def compute_loss() -> float:
        a_emb = model.encode_audio(audio_batch)
        s_emb = model.encode_semantic(semantic_batch)
        fused = model.bridge.fuse(a_emb, s_emb)
        losses = objective(a_emb, s_emb, fused)
        return losses["total"]

    for name, param in params.items():
        grad = np.zeros_like(param)
        flat = param.flatten()
        n_sample = min(len(flat), 20)
        indices = np.linspace(0, len(flat) - 1, n_sample, dtype=int)

        for idx in indices:
            old = flat[idx]
            flat[idx] = old + eps
            param[:] = flat.reshape(param.shape)
            loss_plus = compute_loss()

            flat[idx] = old - eps
            param[:] = flat.reshape(param.shape)
            loss_minus = compute_loss()

            flat[idx] = old
            param[:] = flat.reshape(param.shape)
            grad.flat[idx] = (loss_plus - loss_minus) / (2 * eps)

        grads[name] = grad
    return grads


def clip_gradients(grads: dict[str, np.ndarray], max_norm: float) -> dict[str, np.ndarray]:
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        return {k: v * scale for k, v in grads.items()}
    return grads


class Trainer:
    def __init__(
        self,
        model: MusicFoundationModel,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.objective = PretrainingObjective(
            contrastive_weight=self.config.contrastive_weight,
            masked_weight=self.config.masked_weight,
            alignment_weight=self.config.alignment_weight,
            mask_ratio=self.config.mask_ratio,
        )
        self.masker = MaskedFeaturePrediction(self.config.mask_ratio, self.config.seed)
        self.metrics = TrainingMetrics()
        self.rng = np.random.RandomState(self.config.seed)
        self._velocity: dict[str, np.ndarray] = {}

    def _get_lr(self) -> float:
        return cosine_lr(
            self.metrics.step,
            self.config.warmup_steps,
            self.config.total_steps,
            self.config.lr,
        )

    def _apply_gradients(self, grads: dict[str, np.ndarray]) -> None:
        lr = self._get_lr()
        clipped = clip_gradients(grads, self.config.clip_norm)
        params = self.model.params()

        for name, grad in clipped.items():
            if name not in self._velocity:
                self._velocity[name] = np.zeros_like(grad)
            self._velocity[name] = 0.9 * self._velocity[name] + grad
            update = lr * (self._velocity[name] + self.config.weight_decay * params[name])
            params[name] -= update

    def train_step(
        self,
        audio_batch: np.ndarray,
        semantic_batch: np.ndarray,
    ) -> dict[str, float]:
        a_emb = self.model.encode_audio(audio_batch)
        s_emb = self.model.encode_semantic(semantic_batch)
        fused = self.model.bridge.fuse(a_emb, s_emb)

        masked_audio, mask = self.masker.mask_features(audio_batch)
        predicted = self.model.encode_audio(masked_audio)

        losses = self.objective(
            a_emb, s_emb, fused,
            original_features=a_emb,
            predicted_features=predicted,
            mask=mask[:, :a_emb.shape[-1]] if mask.shape[-1] >= a_emb.shape[-1] else
                 np.pad(mask, ((0, 0), (0, max(0, a_emb.shape[-1] - mask.shape[-1]))), mode='constant')[:, :a_emb.shape[-1]],
        )

        grads = numerical_gradient(self.model, audio_batch, semantic_batch, self.objective)
        self._apply_gradients(grads)

        self.metrics.step += 1
        self.metrics.loss_history.append(losses["total"])
        self.metrics.contrastive_history.append(losses["contrastive"])
        self.metrics.masked_history.append(losses["masked"])
        self.metrics.alignment_history.append(losses["alignment"])
        self.metrics.lr_history.append(self._get_lr())

        return losses

    def train_epoch(
        self,
        audio_data: np.ndarray,
        semantic_data: np.ndarray,
    ) -> float:
        n = len(audio_data)
        indices = self.rng.permutation(n)
        epoch_losses = []

        for start in range(0, n, self.config.batch_size):
            end = min(start + self.config.batch_size, n)
            batch_idx = indices[start:end]
            losses = self.train_step(audio_data[batch_idx], semantic_data[batch_idx])
            epoch_losses.append(losses["total"])

        self.metrics.epoch += 1
        return float(np.mean(epoch_losses))

    def fit(
        self,
        audio_data: np.ndarray,
        semantic_data: np.ndarray,
        val_audio: np.ndarray | None = None,
        val_semantic: np.ndarray | None = None,
    ) -> TrainingMetrics:
        for epoch in range(self.config.epochs):
            epoch_loss = self.train_epoch(audio_data, semantic_data)

            if val_audio is not None and val_semantic is not None:
                a_emb = self.model.encode_audio(val_audio)
                s_emb = self.model.encode_semantic(val_semantic)
                fused = self.model.bridge.fuse(a_emb, s_emb)
                val_losses = self.objective(a_emb, s_emb, fused)
                self.metrics.eval_losses.append((self.metrics.step, val_losses["total"]))

        return self.metrics
