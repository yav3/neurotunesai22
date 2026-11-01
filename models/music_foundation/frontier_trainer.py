"""Training loop for MusicFrontierModel.

Multi-objective training: reconstruction (MSE) + VQ commitment + diffusion (noise prediction).
Cosine LR with warmup, momentum SGD, sparse numerical gradients, early stopping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import os

import numpy as np

from .frontier import MusicFrontierModel, FrontierConfig


@dataclass
class FrontierTrainerConfig:
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 8
    warmup_steps: int = 50
    total_steps: int = 1000
    clip_norm: float = 1.0
    weight_decay: float = 1e-4
    momentum: float = 0.9
    recon_weight: float = 1.0
    commitment_weight: float = 0.25
    diffusion_weight: float = 1.0
    save_every: int = 100
    eval_every: int = 10
    patience: int = 5
    min_delta: float = 1e-4
    seed: int = 42
    sr: int = 22050
    perturb_per_param: int = 8


def cosine_schedule(step: int, warmup: int, total: int,
                    base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))


class AudioDataset:
    def __init__(self, data: list[np.ndarray] | np.ndarray,
                 segment_length: int = 22050 * 5,
                 seed: int = 42) -> None:
        self.segment_length = segment_length
        self.rng = np.random.RandomState(seed)
        self.segments: list[np.ndarray] = []
        if isinstance(data, np.ndarray) and data.ndim == 2:
            for row in data:
                self._add_audio(row)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            self._add_audio(data)
        else:
            for item in data:
                self._add_audio(np.asarray(item))

    def _add_audio(self, audio: np.ndarray) -> None:
        n = len(audio)
        if n <= self.segment_length:
            seg = np.zeros(self.segment_length)
            seg[:n] = audio
            self.segments.append(seg)
        else:
            start = 0
            while start + self.segment_length <= n:
                self.segments.append(audio[start:start + self.segment_length].copy())
                start += self.segment_length
            if start < n:
                seg = np.zeros(self.segment_length)
                seg[:n - start] = audio[start:]
                self.segments.append(seg)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.segments[idx]

    def batch_iterator(self, batch_size: int, shuffle: bool = True):
        n = len(self.segments)
        indices = self.rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield np.array([self.segments[i] for i in batch_idx])


class FrontierTrainer:
    def __init__(self, model: MusicFrontierModel,
                 config: FrontierTrainerConfig | None = None) -> None:
        self.model = model
        self.config = config or FrontierTrainerConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self._step = 0
        self._velocity: dict[str, np.ndarray] = {}
        self.train_losses: list[dict[str, float]] = []
        self.val_losses: list[dict[str, float]] = []
        self._best_val: float | None = None
        self._patience_counter = 0

    def _lr(self) -> float:
        return cosine_schedule(self._step, self.config.warmup_steps,
                               self.config.total_steps, self.config.lr)

    def _compute_losses(self, audio_batch: np.ndarray) -> dict[str, float]:
        batch_losses: list[dict[str, float]] = []
        for i in range(len(audio_batch)):
            losses = self.model.training_loss(
                audio_batch[i].astype(np.float64), sr=self.config.sr
            )
            batch_losses.append(losses)
        avg = {}
        for key in batch_losses[0]:
            avg[key] = float(np.mean([l[key] for l in batch_losses]))
        return avg

    def _sparse_gradient(self, audio_batch: np.ndarray,
                         eps: float = 1e-4) -> dict[str, np.ndarray]:
        params = self.model.params()
        grads = {}
        for name, param in params.items():
            grad = np.zeros_like(param)
            flat = param.flatten()
            n_perturb = min(len(flat), self.config.perturb_per_param)
            indices = np.linspace(0, len(flat) - 1, n_perturb, dtype=int)
            for idx in indices:
                old = flat[idx]
                flat[idx] = old + eps
                param[:] = flat.reshape(param.shape)
                loss_plus = self._compute_losses(audio_batch)["total"]
                flat[idx] = old - eps
                param[:] = flat.reshape(param.shape)
                loss_minus = self._compute_losses(audio_batch)["total"]
                flat[idx] = old
                param[:] = flat.reshape(param.shape)
                grad.flat[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            grads[name] = grad
        return grads

    def _clip_gradients(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > self.config.clip_norm:
            scale = self.config.clip_norm / (total_norm + 1e-12)
            return {k: v * scale for k, v in grads.items()}
        return grads

    def _apply_gradients(self, grads: dict[str, np.ndarray]) -> None:
        lr = self._lr()
        clipped = self._clip_gradients(grads)
        params = self.model.params()
        for name, grad in clipped.items():
            if name not in params:
                continue
            if name not in self._velocity:
                self._velocity[name] = np.zeros_like(grad)
            self._velocity[name] = (self.config.momentum * self._velocity[name] + grad)
            update = lr * (self._velocity[name] + self.config.weight_decay * params[name])
            params[name] -= update

    def train_step(self, audio_batch: np.ndarray) -> dict[str, float]:
        losses = self._compute_losses(audio_batch)
        grads = self._sparse_gradient(audio_batch)
        self._apply_gradients(grads)
        self._step += 1
        self.train_losses.append(losses)
        return losses

    def train_epoch(self, dataset: AudioDataset) -> dict[str, float]:
        epoch_losses: list[dict[str, float]] = []
        for batch in dataset.batch_iterator(self.config.batch_size):
            losses = self.train_step(batch)
            epoch_losses.append(losses)
        avg = {}
        if epoch_losses:
            for key in epoch_losses[0]:
                avg[key] = float(np.mean([l[key] for l in epoch_losses]))
        return avg

    def evaluate(self, dataset: AudioDataset) -> dict[str, float]:
        all_losses: list[dict[str, float]] = []
        for batch in dataset.batch_iterator(self.config.batch_size, shuffle=False):
            losses = self._compute_losses(batch)
            all_losses.append(losses)
        avg = {}
        if all_losses:
            for key in all_losses[0]:
                avg[key] = float(np.mean([l[key] for l in all_losses]))
        self.val_losses.append(avg)
        return avg

    def _check_early_stop(self, val_loss: float) -> bool:
        if self._best_val is None or val_loss < self._best_val - self.config.min_delta:
            self._best_val = val_loss
            self._patience_counter = 0
            return False
        self._patience_counter += 1
        return self._patience_counter >= self.config.patience

    def fit(self, train_data: AudioDataset,
            val_data: AudioDataset | None = None) -> dict[str, Any]:
        epoch_summaries = []
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_data)
            summary = {"epoch": epoch, "train": train_loss}
            if val_data is not None and (epoch + 1) % max(self.config.eval_every, 1) == 0:
                val_loss = self.evaluate(val_data)
                summary["val"] = val_loss
                if self._check_early_stop(val_loss.get("total", float("inf"))):
                    summary["early_stop"] = True
                    epoch_summaries.append(summary)
                    break
            epoch_summaries.append(summary)
        return {
            "epochs_run": len(epoch_summaries),
            "final_train_loss": epoch_summaries[-1]["train"] if epoch_summaries else {},
            "history": epoch_summaries,
        }

    def save_state(self, path: str) -> None:
        params = self.model.params()
        save_dict = {f"param_{k}": v for k, v in params.items()}
        save_dict["_step"] = np.array([self._step])
        for k, v in self._velocity.items():
            save_dict[f"velocity_{k}"] = v
        np.savez(path, **save_dict)

    def load_state(self, path: str) -> None:
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path)
        params = self.model.params()
        for k in params:
            key = f"param_{k}"
            if key in data:
                params[k][:] = data[key]
        if "_step" in data:
            self._step = int(data["_step"][0])
        for key in data.files:
            if key.startswith("velocity_"):
                vk = key[len("velocity_"):]
                self._velocity[vk] = data[key]
