"""Deep training loop for the frontier music foundation model.

Implements the full self-supervised pretraining pipeline:
  - Data batching and augmentation integration
  - Adam optimizer with weight decay (Loshchilov & Hutter, 2019)
  - Cosine warmup learning rate schedule
  - Gradient estimation via SPSA (Spall, 1992) — no autograd needed
  - Validation, checkpointing, and early stopping
  - Curriculum learning: mask ratio annealing
  - Multi-objective loss balancing (contrastive + diversity + prediction)

Pure numpy. Uses stochastic parameter perturbation for gradient estimation.
Suitable for architecture validation and small-scale experiments.
"""
from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .frontier_model import (
    FrontierConfig, MusicFrontierModel, cosine_warmup_schedule, TrainingState,
)


@dataclass
class DeepTrainerConfig:
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    warmup_steps: int = 500
    total_steps: int = 50000

    batch_size: int = 8
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 50
    patience: int = 10

    initial_mask_prob: float = 0.02
    final_mask_prob: float = 0.065
    mask_warmup_steps: int = 5000

    fd_epsilon: float = 1e-4
    n_param_samples: int = 64


class DeepAudioDataset:
    """In-memory audio dataset for pretraining."""

    def __init__(
        self,
        spectrograms: List[np.ndarray],
        labels: Optional[List[int]] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.metadata = metadata or [{}] * len(spectrograms)
        self.n_samples = len(spectrograms)

    def __len__(self) -> int:
        return self.n_samples

    def sample_batch(
        self, batch_size: int, seed: int = 42
    ) -> Tuple[List[np.ndarray], Optional[List[int]]]:
        rng = np.random.RandomState(seed)
        indices = rng.choice(self.n_samples, size=min(batch_size, self.n_samples), replace=False)
        specs = [self.spectrograms[i] for i in indices]
        labels = [self.labels[i] for i in indices] if self.labels else None
        return specs, labels

    @staticmethod
    def from_signals(
        signals: List[np.ndarray],
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        sr: int = 22050,
        labels: Optional[List[int]] = None,
    ) -> "DeepAudioDataset":
        from .frontier_model import _simple_mel_spectrogram
        specs = [_simple_mel_spectrogram(sig, n_mels, n_fft, hop_length, sr) for sig in signals]
        return DeepAudioDataset(specs, labels)


class AdamState:
    """Adam optimizer state (Kingma & Ba, 2015)."""

    def __init__(self, config: DeepTrainerConfig):
        self.config = config
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0

    def step(
        self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float
    ) -> Dict[str, np.ndarray]:
        self.t += 1
        updated = {}

        for name, param in params.items():
            if name not in grads:
                updated[name] = param
                continue

            g = grads[name]

            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            self.m[name] = self.config.beta1 * self.m[name] + (1 - self.config.beta1) * g
            self.v[name] = self.config.beta2 * self.v[name] + (1 - self.config.beta2) * g ** 2

            m_hat = self.m[name] / (1 - self.config.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.config.beta2 ** self.t)

            update = lr * m_hat / (np.sqrt(v_hat) + self.config.eps)
            if self.config.weight_decay > 0:
                update += lr * self.config.weight_decay * param

            updated[name] = param - update

        return updated


class DeepFrontierTrainer:
    """Training loop for the deep MusicFrontierModel.

    Uses SPSA (Spall, 1992) for gradient estimation — only 2 forward
    passes per perturbation sample, independent of parameter count.
    """

    def __init__(
        self,
        model: MusicFrontierModel,
        config: Optional[DeepTrainerConfig] = None,
        seed: int = 42,
    ):
        self.model = model
        self.config = config or DeepTrainerConfig()
        self.rng = np.random.RandomState(seed)
        self.state = TrainingState()
        self.optimizer = AdamState(self.config)
        self.best_model_params: Optional[Dict[str, np.ndarray]] = None
        self._loss_ema = 0.0

    def _current_mask_prob(self) -> float:
        if self.state.step >= self.config.mask_warmup_steps:
            return self.config.final_mask_prob
        t = self.state.step / max(self.config.mask_warmup_steps, 1)
        return self.config.initial_mask_prob + t * (
            self.config.final_mask_prob - self.config.initial_mask_prob
        )

    def _compute_loss(self, spectrogram: np.ndarray, mask_seed: int) -> float:
        old_mp = self.model.config.mask_prob
        self.model.config.mask_prob = self._current_mask_prob()
        result = self.model.pretrain_forward(spectrogram, mask_seed=mask_seed)
        self.model.config.mask_prob = old_mp
        return result['total_loss']

    def _batch_loss(self, specs: List[np.ndarray], base_seed: int) -> float:
        return float(np.mean([
            self._compute_loss(s, base_seed + i) for i, s in enumerate(specs)
        ]))

    def _extract_params(self) -> Dict[str, np.ndarray]:
        params = {}
        for i, layer in enumerate(self.model.transformer.layers):
            for attr in ('W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2'):
                params[f'layer{i}.{attr}'] = getattr(layer, attr)
        params['proj_W'] = self.model.transformer.proj_W
        params['proj_b'] = self.model.transformer.proj_b
        params['codebooks'] = self.model.quantizer.codebooks
        params['quant_proj'] = self.model.quantizer.proj_W
        params['contrast_W'] = self.model.contrastive_head.W
        params['contrast_b'] = self.model.contrastive_head.b
        return params

    def _apply_params(self, params: Dict[str, np.ndarray]) -> None:
        for i, layer in enumerate(self.model.transformer.layers):
            prefix = f'layer{i}.'
            for attr in ('W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2'):
                key = prefix + attr
                if key in params:
                    setattr(layer, attr, params[key])
        for key, (obj_attr, param_attr) in {
            'proj_W': ('transformer', 'proj_W'),
            'proj_b': ('transformer', 'proj_b'),
            'codebooks': ('quantizer', 'codebooks'),
            'quant_proj': ('quantizer', 'proj_W'),
            'contrast_W': ('contrastive_head', 'W'),
            'contrast_b': ('contrastive_head', 'b'),
        }.items():
            if key in params:
                setattr(getattr(self.model, obj_attr), param_attr, params[key])

    def _estimate_gradients(
        self, specs: List[np.ndarray], base_seed: int,
    ) -> Dict[str, np.ndarray]:
        """SPSA gradient estimation (Spall, 1992)."""
        params = self._extract_params()
        eps = self.config.fd_epsilon
        grads = {k: np.zeros_like(v) for k, v in params.items()}

        for _ in range(self.config.n_param_samples):
            deltas = {k: self.rng.choice([-1.0, 1.0], size=v.shape) for k, v in params.items()}

            self._apply_params({k: params[k] + eps * deltas[k] for k in params})
            loss_plus = self._batch_loss(specs, base_seed)

            self._apply_params({k: params[k] - eps * deltas[k] for k in params})
            loss_minus = self._batch_loss(specs, base_seed)

            self._apply_params(params)

            for k in grads:
                grads[k] += (loss_plus - loss_minus) / (2 * eps * deltas[k])

        for k in grads:
            grads[k] /= self.config.n_param_samples

        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if total_norm > self.config.max_grad_norm:
            scale = self.config.max_grad_norm / (total_norm + 1e-8)
            grads = {k: v * scale for k, v in grads.items()}

        return grads

    def train_step(
        self, specs: List[np.ndarray], step_seed: int
    ) -> Dict[str, float]:
        lr = cosine_warmup_schedule(
            self.state.step, self.config.warmup_steps, self.config.total_steps,
            self.config.learning_rate, self.config.min_lr,
        )

        grads = self._estimate_gradients(specs, step_seed)
        params = self._extract_params()
        updated = self.optimizer.step(params, grads, lr)
        self._apply_params(updated)

        loss = self._batch_loss(specs, step_seed + 999)
        self._loss_ema = 0.05 * loss + 0.95 * self._loss_ema if self.state.step > 0 else loss

        self.state.step += 1
        self.state.loss_history.append(loss)
        self.state.lr_history.append(lr)

        return {
            'loss': loss, 'loss_ema': self._loss_ema, 'lr': lr,
            'mask_prob': self._current_mask_prob(),
            'grad_norm': float(np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))),
        }

    def evaluate(self, dataset: DeepAudioDataset, n_batches: int = 5) -> Dict[str, float]:
        losses = []
        for i in range(n_batches):
            specs, _ = dataset.sample_batch(self.config.batch_size, seed=i * 1000)
            losses.append(self._batch_loss(specs, base_seed=i * 7919))
        val_loss = float(np.mean(losses))
        return {'val_loss': val_loss, 'val_std': float(np.std(losses)), 'improved': val_loss < self.state.best_loss}

    def train(
        self, train_data: DeepAudioDataset, val_data: Optional[DeepAudioDataset] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, object]:
        patience_counter = 0
        start_time = time.time()

        for step in range(self.config.total_steps):
            specs, _ = train_data.sample_batch(self.config.batch_size, seed=self.state.step)
            metrics = self.train_step(specs, step_seed=self.state.step)
            if callback:
                callback(self.state.step, metrics)

            if val_data and self.state.step % self.config.eval_every == 0:
                val_metrics = self.evaluate(val_data)
                metrics.update(val_metrics)
                if val_metrics['improved']:
                    self.state.best_loss = val_metrics['val_loss']
                    self.best_model_params = self._extract_params()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        elapsed = time.time() - start_time
        return {
            'final_loss': self.state.loss_history[-1] if self.state.loss_history else 0,
            'best_loss': self.state.best_loss,
            'steps': self.state.step,
            'elapsed_seconds': elapsed,
            'loss_history': self.state.loss_history,
        }

    def save_checkpoint(self, path: str) -> None:
        params = self._extract_params()
        np.savez(path, **{f'param_{k}': v for k, v in params.items()},
                 state_step=self.state.step, state_best_loss=self.state.best_loss)

    def load_checkpoint(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        params = {k.replace('param_', ''): data[k] for k in data.files if k.startswith('param_')}
        self._apply_params(params)
        self.state.step = int(data.get('state_step', 0))
        self.state.best_loss = float(data.get('state_best_loss', float('inf')))
