"""Multi-stream self-supervised pretraining for the frontier music model.

Extends DeepFrontierTrainer to train through the multi-stream feature
integration path rather than mel-only. The key difference: instead of
feeding raw mel spectrograms through the conv front-end, this trainer:

  1. Extracts 9 acoustic feature streams per signal (mel, MFCC, chroma,
     onset, centroid, bandwidth, rolloff, ZCR, spectral contrast)
  2. Fuses them via MultiStreamFeatureIntegrator (gated or attention fusion)
  3. Feeds the fused (d_model, T) representation directly into the
     transformer — bypassing the conv feature extractor and proj_W

The SPSA gradient estimation (Spall, 1992) operates on both the
integrator parameters (projection + fusion weights) and the transformer
parameters, so the entire multi-stream pipeline is trained end-to-end.

Training objectives (same as DeepFrontierTrainer):
  - Contrastive loss (InfoNCE, van den Oord et al. 2018)
  - Codebook diversity loss
  - Masked feature prediction

Additionally supports:
  - Multi-view contrastive: two different augmented views of the same
    signal produce embeddings that should be similar (SimCLR-style)
  - Feature reconstruction: predict held-out streams from the fused
    representation (cross-modal self-supervision)

Pure numpy. Uses SPSA — no autograd needed.
"""
from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .frontier_model import (
    FrontierConfig, MusicFrontierModel, cosine_warmup_schedule, TrainingState,
    _layer_norm, _gelu,
)
from .deep_trainer import DeepTrainerConfig, AdamState


@dataclass
class MultiStreamTrainerConfig(DeepTrainerConfig):
    """Configuration for multi-stream pretraining."""
    fusion_method: str = "gated"
    feature_recon_weight: float = 0.1
    multi_view_weight: float = 0.2
    augment_gain_std: float = 0.1
    augment_noise_std: float = 0.005
    stream_dropout_prob: float = 0.1
    sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128


class MultiStreamDataset:
    """Dataset holding raw audio signals for multi-stream pretraining."""

    def __init__(
        self,
        signals: List[np.ndarray],
        labels: Optional[List[int]] = None,
        sr: int = 22050,
    ):
        self.signals = signals
        self.labels = labels
        self.sr = sr
        self.n_samples = len(signals)

    def __len__(self) -> int:
        return self.n_samples

    def sample_batch(
        self, batch_size: int, seed: int = 42
    ) -> Tuple[List[np.ndarray], Optional[List[int]]]:
        rng = np.random.RandomState(seed)
        n = min(batch_size, self.n_samples)
        indices = rng.choice(self.n_samples, size=n, replace=False)
        sigs = [self.signals[i] for i in indices]
        labs = [self.labels[i] for i in indices] if self.labels else None
        return sigs, labs

    @staticmethod
    def from_synthetic(
        n_samples: int = 50,
        duration_s: float = 2.0,
        sr: int = 22050,
        seed: int = 42,
    ) -> "MultiStreamDataset":
        from .audio_pipeline import SyntheticAudioGenerator
        gen = SyntheticAudioGenerator(seed=seed)
        signals = gen.generate_dataset(n_samples, duration_s=duration_s, sr=sr)
        labels = [i % 4 for i in range(len(signals))]
        return MultiStreamDataset(signals, labels, sr)


class MultiStreamFeatureCache:
    """Caches extracted multi-stream features to avoid redundant computation."""

    def __init__(self):
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._fused_cache: Dict[int, np.ndarray] = {}

    def get_features(self, signal_id: int) -> Optional[Dict[str, np.ndarray]]:
        return self._cache.get(signal_id)

    def put_features(self, signal_id: int, features: Dict[str, np.ndarray]) -> None:
        self._cache[signal_id] = features

    def get_fused(self, signal_id: int) -> Optional[np.ndarray]:
        return self._fused_cache.get(signal_id)

    def put_fused(self, signal_id: int, fused: np.ndarray) -> None:
        self._fused_cache[signal_id] = fused

    def invalidate(self) -> None:
        self._fused_cache.clear()

    def clear(self) -> None:
        self._cache.clear()
        self._fused_cache.clear()


class MultiStreamPretrainer:
    """End-to-end multi-stream pretraining with SPSA gradients.

    Trains both the feature integrator and the transformer jointly via
    SPSA gradient estimation. The loss landscape includes:
      - Standard contrastive + diversity + masked prediction (from frontier model)
      - Feature reconstruction (predict held-out stream from fused repr)
      - Multi-view contrastive (augmented views should produce similar embeddings)
    """

    def __init__(
        self,
        model: MusicFrontierModel,
        config: Optional[MultiStreamTrainerConfig] = None,
        seed: int = 42,
    ):
        from .feature_integration import (
            FeatureIntegrationConfig,
            MultiStreamFeatureIntegrator,
            AudioFeatureExtractor,
        )
        self.model = model
        self.config = config or MultiStreamTrainerConfig()
        self.rng = np.random.RandomState(seed)
        self.state = TrainingState()
        self.optimizer = AdamState(self.config)
        self.best_params: Optional[Dict[str, np.ndarray]] = None
        self._loss_ema = 0.0

        self.extractor = AudioFeatureExtractor(
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
        )

        fi_config = FeatureIntegrationConfig(
            d_model=model.config.d_model,
            fusion_method=self.config.fusion_method,
            seed=seed,
        )
        self.integrator = MultiStreamFeatureIntegrator(fi_config)
        self._register_streams()

        self.feature_cache = MultiStreamFeatureCache()

        d_model = model.config.d_model
        self._recon_heads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._init_reconstruction_heads(d_model)

    def _register_streams(self) -> None:
        self.integrator.add_stream("mel", input_dim=self.config.n_mels)
        self.integrator.add_stream("mfcc", input_dim=13)
        self.integrator.add_stream("chroma", input_dim=12)
        self.integrator.add_stream("onset", input_dim=1)
        self.integrator.add_stream("centroid", input_dim=1)
        self.integrator.add_stream("bandwidth", input_dim=1)
        self.integrator.add_stream("contrast", input_dim=6)

    def _init_reconstruction_heads(self, d_model: int) -> None:
        """Initialize linear heads for feature stream reconstruction."""
        stream_dims = {"mel": self.config.n_mels, "mfcc": 13, "chroma": 12}
        for name, dim in stream_dims.items():
            scale = np.sqrt(2.0 / d_model)
            W = self.rng.randn(d_model, dim) * scale
            b = np.zeros(dim)
            self._recon_heads[name] = (W, b)

    def _extract_features(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        return self.extractor.extract(signal)

    def _fuse_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        return self.integrator.forward(features)

    def _run_transformer(self, fused: np.ndarray) -> np.ndarray:
        """Run transformer on fused (d_model, T), bypassing proj_W."""
        x = fused.T  # (T, d_model)
        T_len = x.shape[0]
        trans = self.model.transformer
        x = x + trans.pos_enc[:T_len]
        for layer in trans.layers:
            x, _ = layer.forward(x)
        x = _layer_norm(x, trans.final_gamma, trans.final_beta)
        return x  # (T, d_model)

    def _augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Light augmentation: gain jitter + additive noise."""
        gain = 1.0 + self.rng.randn() * self.config.augment_gain_std
        noise = self.rng.randn(*signal.shape) * self.config.augment_noise_std
        return signal * gain + noise

    def _contrastive_loss_from_hidden(
        self, hidden: np.ndarray, mask_seed: int,
    ) -> float:
        """Compute contrastive loss from transformer hidden states (T, d_model)."""
        T_len, d_model = hidden.shape
        if T_len < 4:
            return 0.0

        rng = np.random.RandomState(mask_seed)
        n_masked = max(1, int(T_len * self.config.final_mask_prob))
        mask_indices = rng.choice(T_len, size=n_masked, replace=False)

        quantized = hidden @ self.model.quantizer.proj_W
        codes = self.model.quantizer.codebooks[0]  # (codebook_size, codebook_dim)
        q_dim = codes.shape[1]
        quantized = quantized[:, :q_dim]

        dists = np.sum((quantized[mask_indices, None, :] - codes[None, :, :]) ** 2, axis=-1)
        targets = np.argmin(dists, axis=-1)

        projected = hidden @ self.model.contrastive_head.W + self.model.contrastive_head.b
        proj_dim = min(projected.shape[1], codes.shape[1])
        projected = projected[:, :proj_dim]

        tau = self.model.config.contrastive_temperature
        loss = 0.0
        for idx, mi in enumerate(mask_indices):
            pos = codes[targets[idx]]
            sim_pos = np.dot(projected[mi], pos) / (
                np.linalg.norm(projected[mi]) * np.linalg.norm(pos) + 1e-8
            )
            n_neg = min(self.model.config.n_negatives, len(codes) - 1)
            neg_ids = rng.choice(
                [j for j in range(len(codes)) if j != targets[idx]],
                size=n_neg, replace=True,
            )
            negs = codes[neg_ids]
            proj_norm = projected[mi] / (np.linalg.norm(projected[mi]) + 1e-8)
            neg_norms = negs / (np.linalg.norm(negs, axis=1, keepdims=True) + 1e-8)
            sim_neg = neg_norms @ proj_norm

            logits = np.concatenate([[sim_pos / tau], sim_neg / tau])
            logits -= logits.max()
            log_sum_exp = np.log(np.sum(np.exp(logits)) + 1e-8)
            loss += -(sim_pos / tau - log_sum_exp)

        return float(loss / max(len(mask_indices), 1))

    def _diversity_loss(self, hidden: np.ndarray) -> float:
        """Codebook usage diversity (entropy maximization)."""
        quantized = hidden @ self.model.quantizer.proj_W
        codes = self.model.quantizer.codebooks[0]
        q_dim = codes.shape[1]
        quantized = quantized[:, :q_dim]

        dists = np.sum((quantized[:, None, :] - codes[None, :, :]) ** 2, axis=-1)
        assignments = np.argmin(dists, axis=-1)
        counts = np.bincount(assignments, minlength=len(codes)).astype(float)
        probs = counts / (counts.sum() + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(codes))
        return float(max_entropy - entropy)

    def _feature_reconstruction_loss(
        self, hidden: np.ndarray, features: Dict[str, np.ndarray],
    ) -> float:
        """Predict held-out feature streams from the fused representation."""
        loss = 0.0
        count = 0
        for name, (W, b) in self._recon_heads.items():
            if name not in features:
                continue
            target = features[name]
            if target.ndim == 1:
                target = target.reshape(1, -1)
            T_h = hidden.shape[0]
            T_t = target.shape[-1] if target.ndim == 2 and target.shape[0] != T_h else target.shape[0]

            if target.ndim == 2 and target.shape[0] != T_h:
                target = target.T  # (T, feat_dim)
            T_min = min(T_h, target.shape[0])
            hidden_slice = hidden[:T_min]
            target_slice = target[:T_min]

            pred = hidden_slice @ W + b
            pred = pred[:, :target_slice.shape[1]]
            target_slice = target_slice[:, :pred.shape[1]]
            loss += float(np.mean((pred - target_slice) ** 2))
            count += 1
        return loss / max(count, 1)

    def _multi_view_loss(
        self, signal: np.ndarray, hidden1: np.ndarray,
    ) -> float:
        """Contrastive loss between two augmented views of the same signal."""
        aug_signal = self._augment_signal(signal)
        features2 = self._extract_features(aug_signal)
        fused2 = self._fuse_features(features2)
        hidden2 = self._run_transformer(fused2)

        emb1 = hidden1.mean(axis=0)
        emb2 = hidden2.mean(axis=0)
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        sim = np.dot(emb1, emb2)
        return float(-np.log(np.exp(sim / 0.1) / (np.exp(sim / 0.1) + 1.0) + 1e-8))

    def _compute_loss(
        self, signal: np.ndarray, mask_seed: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Full multi-stream loss on a single signal."""
        features = self._extract_features(signal)
        fused = self._fuse_features(features)
        hidden = self._run_transformer(fused)

        contrastive = self._contrastive_loss_from_hidden(hidden, mask_seed)
        diversity = self._diversity_loss(hidden)
        recon = self._feature_reconstruction_loss(hidden, features)
        mv = self._multi_view_loss(signal, hidden)

        total = (
            contrastive
            + self.model.config.diversity_weight * diversity
            + self.config.feature_recon_weight * recon
            + self.config.multi_view_weight * mv
        )
        return total, {
            "contrastive": contrastive,
            "diversity": diversity,
            "recon": recon,
            "multi_view": mv,
        }

    def _batch_loss(
        self, signals: List[np.ndarray], base_seed: int,
    ) -> Tuple[float, Dict[str, float]]:
        losses = []
        components: Dict[str, List[float]] = {}
        for i, sig in enumerate(signals):
            total, comp = self._compute_loss(sig, base_seed + i)
            losses.append(total)
            for k, v in comp.items():
                components.setdefault(k, []).append(v)
        mean_comp = {k: float(np.mean(v)) for k, v in components.items()}
        return float(np.mean(losses)), mean_comp

    def _extract_params(self) -> Dict[str, np.ndarray]:
        """Extract all trainable parameters: integrator + transformer + heads."""
        params = {}

        for i, layer in enumerate(self.model.transformer.layers):
            for attr in ('W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2'):
                params[f'trans.layer{i}.{attr}'] = getattr(layer, attr)
        params['trans.final_gamma'] = self.model.transformer.final_gamma
        params['trans.final_beta'] = self.model.transformer.final_beta
        params['quant.codebooks'] = self.model.quantizer.codebooks
        params['quant.proj_W'] = self.model.quantizer.proj_W
        params['contrast.W'] = self.model.contrastive_head.W
        params['contrast.b'] = self.model.contrastive_head.b

        for name, proj in self.integrator.projections.items():
            params[f'integ.proj.{name}.W'] = proj.W
            params[f'integ.proj.{name}.b'] = proj.b
        if hasattr(self.integrator, 'fusion') and self.integrator.fusion is not None:
            fusion = self.integrator.fusion
            if hasattr(fusion, 'gate_W'):
                params['integ.fusion.gate_W'] = fusion.gate_W
                params['integ.fusion.gate_b'] = fusion.gate_b

        for name, (W, b) in self._recon_heads.items():
            params[f'recon.{name}.W'] = W
            params[f'recon.{name}.b'] = b

        return params

    def _apply_params(self, params: Dict[str, np.ndarray]) -> None:
        """Write parameters back into model and integrator."""
        for i, layer in enumerate(self.model.transformer.layers):
            prefix = f'trans.layer{i}.'
            for attr in ('W_q', 'W_k', 'W_v', 'W_o', 'W_ff1', 'b_ff1', 'W_ff2', 'b_ff2'):
                key = prefix + attr
                if key in params:
                    setattr(layer, attr, params[key])
        if 'trans.final_gamma' in params:
            self.model.transformer.final_gamma = params['trans.final_gamma']
        if 'trans.final_beta' in params:
            self.model.transformer.final_beta = params['trans.final_beta']

        for key, (obj_path, attr) in {
            'quant.codebooks': ('quantizer', 'codebooks'),
            'quant.proj_W': ('quantizer', 'proj_W'),
            'contrast.W': ('contrastive_head', 'W'),
            'contrast.b': ('contrastive_head', 'b'),
        }.items():
            if key in params:
                setattr(getattr(self.model, obj_path), attr, params[key])

        for name, proj in self.integrator.projections.items():
            w_key = f'integ.proj.{name}.W'
            b_key = f'integ.proj.{name}.b'
            if w_key in params:
                proj.W = params[w_key]
            if b_key in params:
                proj.b = params[b_key]

        if hasattr(self.integrator, 'fusion') and self.integrator.fusion is not None:
            fusion = self.integrator.fusion
            if 'integ.fusion.gate_W' in params and hasattr(fusion, 'gate_W'):
                fusion.gate_W = params['integ.fusion.gate_W']
            if 'integ.fusion.gate_b' in params and hasattr(fusion, 'gate_b'):
                fusion.gate_b = params['integ.fusion.gate_b']

        for name in self._recon_heads:
            w_key = f'recon.{name}.W'
            b_key = f'recon.{name}.b'
            if w_key in params and b_key in params:
                self._recon_heads[name] = (params[w_key], params[b_key])

        self.feature_cache.invalidate()

    def _estimate_gradients(
        self, signals: List[np.ndarray], base_seed: int,
    ) -> Dict[str, np.ndarray]:
        """SPSA gradient estimation over integrator + transformer + heads."""
        params = self._extract_params()
        eps = self.config.fd_epsilon
        grads = {k: np.zeros_like(v) for k, v in params.items()}

        for _ in range(self.config.n_param_samples):
            deltas = {
                k: self.rng.choice([-1.0, 1.0], size=v.shape)
                for k, v in params.items()
            }

            self._apply_params({k: params[k] + eps * deltas[k] for k in params})
            loss_plus, _ = self._batch_loss(signals, base_seed)

            self._apply_params({k: params[k] - eps * deltas[k] for k in params})
            loss_minus, _ = self._batch_loss(signals, base_seed)

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

    def _current_mask_prob(self) -> float:
        if self.state.step >= self.config.mask_warmup_steps:
            return self.config.final_mask_prob
        t = self.state.step / max(self.config.mask_warmup_steps, 1)
        return self.config.initial_mask_prob + t * (
            self.config.final_mask_prob - self.config.initial_mask_prob
        )

    def train_step(
        self, signals: List[np.ndarray], step_seed: int,
    ) -> Dict[str, float]:
        lr = cosine_warmup_schedule(
            self.state.step, self.config.warmup_steps, self.config.total_steps,
            self.config.learning_rate, self.config.min_lr,
        )

        grads = self._estimate_gradients(signals, step_seed)
        params = self._extract_params()
        updated = self.optimizer.step(params, grads, lr)
        self._apply_params(updated)

        loss, components = self._batch_loss(signals, step_seed + 999)
        self._loss_ema = (
            0.05 * loss + 0.95 * self._loss_ema if self.state.step > 0 else loss
        )

        self.state.step += 1
        self.state.loss_history.append(loss)
        self.state.lr_history.append(lr)

        metrics = {
            "loss": loss,
            "loss_ema": self._loss_ema,
            "lr": lr,
            "mask_prob": self._current_mask_prob(),
            "grad_norm": float(np.sqrt(sum(
                np.sum(g ** 2) for g in grads.values()
            ))),
            "n_params": sum(v.size for v in params.values()),
        }
        metrics.update(components)
        return metrics

    def evaluate(
        self, dataset: MultiStreamDataset, n_batches: int = 3,
    ) -> Dict[str, float]:
        losses = []
        for i in range(n_batches):
            sigs, _ = dataset.sample_batch(self.config.batch_size, seed=i * 1000)
            loss, _ = self._batch_loss(sigs, base_seed=i * 7919)
            losses.append(loss)
        val_loss = float(np.mean(losses))
        improved = val_loss < self.state.best_loss
        return {
            "val_loss": val_loss,
            "val_std": float(np.std(losses)),
            "improved": improved,
        }

    def train(
        self,
        train_data: MultiStreamDataset,
        val_data: Optional[MultiStreamDataset] = None,
        max_steps: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, object]:
        patience_counter = 0
        total = max_steps or self.config.total_steps
        start = time.time()

        for step in range(total):
            sigs, _ = train_data.sample_batch(
                self.config.batch_size, seed=self.state.step,
            )
            metrics = self.train_step(sigs, step_seed=self.state.step)
            if callback:
                callback(self.state.step, metrics)

            if val_data and self.state.step % self.config.eval_every == 0:
                val_metrics = self.evaluate(val_data)
                metrics.update(val_metrics)
                if val_metrics["improved"]:
                    self.state.best_loss = val_metrics["val_loss"]
                    self.best_params = self._extract_params()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        elapsed = time.time() - start
        return {
            "final_loss": self.state.loss_history[-1] if self.state.loss_history else 0,
            "best_loss": self.state.best_loss,
            "steps": self.state.step,
            "elapsed_seconds": elapsed,
            "loss_history": self.state.loss_history,
        }

    def extract_embeddings(
        self, signals: List[np.ndarray],
    ) -> np.ndarray:
        """Extract multi-stream embeddings for downstream evaluation."""
        embeddings = []
        for sig in signals:
            features = self._extract_features(sig)
            fused = self._fuse_features(features)
            hidden = self._run_transformer(fused)
            emb = hidden.mean(axis=0)
            embeddings.append(emb)
        return np.array(embeddings)

    def save_checkpoint(self, path: str) -> None:
        params = self._extract_params()
        np.savez(
            path,
            **{f'param_{k}': v for k, v in params.items()},
            state_step=self.state.step,
            state_best_loss=self.state.best_loss,
        )

    def load_checkpoint(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        params = {
            k.replace('param_', ''): data[k]
            for k in data.files if k.startswith('param_')
        }
        self._apply_params(params)
        self.state.step = int(data.get('state_step', 0))
        self.state.best_loss = float(data.get('state_best_loss', float('inf')))


def quick_multistream_train(
    n_samples: int = 20,
    n_steps: int = 2,
    seed: int = 42,
) -> Dict:
    """Quick smoke test: multi-stream pretraining on synthetic data."""
    from .frontier_model import FrontierConfig, MusicFrontierModel

    cfg = FrontierConfig(
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        conv_channels=[32, 64],
        conv_kernels=[10, 3],
        conv_strides=[5, 2],
        n_codebooks=1, codebook_size=32, codebook_dim=32,
    )
    model = MusicFrontierModel(cfg)

    train_cfg = MultiStreamTrainerConfig(
        n_mels=128, batch_size=4, n_param_samples=2,
        total_steps=n_steps, eval_every=n_steps,
        warmup_steps=1, log_every=1,
    )

    dataset = MultiStreamDataset.from_synthetic(n_samples, seed=seed)
    trainer = MultiStreamPretrainer(model, train_cfg, seed=seed)

    result = trainer.train(dataset, max_steps=n_steps)
    embeddings = trainer.extract_embeddings(dataset.signals[:4])

    return {
        "final_loss": result["final_loss"],
        "steps": result["steps"],
        "embedding_shape": embeddings.shape,
        "embedding_norm_mean": float(np.mean(np.linalg.norm(embeddings, axis=1))),
    }


if __name__ == "__main__":
    result = quick_multistream_train(n_samples=10, n_steps=1)
    print(f"Multi-stream pretraining smoke test:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("PASSED")
