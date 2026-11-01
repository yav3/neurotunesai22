"""End-to-end frontier model pipeline.

Wires the deep transformer frontier model (frontier_model.py) to the
data infrastructure (audio_pipeline.py, augmentation.py) and evaluation
(evaluation.py) into a single, runnable pretraining + fine-tuning pipeline.

Architecture:
    Raw audio → mel spectrogram → conv feature extractor → transformer
    → quantization → contrastive + diversity + masked-prediction losses
    → SPSA gradient estimation → AdamW → checkpoint/eval loop

This is the "glue" that makes the frontier model trainable end-to-end
on real or synthetic audio data.  Pure numpy/scipy.

References:
    - Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised
      Learning of Speech Representations.
    - Li et al. (2023). MERT: Acoustic Music Understanding Model with
      Large-Scale Self-Supervised Training.
    - Spall (1992). Multivariate stochastic approximation using a
      simultaneous perturbation gradient approximation.
"""
from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .frontier_model import (
    FrontierConfig,
    MusicFrontierModel,
    cosine_warmup_schedule,
    TrainingState,
    batch_spectrograms,
    generate_span_mask,
    contrastive_loss,
    diversity_loss,
    masked_prediction_loss,
)
from .deep_trainer import DeepTrainerConfig, DeepAudioDataset, AdamState, DeepFrontierTrainer
from .audio_pipeline import AudioPipelineConfig, AudioPreprocessor, SyntheticAudioGenerator
from .augmentation import (
    two_view_augmentation,
    default_pretraining_chain,
    spec_augment,
    frequency_warp,
    AugResult,
)


@dataclass
class FrontierPipelineConfig:
    """Configuration for the full frontier training pipeline."""

    # Audio
    sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    segment_duration_s: float = 5.0

    # Model architecture
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    conv_kernels: List[int] = field(default_factory=lambda: [10, 3, 3])
    conv_strides: List[int] = field(default_factory=lambda: [5, 2, 2])
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 1024
    n_codebooks: int = 2
    codebook_size: int = 128
    codebook_dim: int = 64

    # Masking
    mask_prob: float = 0.065
    mask_length: int = 10
    contrastive_temperature: float = 0.1
    n_negatives: int = 100

    # Loss weights (wav2vec 2.0 defaults)
    diversity_weight: float = 0.1
    n_classes: int = 10

    # Training
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 500
    total_steps: int = 50000
    batch_size: int = 8
    eval_every: int = 500
    save_every: int = 2000
    log_every: int = 50
    patience: int = 10
    max_grad_norm: float = 1.0

    # SPSA
    fd_epsilon: float = 1e-4
    n_param_samples: int = 32

    # Curriculum
    initial_mask_prob: float = 0.02
    final_mask_prob: float = 0.065
    mask_warmup_steps: int = 5000

    # Augmentation
    augment_prob: float = 0.5
    spec_augment_prob: float = 0.3
    freq_warp_prob: float = 0.2

    seed: int = 42

    def to_frontier_config(self) -> FrontierConfig:
        return FrontierConfig(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            conv_channels=self.conv_channels,
            conv_kernels=self.conv_kernels,
            conv_strides=self.conv_strides,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
            contrastive_temperature=self.contrastive_temperature,
            n_negatives=self.n_negatives,
            diversity_weight=self.diversity_weight,
            n_classes=self.n_classes,
        )

    def to_trainer_config(self) -> DeepTrainerConfig:
        return DeepTrainerConfig(
            learning_rate=self.learning_rate,
            min_lr=self.min_lr,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            batch_size=self.batch_size,
            eval_every=self.eval_every,
            save_every=self.save_every,
            log_every=self.log_every,
            patience=self.patience,
            initial_mask_prob=self.initial_mask_prob,
            final_mask_prob=self.final_mask_prob,
            mask_warmup_steps=self.mask_warmup_steps,
            fd_epsilon=self.fd_epsilon,
            n_param_samples=self.n_param_samples,
            max_grad_norm=self.max_grad_norm,
        )


def _simple_mel_spectrogram(
    signal: np.ndarray,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    sr: int = 22050,
) -> np.ndarray:
    """Compute log-mel spectrogram from raw audio.

    Returns (n_mels, T) array in log scale.
    """
    # STFT
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    if n_frames < 1:
        pad_len = n_fft - len(signal)
        signal = np.concatenate([signal, np.zeros(pad_len)])
        n_frames = 1

    window = np.hanning(n_fft)
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        frame = signal[i * hop_length: i * hop_length + n_fft] * window
        stft[:, i] = np.fft.rfft(frame)

    power = np.abs(stft) ** 2

    # Mel filterbank
    mel_low = 0.0
    mel_high = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        f_left, f_center, f_right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(f_left, f_center):
            if f_center > f_left:
                fb[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                fb[m, k] = (f_right - k) / (f_right - f_center)

    mel_spec = fb @ power
    return np.log(mel_spec + 1e-9)


# ---------- online augmentation for spectrograms ----------

class SpectrogramAugmenter:
    """Online augmentation applied to mel spectrograms during training.

    Applies SpecAugment (Park et al., 2019) and frequency warping
    stochastically to each batch item.
    """

    def __init__(self, config: FrontierPipelineConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        if self.rng.rand() < self.config.spec_augment_prob:
            spec = spec_augment(
                spec,
                n_freq_masks=2, freq_width=min(15, spec.shape[0] // 4),
                n_time_masks=2, time_width=min(35, spec.shape[1] // 4),
                seed=self.rng.randint(0, 2**31),
            )
        if self.rng.rand() < self.config.freq_warp_prob:
            factor = self.rng.uniform(0.95, 1.05)
            spec = frequency_warp(spec, warp_factor=factor)
        return spec


# ---------- data preparation ----------

class FrontierDataset:
    """Dataset that stores mel spectrograms and supports batched iteration.

    Integrates with AudioPreprocessor for raw audio → spectrogram conversion
    and SpectrogramAugmenter for online augmentation.
    """

    def __init__(
        self,
        config: FrontierPipelineConfig,
        augment: bool = True,
    ):
        self.config = config
        self.spectrograms: List[np.ndarray] = []
        self.labels: List[Optional[int]] = []
        self.metadata: List[Dict] = []
        self.rng = np.random.RandomState(config.seed)
        self.augmenter = SpectrogramAugmenter(config) if augment else None

    def add_audio(
        self,
        audio: np.ndarray,
        label: Optional[int] = None,
        meta: Optional[Dict] = None,
    ) -> int:
        """Add raw audio, converting to mel spectrogram."""
        spec = _simple_mel_spectrogram(
            audio,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            sr=self.config.sr,
        )
        return self._add_spec(spec, label, meta)

    def add_spectrogram(
        self,
        spec: np.ndarray,
        label: Optional[int] = None,
        meta: Optional[Dict] = None,
    ) -> int:
        """Add a pre-computed spectrogram directly."""
        return self._add_spec(spec, label, meta)

    def _add_spec(self, spec: np.ndarray, label, meta) -> int:
        idx = len(self.spectrograms)
        self.spectrograms.append(spec)
        self.labels.append(label)
        self.metadata.append(meta or {})
        return idx

    def __len__(self) -> int:
        return len(self.spectrograms)

    def sample_batch(
        self, batch_size: int, seed: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[Optional[int]]]:
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        n = len(self.spectrograms)
        indices = rng.choice(n, size=min(batch_size, n), replace=False)
        specs = []
        for i in indices:
            s = self.spectrograms[i]
            if self.augmenter is not None:
                s = self.augmenter(s.copy())
            specs.append(s)
        labels = [self.labels[i] for i in indices]
        return specs, labels

    def to_deep_dataset(self) -> DeepAudioDataset:
        """Convert to a DeepAudioDataset for the DeepFrontierTrainer."""
        int_labels = [l if l is not None else -1 for l in self.labels]
        return DeepAudioDataset(
            spectrograms=list(self.spectrograms),
            labels=int_labels,
            metadata=list(self.metadata),
        )

    @staticmethod
    def from_synthetic(
        n_samples: int = 100,
        config: Optional[FrontierPipelineConfig] = None,
        augment: bool = True,
    ) -> "FrontierDataset":
        """Create a dataset from synthetic audio for testing."""
        config = config or FrontierPipelineConfig()
        gen = SyntheticAudioGenerator(seed=config.seed)
        signals = gen.generate_dataset(
            n_samples,
            duration_s=config.segment_duration_s,
            sr=config.sr,
        )
        ds = FrontierDataset(config, augment=augment)
        for i, sig in enumerate(signals):
            ds.add_audio(sig, label=i % 4, meta={"synthetic": True, "index": i})
        return ds


# ---------- evaluation ----------

@dataclass
class FrontierEvalResult:
    """Results from evaluating the frontier model."""
    contrastive_loss: float
    diversity_loss: float
    prediction_loss: float
    total_loss: float
    codebook_perplexity: float
    embedding_isotropy: float
    embedding_variance: float
    n_samples: int


def _compute_isotropy(embeddings: np.ndarray) -> float:
    """Measure isotropy of embedding space (Mu et al., 2018).

    Isotropy = min(λ) / max(λ) of the covariance eigenvalues.
    1.0 = perfectly isotropic; 0.0 = degenerate.
    """
    if embeddings.shape[0] < 2:
        return 0.0
    centered = embeddings - embeddings.mean(axis=0)
    cov = centered.T @ centered / (embeddings.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 0.0
    return float(eigvals.min() / eigvals.max())


def _compute_codebook_perplexity(model: MusicFrontierModel, specs: List[np.ndarray]) -> float:
    """Average codebook perplexity across a batch.

    Higher perplexity = better codebook utilization (all entries used equally).
    """
    all_probs = []
    for spec in specs:
        features = model.feature_extractor.forward(spec)
        projected = features @ model.quantizer.proj_W
        T, D = projected.shape
        G = model.quantizer.n_codebooks
        d_g = D // G
        for g in range(G):
            z_g = projected[:, g * d_g:(g + 1) * d_g]
            dists = np.sum((z_g[:, np.newaxis, :] - model.quantizer.codebooks[g][np.newaxis, :, :]) ** 2, axis=-1)
            probs = np.exp(-dists)
            probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-10)
            avg_probs = probs.mean(axis=0)
            all_probs.append(avg_probs)

    if not all_probs:
        return 0.0
    combined = np.mean(all_probs, axis=0)
    combined = combined / (combined.sum() + 1e-10)
    entropy = -np.sum(combined * np.log(combined + 1e-10))
    return float(np.exp(entropy))


class FrontierEvaluator:
    """Evaluation harness for the frontier model.

    Computes pretraining quality metrics:
    - Contrastive / diversity / prediction losses on held-out data
    - Codebook perplexity (utilization of quantized codes)
    - Embedding isotropy (uniformity of representation space)
    - Embedding variance (information content of representations)
    """

    def __init__(self, model: MusicFrontierModel, config: FrontierPipelineConfig):
        self.model = model
        self.config = config

    def evaluate(
        self,
        dataset: "FrontierDataset",
        n_batches: int = 5,
    ) -> FrontierEvalResult:
        losses_c, losses_d, losses_p, losses_t = [], [], [], []
        all_embeddings = []

        for i in range(n_batches):
            specs, _ = dataset.sample_batch(
                self.config.batch_size, seed=i * 7919,
            )
            for spec in specs:
                result = self.model.pretrain_forward(spec, mask_seed=i * 31 + 17)
                losses_c.append(result["contrastive_loss"])
                losses_d.append(result["diversity_loss"])
                losses_p.append(result["prediction_loss"])
                losses_t.append(result["total_loss"])

                features = self.model.feature_extractor.forward(spec)
                enc_out = self.model.transformer.forward(features)
                hidden = enc_out["hidden_states"]
                all_embeddings.append(hidden.mean(axis=0))

        embeddings = np.array(all_embeddings) if all_embeddings else np.zeros((1, self.config.d_model))
        perplexity = _compute_codebook_perplexity(self.model, specs[:3] if specs else [])

        return FrontierEvalResult(
            contrastive_loss=float(np.mean(losses_c)) if losses_c else 0.0,
            diversity_loss=float(np.mean(losses_d)) if losses_d else 0.0,
            prediction_loss=float(np.mean(losses_p)) if losses_p else 0.0,
            total_loss=float(np.mean(losses_t)) if losses_t else 0.0,
            codebook_perplexity=perplexity,
            embedding_isotropy=_compute_isotropy(embeddings),
            embedding_variance=float(np.var(embeddings)),
            n_samples=len(all_embeddings),
        )


# ---------- downstream task heads ----------

class LinearProbe:
    """Linear classifier probe for evaluating learned representations.

    Freezes the frontier model and trains a single linear layer
    on top of mean-pooled transformer embeddings.
    """

    def __init__(self, d_model: int, n_classes: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (d_model + n_classes))
        self.W = rng.randn(d_model, n_classes) * scale
        self.b = np.zeros(n_classes)
        self.d_model = d_model
        self.n_classes = n_classes

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict logits from embeddings. (N, d_model) → (N, n_classes)."""
        return embeddings @ self.W + self.b

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(embeddings), axis=-1)

    def accuracy(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        preds = self.predict(embeddings)
        return float(np.mean(preds == labels))

    def train_step(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        lr: float = 1e-3,
    ) -> float:
        """One gradient step via cross-entropy + SPSA on probe weights."""
        N = embeddings.shape[0]
        logits = self.forward(embeddings)

        # Softmax + cross-entropy
        logits_stable = logits - logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits_stable)
        probs = exp_l / (exp_l.sum(axis=-1, keepdims=True) + 1e-10)
        loss = -np.mean(np.log(probs[np.arange(N), labels] + 1e-10))

        # Gradient of CE w.r.t. logits = probs - one_hot
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), labels] = 1.0
        d_logits = (probs - one_hot) / N

        # Gradient w.r.t. W, b
        dW = embeddings.T @ d_logits
        db = d_logits.sum(axis=0)

        self.W -= lr * dW
        self.b -= lr * db
        return float(loss)


# ---------- full pipeline ----------

class FrontierTrainingPipeline:
    """Complete pipeline: data → model → train → evaluate → checkpoint.

    Usage:
        pipe = FrontierTrainingPipeline()
        pipe.add_audio(audio_array)
        # or: pipe.load_synthetic(n_samples=200)
        result = pipe.run(max_steps=1000)
    """

    def __init__(self, config: Optional[FrontierPipelineConfig] = None, seed: int = 42):
        self.config = config or FrontierPipelineConfig()
        self.frontier_config = self.config.to_frontier_config()
        self.model = MusicFrontierModel(self.frontier_config, seed=seed)
        self.train_data = FrontierDataset(self.config, augment=True)
        self.val_data = FrontierDataset(self.config, augment=False)
        self.evaluator = FrontierEvaluator(self.model, self.config)
        self.rng = np.random.RandomState(seed)
        self._trainer: Optional[DeepFrontierTrainer] = None

    def add_audio(
        self,
        audio: np.ndarray,
        label: Optional[int] = None,
        meta: Optional[Dict] = None,
        val_split: float = 0.1,
    ) -> None:
        """Add raw audio to the dataset, with automatic train/val split."""
        if self.rng.rand() < val_split:
            self.val_data.add_audio(audio, label, meta)
        else:
            self.train_data.add_audio(audio, label, meta)

    def add_spectrogram(
        self,
        spec: np.ndarray,
        label: Optional[int] = None,
        meta: Optional[Dict] = None,
        val_split: float = 0.1,
    ) -> None:
        if self.rng.rand() < val_split:
            self.val_data.add_spectrogram(spec, label, meta)
        else:
            self.train_data.add_spectrogram(spec, label, meta)

    def load_synthetic(self, n_samples: int = 200) -> None:
        """Populate with synthetic audio for testing the pipeline."""
        gen = SyntheticAudioGenerator(seed=self.config.seed)
        signals = gen.generate_dataset(
            n_samples,
            duration_s=self.config.segment_duration_s,
            sr=self.config.sr,
        )
        for i, sig in enumerate(signals):
            self.add_audio(sig, label=i % 4, meta={"synthetic": True})

    def _build_trainer(self) -> DeepFrontierTrainer:
        trainer_config = self.config.to_trainer_config()
        return DeepFrontierTrainer(
            model=self.model,
            config=trainer_config,
            seed=self.config.seed,
        )

    def run(
        self,
        max_steps: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """Run the full pretraining pipeline.

        Returns a dict with training metrics, evaluation results,
        and timing information.
        """
        if len(self.train_data) == 0:
            self.load_synthetic(n_samples=200)

        max_steps = max_steps or self.config.total_steps
        trainer = self._build_trainer()
        self._trainer = trainer

        train_ds = self.train_data.to_deep_dataset()
        val_ds = self.val_data.to_deep_dataset() if len(self.val_data) > 0 else None

        log = []
        start = time.time()

        for step in range(max_steps):
            specs, _ = train_ds.sample_batch(self.config.batch_size, seed=step)
            metrics = trainer.train_step(specs, step_seed=step)

            if step % self.config.log_every == 0:
                log.append({"step": step, **metrics})
                if callback:
                    callback(step, metrics)

            if val_ds and step > 0 and step % self.config.eval_every == 0:
                val_metrics = trainer.evaluate(val_ds, n_batches=3)
                log[-1].update(val_metrics)

        elapsed = time.time() - start

        # Final evaluation
        eval_result = self.evaluator.evaluate(self.train_data, n_batches=3)

        return {
            "training": {
                "final_loss": trainer.state.loss_history[-1] if trainer.state.loss_history else 0,
                "best_loss": trainer.state.best_loss,
                "steps": trainer.state.step,
                "elapsed_seconds": elapsed,
            },
            "evaluation": {
                "contrastive_loss": eval_result.contrastive_loss,
                "diversity_loss": eval_result.diversity_loss,
                "prediction_loss": eval_result.prediction_loss,
                "total_loss": eval_result.total_loss,
                "codebook_perplexity": eval_result.codebook_perplexity,
                "embedding_isotropy": eval_result.embedding_isotropy,
                "embedding_variance": eval_result.embedding_variance,
            },
            "log": log,
        }

    def extract_embeddings(
        self,
        dataset: Optional["FrontierDataset"] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mean-pooled transformer embeddings from the dataset.

        Returns (embeddings, labels) arrays suitable for downstream probing.
        """
        ds = dataset or self.train_data
        embeddings = []
        labels = []
        for i, spec in enumerate(ds.spectrograms):
            features = self.model.feature_extractor.forward(spec)
            enc_out = self.model.transformer.forward(features)
            hidden = enc_out["hidden_states"]
            emb = hidden.mean(axis=0)
            embeddings.append(emb)
            labels.append(ds.labels[i] if ds.labels[i] is not None else -1)
        return np.array(embeddings), np.array(labels)

    def probe_classification(
        self,
        n_classes: int = 4,
        n_steps: int = 100,
        lr: float = 1e-3,
    ) -> Dict:
        """Train a linear probe on frozen embeddings (linear evaluation protocol).

        Standard protocol: freeze pretrained model, train linear classifier
        on top (Baevski et al., 2020; Chen et al., 2020).
        """
        embeddings, labels = self.extract_embeddings()
        valid = labels >= 0
        embeddings, labels = embeddings[valid], labels[valid]

        if len(embeddings) == 0:
            return {"accuracy": 0.0, "loss": float("inf"), "n_samples": 0}

        probe = LinearProbe(self.config.d_model, n_classes, seed=self.config.seed)
        losses = []
        for step in range(n_steps):
            loss = probe.train_step(embeddings, labels, lr=lr)
            losses.append(loss)

        acc = probe.accuracy(embeddings, labels)
        return {
            "accuracy": acc,
            "final_loss": losses[-1] if losses else 0,
            "n_samples": len(embeddings),
            "n_classes": n_classes,
        }

    def save_checkpoint(self, path: str) -> None:
        if self._trainer is not None:
            self._trainer.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        if self._trainer is None:
            self._trainer = self._build_trainer()
        self._trainer.load_checkpoint(path)


# ---------- convenience entry points ----------

def quick_pretrain(
    n_samples: int = 50,
    max_steps: int = 10,
    seed: int = 42,
) -> Dict:
    """Quick smoke test: generate synthetic data, pretrain, evaluate."""
    config = FrontierPipelineConfig(
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        conv_channels=[32, 64],
        conv_kernels=[10, 3],
        conv_strides=[5, 2],
        n_codebooks=1, codebook_size=32, codebook_dim=32,
        n_param_samples=4,
        seed=seed,
    )
    pipe = FrontierTrainingPipeline(config, seed=seed)
    pipe.load_synthetic(n_samples)
    result = pipe.run(max_steps=max_steps)
    probe_result = pipe.probe_classification(n_classes=4, n_steps=20)
    result["probe"] = probe_result
    return result


def pretrain_from_signals(
    signals: List[np.ndarray],
    labels: Optional[List[int]] = None,
    sr: int = 22050,
    max_steps: int = 100,
    config: Optional[FrontierPipelineConfig] = None,
) -> Tuple[FrontierTrainingPipeline, Dict]:
    """Pretrain a frontier model from raw audio signals.

    Returns (pipeline, results) so the caller can extract embeddings,
    probe, or continue training.
    """
    config = config or FrontierPipelineConfig(sr=sr)
    pipe = FrontierTrainingPipeline(config)
    for i, sig in enumerate(signals):
        label = labels[i] if labels else i % 4
        pipe.add_audio(sig, label=label)
    result = pipe.run(max_steps=max_steps)
    return pipe, result
