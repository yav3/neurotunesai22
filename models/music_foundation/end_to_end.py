"""End-to-end music foundation model pipeline.

Wires audio loading → spectrogram → feature extraction → tokenization →
training → evaluation into a single runnable entry point.  Pure numpy/scipy
— no external ML framework.

Usage:
    from models.music_foundation.end_to_end import MusicFoundationPipeline
    pipe = MusicFoundationPipeline()
    pipe.add_audio(audio_array, sr=22050, metadata={"genre": "jazz"})
    results = pipe.train(epochs=20)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .encoder import MusicFoundationModel, EncoderConfig
from .spectrogram import SpectrogramEncoder, SpectrogramEncoderConfig, _compute_mel_spectrogram
from .tokenizer import MusicTokenizer
from .pretraining import PretrainingObjective
from .trainer import MusicTrainer, TrainerConfig, TrainingHistory
from .checkpoint import save_checkpoint, load_checkpoint


@dataclass
class PipelineConfig:
    sr: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    audio_dim: int = 128
    semantic_dim: int = 64
    latent_dim: int = 128
    hidden_dim: int = 256
    n_heads: int = 4
    n_codebooks: int = 4
    n_codes: int = 512
    batch_size: int = 16
    lr: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10
    max_frames: int = 512
    segment_duration: float = 5.0
    val_ratio: float = 0.15
    seed: int = 42


@dataclass
class AudioSample:
    audio: np.ndarray
    sr: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    features: Optional[np.ndarray] = None
    tokens: Optional[List[np.ndarray]] = None


@dataclass
class TrainingResult:
    history: TrainingHistory
    final_train_loss: float
    final_val_loss: Optional[float]
    n_samples: int
    n_epochs_run: int
    codebook_perplexity: List[float]
    reconstruction_error: float


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    """Load a WAV file using only stdlib + numpy (no librosa/soundfile)."""
    import wave
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr


def generate_synthetic_dataset(
    n_samples: int = 100,
    duration: float = 5.0,
    sr: int = 22050,
    seed: int = 42,
) -> List[AudioSample]:
    """Generate synthetic audio samples for pipeline testing.

    Creates realistic-ish audio: sums of harmonic partials with ADSR envelopes,
    varying timbres, and optional noise.  Each sample gets metadata (genre, tempo,
    key) drawn from a categorical distribution so the semantic tower has signal.
    """
    rng = np.random.RandomState(seed)
    n_samp = int(duration * sr)
    t = np.linspace(0, duration, n_samp, endpoint=False)

    genres = ["classical", "jazz", "rock", "electronic", "folk", "blues", "ambient"]
    keys = ["C", "D", "E", "F", "G", "A", "Bb"]
    modes = ["major", "minor"]

    samples: List[AudioSample] = []
    for i in range(n_samples):
        fund_freq = 110.0 * 2 ** (rng.randint(0, 24) / 12.0)
        n_partials = rng.randint(3, 12)
        audio = np.zeros(n_samp)
        for h in range(1, n_partials + 1):
            amp = rng.exponential(0.3) / h
            phase = rng.uniform(0, 2 * np.pi)
            audio += amp * np.sin(2 * np.pi * fund_freq * h * t + phase)

        attack = rng.uniform(0.01, 0.1)
        decay = rng.uniform(0.05, 0.3)
        sustain_level = rng.uniform(0.3, 0.8)
        release = rng.uniform(0.1, 0.5)
        env = _adsr_envelope(n_samp, sr, attack, decay, sustain_level, release, duration)
        audio *= env

        noise_level = rng.uniform(0.0, 0.05)
        audio += noise_level * rng.randn(n_samp)

        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio /= peak
        audio *= rng.uniform(0.3, 0.9)

        genre = genres[rng.randint(len(genres))]
        key = keys[rng.randint(len(keys))]
        mode = modes[rng.randint(len(modes))]
        tempo = rng.uniform(60, 180)

        meta = {
            "genre": genre,
            "key": key,
            "mode": mode,
            "tempo": float(tempo),
            "fundamental_freq": float(fund_freq),
            "n_partials": int(n_partials),
            "sample_id": f"synth_{i:04d}",
        }
        samples.append(AudioSample(audio=audio, sr=sr, metadata=meta))
    return samples


def _adsr_envelope(
    n_samples: int, sr: int,
    attack: float, decay: float, sustain_level: float, release: float,
    total_duration: float,
) -> np.ndarray:
    env = np.zeros(n_samples)
    a_samp = int(attack * sr)
    d_samp = int(decay * sr)
    r_samp = int(release * sr)
    s_samp = max(0, n_samples - a_samp - d_samp - r_samp)

    pos = 0
    if a_samp > 0:
        env[pos:pos + a_samp] = np.linspace(0, 1, a_samp)
        pos += a_samp
    if d_samp > 0:
        env[pos:pos + d_samp] = np.linspace(1, sustain_level, d_samp)
        pos += d_samp
    if s_samp > 0:
        env[pos:pos + s_samp] = sustain_level
        pos += s_samp
    if r_samp > 0:
        actual_r = min(r_samp, n_samples - pos)
        if actual_r > 0:
            env[pos:pos + actual_r] = np.linspace(sustain_level, 0, actual_r)
    return env


def _encode_metadata(meta: Dict[str, Any], dim: int, seed: int = 0) -> np.ndarray:
    """Encode metadata dict into a fixed-dim semantic vector.

    Uses a simple hash-based projection: deterministic, no learned params,
    adequate for pretraining signal.
    """
    rng = np.random.RandomState(seed)
    vec = np.zeros(dim)
    genre_map = {
        "classical": 0, "jazz": 1, "rock": 2, "electronic": 3,
        "folk": 4, "blues": 5, "ambient": 6, "pop": 7, "metal": 8,
        "hiphop": 9, "country": 10, "reggae": 11,
    }
    mode_map = {"major": 0, "minor": 1, "dorian": 2, "mixolydian": 3}
    key_map = {
        "C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
        "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11,
    }

    genre_idx = genre_map.get(meta.get("genre", ""), 0)
    rng_g = np.random.RandomState(seed + genre_idx * 7 + 1)
    vec[:dim // 4] = rng_g.randn(dim // 4) * 0.5

    key_idx = key_map.get(meta.get("key", "C"), 0)
    rng_k = np.random.RandomState(seed + key_idx * 13 + 100)
    vec[dim // 4:dim // 2] = rng_k.randn(dim // 4) * 0.3

    mode_idx = mode_map.get(meta.get("mode", "major"), 0)
    rng_m = np.random.RandomState(seed + mode_idx * 11 + 200)
    vec[dim // 2:3 * dim // 4] = rng_m.randn(dim // 4) * 0.3

    tempo = meta.get("tempo", 120.0)
    tempo_norm = (tempo - 60.0) / 120.0
    vec[3 * dim // 4:] = tempo_norm * rng.randn(dim - 3 * dim // 4) * 0.2

    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


class MusicFoundationPipeline:
    """End-to-end pipeline: audio → features → tokens → train → evaluate."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        c = self.config

        self.encoder_config = EncoderConfig(
            audio_dim=c.audio_dim,
            semantic_dim=c.semantic_dim,
            hidden_dim=c.hidden_dim,
            latent_dim=c.latent_dim,
            n_heads=c.n_heads,
        )
        self.model = MusicFoundationModel(self.encoder_config)

        self.spec_config = SpectrogramEncoderConfig(
            n_mels=c.n_mels,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            hidden_dim=c.hidden_dim,
            latent_dim=c.latent_dim,
            n_heads=c.n_heads,
            max_frames=c.max_frames,
        )
        self.spec_encoder = SpectrogramEncoder(self.spec_config)
        self.tokenizer = MusicTokenizer(
            n_codes=c.n_codes, dim=c.latent_dim, n_quantizers=c.n_codebooks, seed=c.seed,
        )
        self._checkpoint_save = save_checkpoint
        self._checkpoint_load = load_checkpoint

        self._samples: List[AudioSample] = []
        self._audio_features: Optional[np.ndarray] = None
        self._semantic_features: Optional[np.ndarray] = None

    def add_audio(
        self, audio: np.ndarray, sr: int = 22050, metadata: Dict[str, Any] | None = None,
    ) -> int:
        sample = AudioSample(audio=audio, sr=sr, metadata=metadata or {})
        self._samples.append(sample)
        self._audio_features = None
        self._semantic_features = None
        return len(self._samples) - 1

    def add_wav(self, path: str, metadata: Dict[str, Any] | None = None) -> int:
        audio, sr = load_wav(path)
        return self.add_audio(audio, sr, metadata)

    def add_samples(self, samples: List[AudioSample]) -> None:
        self._samples.extend(samples)
        self._audio_features = None
        self._semantic_features = None

    def extract_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract audio and semantic features from all loaded samples."""
        c = self.config
        audio_feats = []
        semantic_feats = []

        for sample in self._samples:
            audio = sample.audio
            if sample.sr != c.sr:
                ratio = c.sr / sample.sr
                n_out = int(len(audio) * ratio)
                x_old = np.linspace(0, 1, len(audio))
                x_new = np.linspace(0, 1, n_out)
                audio = np.interp(x_new, x_old, audio)

            seg_len = int(c.segment_duration * c.sr)
            if len(audio) > seg_len:
                audio = audio[:seg_len]
            elif len(audio) < seg_len:
                audio = np.pad(audio, (0, seg_len - len(audio)))

            mel = _compute_mel_spectrogram(audio, c.sr, c.n_fft, c.hop_length, c.n_mels)
            mel_db = 10 * np.log10(np.maximum(mel, 1e-10))

            n_frames = mel_db.shape[0]
            n_bins = mel_db.shape[1]
            if n_frames > c.max_frames:
                mel_db = mel_db[:c.max_frames, :]
            elif n_frames < c.max_frames:
                mel_db = np.pad(mel_db, ((0, c.max_frames - n_frames), (0, 0)))

            bin_mean = mel_db.mean(axis=0)
            bin_std = mel_db.std(axis=0)
            spectral_centroid = np.sum(mel_db * np.arange(n_bins).reshape(1, -1), axis=1) / (
                np.sum(mel_db, axis=1) + 1e-10
            )
            sc_stats = np.array([spectral_centroid.mean(), spectral_centroid.std()])

            feat_vec = np.concatenate([bin_mean, bin_std, sc_stats])
            if len(feat_vec) < c.audio_dim:
                feat_vec = np.pad(feat_vec, (0, c.audio_dim - len(feat_vec)))
            else:
                feat_vec = feat_vec[:c.audio_dim]

            sample.features = feat_vec
            audio_feats.append(feat_vec)

            sem_vec = _encode_metadata(sample.metadata, c.semantic_dim, c.seed)
            semantic_feats.append(sem_vec)

        self._audio_features = np.array(audio_feats)
        self._semantic_features = np.array(semantic_feats)
        return self._audio_features, self._semantic_features

    def tokenize_all(self) -> List[List[np.ndarray]]:
        """Tokenize all samples' audio features through the residual VQ codebooks."""
        if self._audio_features is None:
            self.extract_features()
        all_tokens = []
        for i, sample in enumerate(self._samples):
            feat = self._audio_features[i:i + 1]
            tokens = self.tokenizer.tokenize(feat)
            sample.tokens = tokens
            all_tokens.append(tokens)
        return all_tokens

    def train(
        self,
        epochs: int = 50,
        val_ratio: float | None = None,
    ) -> TrainingResult:
        """Run the full training loop.

        1. Extract features (if not already done)
        2. Split into train/val
        3. Train with early stopping
        4. Report results
        """
        if self._audio_features is None:
            self.extract_features()

        vr = val_ratio if val_ratio is not None else self.config.val_ratio
        n = len(self._audio_features)
        rng = np.random.RandomState(self.config.seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * vr))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_audio = self._audio_features[train_idx]
        train_sem = self._semantic_features[train_idx]
        val_audio = self._audio_features[val_idx]
        val_sem = self._semantic_features[val_idx]

        trainer_config = TrainerConfig(
            lr=self.config.lr,
            epochs=epochs,
            batch_size=self.config.batch_size,
            grad_clip=self.config.grad_clip,
            patience=self.config.patience,
            eval_every=1,
        )
        trainer = MusicTrainer(
            model=self.model,
            config=trainer_config,
        )
        history = trainer.fit(train_audio, train_sem, val_audio, val_sem, epochs=epochs)

        self.tokenize_all()

        perplexities = [cb.perplexity() for cb in self.tokenizer.codebooks]
        recon_err = self._compute_reconstruction_error()

        final_train = history.epoch_losses[-1].get("total", 0.0) if history.epoch_losses else 0.0
        final_val = history.val_metrics[-1].get("total", 0.0) if history.val_metrics else None
        n_epochs_run = len(history.epoch_losses)

        return TrainingResult(
            history=history,
            final_train_loss=final_train,
            final_val_loss=final_val,
            n_samples=n,
            n_epochs_run=n_epochs_run,
            codebook_perplexity=perplexities,
            reconstruction_error=recon_err,
        )

    def _compute_reconstruction_error(self) -> float:
        """Measure VQ reconstruction fidelity: mean L2 between original and decoded."""
        if self._audio_features is None:
            return 0.0
        errors = []
        for i in range(len(self._audio_features)):
            feat = self._audio_features[i:i + 1]
            tokens = self.tokenizer.tokenize(feat)
            reconstructed = np.zeros_like(feat)
            for level_tokens, codebook in zip(tokens, self.tokenizer.codebooks):
                reconstructed += codebook.decode(level_tokens)
            err = float(np.mean((feat - reconstructed) ** 2))
            errors.append(err)
        return float(np.mean(errors))

    def encode(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Encode a single audio array into the joint latent space."""
        idx = self.add_audio(audio, sr)
        self.extract_features()
        audio_emb = self.model.encode_audio(self._audio_features[idx:idx + 1])
        return audio_emb.flatten()

    def similarity(self, audio_a: np.ndarray, audio_b: np.ndarray, sr: int = 22050) -> float:
        """Cosine similarity between two audio clips in the learned space."""
        emb_a = self.encode(audio_a, sr)
        emb_b = self.encode(audio_b, sr)
        dot = np.dot(emb_a, emb_b)
        norms = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        return float(dot / (norms + 1e-12))

    def save_checkpoint(self, path: str) -> None:
        state = {
            "model_params": self.model.params(),
            "codebook_embeddings": [cb.embeddings for cb in self.tokenizer.codebooks],
            "config": {
                "audio_dim": self.config.audio_dim,
                "semantic_dim": self.config.semantic_dim,
                "latent_dim": self.config.latent_dim,
                "n_codes": self.config.n_codes,
                "n_codebooks": self.config.n_codebooks,
            },
        }
        np.savez_compressed(path, **{k: v if isinstance(v, np.ndarray) else np.array(v, dtype=object) for k, v in state.items()})

    def load_checkpoint(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        if "model_params" in data:
            params = data["model_params"].item()
            self.model.load_params(params)
        if "codebook_embeddings" in data:
            embeddings = data["codebook_embeddings"]
            for i, cb in enumerate(self.tokenizer.codebooks):
                if i < len(embeddings):
                    cb.embeddings = embeddings[i]


def run_demo(n_samples: int = 20, epochs: int = 2) -> TrainingResult:
    """Quick demo: generate synthetic data, train, report.

    Uses small dimensions for tractable pure-numpy numerical gradients.
    Scale up audio_dim/hidden_dim/latent_dim for production quality.
    """
    samples = generate_synthetic_dataset(n_samples=n_samples, duration=1.0)
    pipe = MusicFoundationPipeline(PipelineConfig(
        segment_duration=1.0,
        batch_size=min(8, n_samples),
        max_frames=32,
        audio_dim=16,
        semantic_dim=8,
        hidden_dim=16,
        latent_dim=16,
        n_heads=2,
        n_codebooks=2,
        n_codes=32,
    ))
    pipe.add_samples(samples)
    return pipe.train(epochs=epochs)
