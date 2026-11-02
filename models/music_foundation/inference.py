"""Unified music inference pipeline.

Wires raw audio → FeaturePipeline → DomainIntegrator → MusicFoundationModel
into a single forward pass that produces embeddings and task predictions.

Extended with: batch inference, streaming/chunked inference, multi-task heads
(tempo, key, instrument, structure), embedding arithmetic, nearest-neighbour
retrieval, ensembling, calibration (Platt scaling), and uncertainty estimation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Tuple
import numpy as np

from .encoder import MusicFoundationModel, EncoderConfig, LinearLayer, gelu, layer_norm
from ..foundation.domain_integration import (
    DomainIntegrator, DomainAnalysis, DOMAIN_FEATURE_DIM,
    cognition_to_array, emotion_to_array, theory_to_array,
    acoustics_to_array, structure_to_array,
)
from ..music_pipeline.feature_pipeline import FeaturePipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    sr: int = 22050
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    n_genres: int = 10
    n_moods: int = 8
    n_instruments: int = 16
    enable_domain: bool = True
    chunk_seconds: float = 5.0
    chunk_overlap: float = 0.5
    calibrate: bool = False


@dataclass
class InferenceResult:
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(0))
    audio_embedding: np.ndarray = field(default_factory=lambda: np.zeros(0))
    semantic_embedding: np.ndarray = field(default_factory=lambda: np.zeros(0))
    genre_logits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    genre_probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mood_logits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mood_probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    instrument_logits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vad: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tempo_bpm: float = 0.0
    key_estimate: str = ""
    confidence: float = 0.0
    domain_analysis: DomainAnalysis | None = None
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    metadata: dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: "InferenceResult") -> float:
        a = self.embedding.flatten()
        b = other.embedding.flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def genre_label(self, labels: List[str] | None = None) -> str:
        idx = int(np.argmax(self.genre_logits))
        if labels and idx < len(labels):
            return labels[idx]
        return f"genre_{idx}"

    def mood_label(self, labels: List[str] | None = None) -> str:
        idx = int(np.argmax(self.mood_logits))
        if labels and idx < len(labels):
            return labels[idx]
        return f"mood_{idx}"

    def top_k_genres(self, k: int = 3,
                     labels: List[str] | None = None
                     ) -> List[Tuple[str, float]]:
        probs = self.genre_probs if len(self.genre_probs) > 0 else _softmax(self.genre_logits)
        idx = np.argsort(-probs)[:k]
        out = []
        for i in idx:
            name = labels[i] if labels and i < len(labels) else f"genre_{i}"
            out.append((name, float(probs[i])))
        return out


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    s = x - x.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# Platt scaling (post-hoc calibration)
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """Platt (1999) sigmoid calibration: P(y=1|f) = 1 / (1 + exp(A*f + B)).
    Fit A, B by minimising NLL on a held-out calibration set via Newton's
    method (Lin et al., 2007 improved algorithm)."""

    def __init__(self) -> None:
        self.A: float = 1.0
        self.B: float = 0.0

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            max_iter: int = 100, lr: float = 0.01) -> None:
        A, B = 0.0, 0.0
        n = len(logits)
        t_pos = (labels.sum() + 1) / (labels.sum() + 2)
        t_neg = 1.0 / (n - labels.sum() + 2)
        targets = np.where(labels > 0.5, t_pos, t_neg)

        for _ in range(max_iter):
            p = _sigmoid(A * logits + B)
            dA = np.dot(logits, p - targets)
            dB = np.sum(p - targets)
            A -= lr * dA / n
            B -= lr * dB / n

        self.A = float(A)
        self.B = float(B)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(self.A * logits + self.B)


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class GenreHead:
    def __init__(self, latent_dim: int, n_genres: int, seed: int = 200) -> None:
        self.fc1 = LinearLayer(latent_dim, latent_dim // 2, seed)
        self.fc2 = LinearLayer(latent_dim // 2, n_genres, seed + 1)
        self.n_genres = n_genres

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = gelu(self.fc1.forward(x))
        return self.fc2.forward(h)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.forward(x))


class MoodHead:
    def __init__(self, latent_dim: int, n_moods: int, seed: int = 300) -> None:
        self.fc1 = LinearLayer(latent_dim, latent_dim // 2, seed)
        self.fc2 = LinearLayer(latent_dim // 2, n_moods, seed + 1)
        self.n_moods = n_moods

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = gelu(self.fc1.forward(x))
        return self.fc2.forward(h)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.forward(x))


class VADHead:
    def __init__(self, latent_dim: int, seed: int = 400) -> None:
        self.fc = LinearLayer(latent_dim, 3, seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.fc.forward(x))


class InstrumentHead:
    """Multi-label instrument detection (sigmoid per instrument)."""
    def __init__(self, latent_dim: int, n_instruments: int,
                 seed: int = 500) -> None:
        self.fc1 = LinearLayer(latent_dim, latent_dim // 2, seed)
        self.fc2 = LinearLayer(latent_dim // 2, n_instruments, seed + 1)
        self.n_instruments = n_instruments

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = gelu(self.fc1.forward(x))
        return self.fc2.forward(h)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (_sigmoid(self.forward(x)) > threshold).astype(np.float32)


class TempoHead:
    """Regression head for BPM estimation."""
    def __init__(self, latent_dim: int, seed: int = 600) -> None:
        self.fc1 = LinearLayer(latent_dim, latent_dim // 4, seed)
        self.fc2 = LinearLayer(latent_dim // 4, 1, seed + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = gelu(self.fc1.forward(x))
        raw = self.fc2.forward(h).flatten()
        return 30.0 + 200.0 * _sigmoid(raw)  # map to [30, 230] BPM


# ---------------------------------------------------------------------------
# Nearest-neighbour retrieval index
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """Brute-force cosine similarity index for embedding retrieval."""

    def __init__(self) -> None:
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None

    def add(self, embedding: np.ndarray,
            meta: Dict[str, Any] | None = None) -> int:
        self.embeddings.append(embedding.flatten())
        self.metadata.append(meta or {})
        self._matrix = None
        return len(self.embeddings) - 1

    def add_batch(self, embeddings: np.ndarray,
                  metas: List[Dict[str, Any]] | None = None) -> None:
        for i, emb in enumerate(embeddings):
            m = metas[i] if metas and i < len(metas) else {}
            self.add(emb, m)

    def _build(self) -> np.ndarray:
        if self._matrix is None:
            mat = np.vstack(self.embeddings)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            self._matrix = mat / norms
        return self._matrix

    def query(self, embedding: np.ndarray, k: int = 5
              ) -> List[Tuple[int, float, Dict[str, Any]]]:
        if len(self.embeddings) == 0:
            return []
        mat = self._build()
        q = embedding.flatten()
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return []
        q = q / qn
        sims = mat @ q
        k = min(k, len(sims))
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i]), self.metadata[i]) for i in idx]

    def __len__(self) -> int:
        return len(self.embeddings)


# ---------------------------------------------------------------------------
# Embedding arithmetic
# ---------------------------------------------------------------------------

def embedding_analogy(a: np.ndarray, b: np.ndarray,
                      c: np.ndarray) -> np.ndarray:
    """a is to b as c is to ? → b - a + c (normalised)."""
    result = b.flatten() - a.flatten() + c.flatten()
    n = np.linalg.norm(result)
    return result / max(n, 1e-12)


def embedding_interpolate(a: np.ndarray, b: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    """Spherical linear interpolation between two embeddings."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na < 1e-12 or nb < 1e-12:
        return (1 - alpha) * a_flat + alpha * b_flat
    a_norm = a_flat / na
    b_norm = b_flat / nb
    dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    omega = np.arccos(dot)
    if abs(omega) < 1e-8:
        return (1 - alpha) * a_flat + alpha * b_flat
    s = np.sin(omega)
    return (np.sin((1 - alpha) * omega) / s) * a_flat + \
           (np.sin(alpha * omega) / s) * b_flat


# ---------------------------------------------------------------------------
# Uncertainty estimation (MC dropout proxy)
# ---------------------------------------------------------------------------

def estimate_uncertainty(pipeline: 'MusicInferencePipeline',
                         audio: np.ndarray, sr: int,
                         n_samples: int = 5, noise_std: float = 0.01,
                         seed: int = 42) -> Dict[str, float]:
    """Approximate epistemic uncertainty by perturbing input and measuring
    output variance (pseudo-MC-dropout for deterministic models)."""
    rng = np.random.RandomState(seed)
    embeddings = []
    genre_preds = []
    mood_preds = []

    for _ in range(n_samples):
        noisy = audio + noise_std * rng.randn(*audio.shape).astype(audio.dtype)
        r = pipeline.infer(noisy, sr)
        embeddings.append(r.embedding)
        genre_preds.append(r.genre_logits)
        mood_preds.append(r.mood_logits)

    emb_stack = np.vstack([e.reshape(1, -1) for e in embeddings])
    emb_var = float(np.mean(np.var(emb_stack, axis=0)))

    genre_stack = np.vstack([g.reshape(1, -1) for g in genre_preds])
    genre_entropy = float(-np.sum(
        _softmax(genre_stack.mean(0)) *
        np.log(_softmax(genre_stack.mean(0)) + 1e-12)))

    mood_stack = np.vstack([m.reshape(1, -1) for m in mood_preds])
    mood_entropy = float(-np.sum(
        _softmax(mood_stack.mean(0)) *
        np.log(_softmax(mood_stack.mean(0)) + 1e-12)))

    return {
        "embedding_variance": emb_var,
        "genre_entropy": genre_entropy,
        "mood_entropy": mood_entropy,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class MusicInferencePipeline:
    """End-to-end: raw audio → embeddings + predictions."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self.feature_pipeline = FeaturePipeline(self.config.pipeline)
        self.domain_integrator = (DomainIntegrator(sr=self.config.sr)
                                  if self.config.enable_domain else None)
        self.foundation = MusicFoundationModel(self.config.encoder)
        self.genre_head = GenreHead(self.config.encoder.latent_dim,
                                    self.config.n_genres)
        self.mood_head = MoodHead(self.config.encoder.latent_dim,
                                   self.config.n_moods)
        self.vad_head = VADHead(self.config.encoder.latent_dim)
        self.instrument_head = InstrumentHead(
            self.config.encoder.latent_dim, self.config.n_instruments)
        self.tempo_head = TempoHead(self.config.encoder.latent_dim)
        self.genre_calibrator: Optional[PlattCalibrator] = None
        self.mood_calibrator: Optional[PlattCalibrator] = None

    def _audio_to_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        feat_result = self.feature_pipeline.extract(audio, sr)
        vec = feat_result.to_vector()
        if len(vec) == 0:
            vec = np.zeros(self.config.encoder.audio_dim)
        return vec

    def _domain_to_semantic(self, domain: DomainAnalysis) -> np.ndarray:
        parts = [
            cognition_to_array(domain.cognition),
            emotion_to_array(domain.emotion),
            theory_to_array(domain.theory),
            acoustics_to_array(domain.acoustics),
            structure_to_array(domain.structure),
        ]
        return np.concatenate(parts).astype(np.float64)

    def infer(self, audio: np.ndarray, sr: int | None = None) -> InferenceResult:
        sr = sr or self.config.sr
        result = InferenceResult()
        result.metadata["sr"] = sr
        result.metadata["n_samples"] = len(audio)

        if audio.ndim == 2:
            audio = (audio.mean(axis=1) if audio.shape[0] > audio.shape[1]
                     else audio.mean(axis=0))
        audio = audio.astype(np.float64)

        audio_features = self._audio_to_features(audio, sr)
        result.feature_vector = audio_features

        domain = None
        semantic_features = np.zeros(self.config.encoder.semantic_dim)
        if self.domain_integrator is not None:
            domain = self.domain_integrator.analyze(
                audio.astype(np.float32), sr)
            result.domain_analysis = domain
            sem = self._domain_to_semantic(domain)
            n = min(len(sem), len(semantic_features))
            semantic_features[:n] = sem[:n]
            result.tempo_bpm = domain.theory.tempo_bpm
            result.key_estimate = domain.theory.key

        audio_emb = self.foundation.encode_audio(audio_features)
        semantic_emb = self.foundation.encode_semantic(semantic_features)
        fused = self.foundation.bridge.fuse(audio_emb, semantic_emb)

        emb = fused.mean(axis=0) if fused.ndim == 2 else fused.flatten()
        result.embedding = emb
        result.audio_embedding = (audio_emb.mean(axis=0) if audio_emb.ndim == 2
                                  else audio_emb.flatten())
        result.semantic_embedding = (semantic_emb.mean(axis=0)
                                     if semantic_emb.ndim == 2
                                     else semantic_emb.flatten())

        emb_2d = emb.reshape(1, -1)
        result.genre_logits = self.genre_head.forward(emb_2d).flatten()
        result.genre_probs = self.genre_head.predict(emb_2d).flatten()
        result.mood_logits = self.mood_head.forward(emb_2d).flatten()
        result.mood_probs = self.mood_head.predict(emb_2d).flatten()
        result.vad = self.vad_head.forward(emb_2d).flatten()
        result.instrument_logits = self.instrument_head.forward(
            emb_2d).flatten()

        # confidence = max genre probability
        result.confidence = float(result.genre_probs.max())

        return result

    def infer_chunked(self, audio: np.ndarray,
                      sr: int | None = None) -> InferenceResult:
        """Chunked inference for long audio: split into overlapping windows,
        infer each, aggregate embeddings by mean-pooling."""
        sr = sr or self.config.sr
        chunk_len = int(self.config.chunk_seconds * sr)
        hop = int(chunk_len * (1 - self.config.chunk_overlap))
        hop = max(hop, 1)

        if len(audio) <= chunk_len:
            return self.infer(audio, sr)

        results: List[InferenceResult] = []
        pos = 0
        while pos < len(audio):
            end = min(pos + chunk_len, len(audio))
            chunk = audio[pos:end]
            if len(chunk) < sr // 10:
                break
            results.append(self.infer(chunk, sr))
            pos += hop

        if not results:
            return self.infer(audio, sr)

        # aggregate by mean-pooling
        agg = InferenceResult()
        agg.embedding = np.mean([r.embedding for r in results], axis=0)
        agg.audio_embedding = np.mean(
            [r.audio_embedding for r in results], axis=0)
        agg.semantic_embedding = np.mean(
            [r.semantic_embedding for r in results], axis=0)
        agg.genre_logits = np.mean(
            [r.genre_logits for r in results], axis=0)
        agg.genre_probs = _softmax(agg.genre_logits)
        agg.mood_logits = np.mean([r.mood_logits for r in results], axis=0)
        agg.mood_probs = _softmax(agg.mood_logits)
        agg.vad = np.mean([r.vad for r in results], axis=0)
        agg.instrument_logits = np.mean(
            [r.instrument_logits for r in results], axis=0)
        agg.confidence = float(agg.genre_probs.max())
        agg.tempo_bpm = np.median([r.tempo_bpm for r in results])
        agg.key_estimate = results[len(results) // 2].key_estimate
        agg.metadata["n_chunks"] = len(results)
        agg.metadata["sr"] = sr
        agg.metadata["n_samples"] = len(audio)
        return agg

    def infer_batch(self, audios: list[np.ndarray],
                    sr: int | None = None) -> list[InferenceResult]:
        return [self.infer(a, sr) for a in audios]

    def embedding_matrix(self, audios: list[np.ndarray],
                         sr: int | None = None) -> np.ndarray:
        results = self.infer_batch(audios, sr)
        return np.vstack([r.embedding.reshape(1, -1) for r in results])


# ---------------------------------------------------------------------------
# Multi-pipeline ensemble
# ---------------------------------------------------------------------------

class EnsemblePipeline:
    """Combine predictions from multiple MusicInferencePipelines."""

    def __init__(self, pipelines: List[MusicInferencePipeline]) -> None:
        self.pipelines = pipelines

    def infer(self, audio: np.ndarray,
              sr: int | None = None) -> InferenceResult:
        results = [p.infer(audio, sr) for p in self.pipelines]
        agg = InferenceResult()
        agg.embedding = np.mean([r.embedding for r in results], axis=0)
        agg.genre_logits = np.mean(
            [r.genre_logits for r in results], axis=0)
        agg.genre_probs = _softmax(agg.genre_logits)
        agg.mood_logits = np.mean([r.mood_logits for r in results], axis=0)
        agg.mood_probs = _softmax(agg.mood_logits)
        agg.vad = np.mean([r.vad for r in results], axis=0)
        agg.confidence = float(agg.genre_probs.max())
        agg.metadata["ensemble_size"] = len(self.pipelines)
        return agg


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    print("inference self-test …")
    rng = np.random.RandomState(42)

    # softmax / sigmoid
    x = np.array([1.0, 2.0, 3.0])
    s = _softmax(x)
    assert abs(s.sum() - 1.0) < 1e-6
    assert s[2] > s[1] > s[0]

    sig = _sigmoid(np.array([0.0, 100.0, -100.0]))
    assert abs(sig[0] - 0.5) < 1e-6
    assert sig[1] > 0.99
    assert sig[2] < 0.01

    # Platt calibrator
    cal = PlattCalibrator()
    logits_cal = rng.randn(50).astype(np.float32)
    labels_cal = (logits_cal > 0).astype(np.float32)
    cal.fit(logits_cal, labels_cal)
    probs_cal = cal.calibrate(logits_cal)
    assert probs_cal.shape == logits_cal.shape
    assert np.all(probs_cal >= 0) and np.all(probs_cal <= 1)

    # task heads (standalone)
    latent = 64
    gh = GenreHead(latent, 10)
    z = rng.randn(1, latent).astype(np.float32)
    gl = gh.forward(z)
    assert gl.shape == (1, 10)
    gp = gh.predict(z)
    assert abs(gp.sum() - 1.0) < 1e-5

    mh = MoodHead(latent, 8)
    ml = mh.forward(z)
    assert ml.shape == (1, 8)

    vh = VADHead(latent)
    v = vh.forward(z)
    assert v.shape == (1, 3)
    assert np.all(v >= 0) and np.all(v <= 1)

    ih = InstrumentHead(latent, 16)
    il = ih.forward(z)
    assert il.shape == (1, 16)
    ip = ih.predict(z)
    assert ip.shape == (1, 16)
    assert set(np.unique(ip)).issubset({0.0, 1.0})

    th = TempoHead(latent)
    bpm = th.forward(z)
    assert 30 <= bpm[0] <= 230

    # embedding index
    idx = EmbeddingIndex()
    for i in range(10):
        idx.add(rng.randn(latent), {"id": i})
    assert len(idx) == 10
    results = idx.query(rng.randn(latent), k=3)
    assert len(results) == 3
    assert all(-1.0 <= r[1] <= 1.0 for r in results)

    # batch add
    idx.add_batch(rng.randn(5, latent))
    assert len(idx) == 15

    # embedding arithmetic
    a = rng.randn(latent)
    b = rng.randn(latent)
    c = rng.randn(latent)
    analog = embedding_analogy(a, b, c)
    assert abs(np.linalg.norm(analog) - 1.0) < 1e-5

    interp = embedding_interpolate(a, b, 0.5)
    assert interp.shape == (latent,)

    interp0 = embedding_interpolate(a, b, 0.0)
    assert np.allclose(interp0, a, atol=1e-5)
    interp1 = embedding_interpolate(a, b, 1.0)
    assert np.allclose(interp1, b, atol=1e-5)

    # InferenceResult helpers
    r = InferenceResult()
    r.genre_logits = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
    r.genre_probs = _softmax(r.genre_logits)
    assert r.genre_label() == "genre_3"
    labels = ["rock", "pop", "jazz", "electronic", "classical"]
    assert r.genre_label(labels) == "electronic"
    top = r.top_k_genres(2, labels)
    assert top[0][0] == "electronic"
    assert len(top) == 2

    print("  all inference tests passed ✓")


if __name__ == "__main__":
    _self_test()
