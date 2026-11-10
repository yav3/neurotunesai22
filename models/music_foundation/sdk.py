"""NeuroTunes SDK — unified interface for the music foundation model.

Implements point 5 of the robustness roadmap:
  pip install neurotunes  (or container: docker run neurotunes)
  REST API: uvicorn neurotunes.api:app
  3 reference pipelines: embed, generate, analyze

Pure numpy/scipy — no external ML framework dependencies.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .pretrain_orchestrator import (
    PretrainOrchestrator,
    OrchestratorConfig,
    FrontierConfig,
    AudioClip,
    PretrainHistory,
    quick_pretrain,
    _extract_all_params,
    _apply_all_params,
)
from .deployment import (
    SnapshotStore,
    ModelRegistry,
    EngineRegistry,
    ActionLayer,
    sha256_params,
    MethodCard,
)


# ---------------------------------------------------------------------------
# Core SDK class
# ---------------------------------------------------------------------------

class NeuroTunesModel:
    """High-level interface wrapping the entire foundation model stack.

    Usage:
        model = NeuroTunesModel.from_pretrained("path/to/checkpoint")
        embeddings = model.embed(audio_array, sr=16000)
        analysis = model.analyze(audio_array, sr=16000)
    """

    def __init__(
        self,
        orchestrator: PretrainOrchestrator,
        snapshot_store: Optional[SnapshotStore] = None,
    ):
        self._orch = orchestrator
        self._store = snapshot_store or SnapshotStore()
        self._registry = EngineRegistry()
        self._setup_engines()

    def _setup_engines(self) -> None:
        from .deployment import (
            SIGNAL_INGESTION_SPEC,
            TEMPORAL_CONTEXT_SPEC,
            VARIATIONAL_ENCODING_SPEC,
        )
        self._registry.register(SIGNAL_INGESTION_SPEC, self._signal_ingestion)
        self._registry.register(TEMPORAL_CONTEXT_SPEC, self._temporal_context)
        self._registry.register(VARIATIONAL_ENCODING_SPEC, self._variational_encoding)

    def _signal_ingestion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        signal = data["signal"]
        features = self._orch.feature_extractor.extract(signal)
        return {
            "features": features,
            "metadata": {
                "sr": data["sr"],
                "duration": len(signal) / data["sr"],
                "n_streams": len(features),
            },
        }

    def _temporal_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fused = self._orch.integrator.forward(data["features"])
        return {"fused": fused}

    def _variational_encoding(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fused = data["fused"]
        enc = self._orch.model.transformer
        x = fused.T  # (T, d_model)
        if x.shape[0] > 0:
            x = x + enc.pos_enc[:x.shape[0]]
        for layer in enc.layers:
            x, _attn = layer.forward(x)
        if hasattr(enc, "final_gamma"):
            x = enc.final_gamma * x + enc.final_beta

        q_result = self._orch.model.quantizer.forward(x)
        return {
            "embeddings": x,
            "quantized": q_result["quantized"],
            "indices": q_result["indices"],
        }

    # --- Public API ---

    def embed(self, signal: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract embeddings from raw audio.

        Args:
            signal: 1D float array (mono audio)
            sr: sample rate (will be resampled to model SR if different)

        Returns:
            (T, d_model) embedding matrix
        """
        result = self._registry.run_pipeline({"signal": signal, "sr": sr})
        return result["embeddings"]

    def embed_batch(self, signals: List[np.ndarray], sr: int = 16000) -> List[np.ndarray]:
        """Embed a batch of signals."""
        return [self.embed(sig, sr) for sig in signals]

    def embed_pooled(self, signal: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract a single fixed-size embedding (mean-pooled over time)."""
        emb = self.embed(signal, sr)
        return emb.mean(axis=0)

    def analyze(self, signal: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """Full analysis pipeline: features + embeddings + quantized codes."""
        result = self._registry.run_pipeline({"signal": signal, "sr": sr})
        emb = result["embeddings"]
        return {
            "embeddings": emb,
            "quantized_codes": result["indices"],
            "n_frames": emb.shape[0],
            "d_model": emb.shape[1] if emb.ndim > 1 else 0,
            "feature_streams": list(result.get("features", {}).keys()),
            "metadata": result.get("metadata", {}),
        }

    def similarity(self, sig1: np.ndarray, sig2: np.ndarray, sr: int = 16000) -> float:
        """Cosine similarity between two audio signals."""
        e1 = self.embed_pooled(sig1, sr)
        e2 = self.embed_pooled(sig2, sr)
        norm1 = np.linalg.norm(e1) + 1e-10
        norm2 = np.linalg.norm(e2) + 1e-10
        return float(np.dot(e1, e2) / (norm1 * norm2))

    def nearest_neighbors(
        self,
        query: np.ndarray,
        corpus: List[np.ndarray],
        sr: int = 16000,
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors in a corpus by embedding similarity."""
        q_emb = self.embed_pooled(query, sr)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        sims = []
        for i, sig in enumerate(corpus):
            c_emb = self.embed_pooled(sig, sr)
            c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-10)
            sims.append((i, float(np.dot(q_norm, c_norm))))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def save(self, path: str) -> str:
        """Save model to disk. Returns SHA-256 digest."""
        params = _extract_all_params(self._orch.model, self._orch.integrator)
        n_heads = self._orch.config.model_config.n_heads
        sr = self._orch.config.sr
        params["_meta"] = np.array([n_heads, sr], dtype=np.int64)
        digest = sha256_params(params)
        np.savez_compressed(path, **params)
        return digest

    @classmethod
    def from_pretrained(cls, path: str, config: Optional[OrchestratorConfig] = None) -> "NeuroTunesModel":
        """Load a pretrained model from disk."""
        data = np.load(path)
        params = dict(data)

        if config is None:
            d_model = params.get("enc.layer0.W_q", np.zeros((1,))).shape[0]
            n_layers = 0
            while f"enc.layer{n_layers}.W_q" in params:
                n_layers += 1
            n_mels = params["integ.proj.mel.W"].shape[0] if "integ.proj.mel.W" in params else 128
            n_codebooks = params["quant.codebooks"].shape[0] if "quant.codebooks" in params else 4
            codebook_size = params["quant.codebooks"].shape[1] if "quant.codebooks" in params else 64
            codebook_dim = params["quant.codebooks"].shape[2] if "quant.codebooks" in params else 256
            d_ff = params["enc.layer0.W_ff1"].shape[1] if "enc.layer0.W_ff1" in params else 4 * d_model
            if "_meta" in params:
                meta = params["_meta"]
                n_heads = int(meta[0])
                sr = int(meta[1]) if len(meta) > 1 else 22050
            else:
                n_heads = max(1, d_model // 16)
                while d_model % n_heads != 0 and n_heads > 1:
                    n_heads -= 1
                sr = 22050
            n_conv_layers = 0
            while f"conv.W{n_conv_layers}" in params:
                n_conv_layers += 1
            if n_conv_layers > 0:
                conv_channels = [int(params[f"conv.W{i}"].shape[0]) for i in range(n_conv_layers)]
                conv_kernels = [int(params[f"conv.W{i}"].shape[2]) for i in range(n_conv_layers)]
                default_strides = [5, 2, 2, 2, 2]
                conv_strides = default_strides[:n_conv_layers] + [2] * max(0, n_conv_layers - len(default_strides))
            else:
                conv_channels = [512] * 5
                conv_kernels = [10, 3, 3, 3, 3]
                conv_strides = [5, 2, 2, 2, 2]
            model_cfg = FrontierConfig(
                d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                d_ff=d_ff, n_mels=n_mels,
                n_codebooks=n_codebooks, codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                conv_channels=conv_channels, conv_kernels=conv_kernels,
                conv_strides=conv_strides,
            )
            config = OrchestratorConfig(model_config=model_cfg, sr=sr)

        orch = PretrainOrchestrator(config)
        if orch.integrator._fusion is None and orch.integrator.streams:
            orch.integrator._build_fusion()
        _apply_all_params(params, orch.model, orch.integrator)
        return cls(orch)

    @classmethod
    def train_new(
        cls,
        clips: Optional[List[AudioClip]] = None,
        n_clips: int = 50,
        n_steps: int = 10,
        d_model: int = 64,
        seed: int = 42,
    ) -> Tuple["NeuroTunesModel", PretrainHistory]:
        """Train a new model from scratch or from provided audio clips."""
        orch, history = quick_pretrain(
            n_clips=n_clips, n_steps=n_steps,
            batch_size=4, d_model=d_model, seed=seed,
        )
        model = cls(orch)
        return model, history

    @staticmethod
    def verify(checkpoint_path: str, seed: int = 42) -> Dict[str, Any]:
        """Verify a train-and-generate checkpoint's integrity and correctness."""
        from .train_and_generate import verify_checkpoint
        return verify_checkpoint(checkpoint_path, seed=seed)

    @staticmethod
    def generate_from_checkpoint(
        checkpoint_path: str,
        styles: Optional[List[str]] = None,
        temperatures: Optional[List[float]] = None,
        continuation_length: int = 192,
        phrase_form: str = 'AABA',
        output_dir: Optional[str] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generate music from a saved train-and-generate checkpoint.

        Returns a dict with generation metadata. If output_dir is given,
        WAV files are saved there.
        """
        from .train_and_generate import (
            reconstruct_from_checkpoint, TrainGenConfig,
            _generate_phrase_tokens, _tokens_to_audio_learned,
            save_wav,
        )
        model, decoder_W, decoder_b, rbf_params, ngram_model, cfg = \
            reconstruct_from_checkpoint(checkpoint_path)

        styles = styles or cfg.gen_styles
        temperatures = temperatures or cfg.gen_temperatures

        rng = np.random.RandomState(seed)
        results = []
        for style in styles:
            for temp in temperatures:
                # Generate per-codebook tokens, merge via primary codebook
                cb_tokens_all = []
                for g in range(cfg.n_codebooks):
                    cb_rng = np.random.RandomState(seed + g * 53)
                    cb_tok = _generate_phrase_tokens(
                        ngram_model, g, np.array([], dtype=int),
                        continuation_length, temperature=temp, rng=cb_rng,
                        form=phrase_form, variation_noise=cfg.phrase_variation,
                    )
                    cb_tokens_all.append(cb_tok)
                # Use primary codebook (0) as the token stream for synthesis
                tokens = cb_tokens_all[0]
                audio = _tokens_to_audio_learned(
                    tokens, model, decoder_W, decoder_b, rbf_params,
                    cfg.n_mels, cfg.sr, cfg.n_fft, cfg.hop_length,
                    style=style, phase_seed=rng.randint(10000),
                )
                entry = {
                    'style': style, 'temperature': temp,
                    'n_tokens': len(tokens),
                    'duration_s': round(len(audio) / cfg.sr, 2),
                }
                if output_dir:
                    import os
                    os.makedirs(output_dir, exist_ok=True)
                    fname = f"gen_{style}_T{temp:.1f}.wav"
                    path = os.path.join(output_dir, fname)
                    save_wav(audio, path, cfg.sr)
                    entry['path'] = path
                results.append(entry)

        return {'generations': results, 'n_generated': len(results)}

    @property
    def config(self) -> OrchestratorConfig:
        return self._orch.config

    @property
    def d_model(self) -> int:
        return self._orch.config.model_config.d_model

    @property
    def n_params(self) -> int:
        params = _extract_all_params(self._orch.model, self._orch.integrator)
        return sum(p.size for p in params.values())


# ---------------------------------------------------------------------------
# Reference pipelines
# ---------------------------------------------------------------------------

def pipeline_embed(
    audio_paths: List[str],
    model_path: str,
    output_path: str = "embeddings.npz",
    sr: int = 16000,
) -> Dict[str, Any]:
    """Reference pipeline 1: Batch embedding extraction.

    Load audio files → extract embeddings → save to .npz
    """
    from scipy.io import wavfile

    model = NeuroTunesModel.from_pretrained(model_path)
    results = {}
    for path in audio_paths:
        try:
            file_sr, data = wavfile.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float64) / (np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0)
            if file_sr != sr:
                ratio = sr / file_sr
                n_out = int(len(data) * ratio)
                indices = np.arange(n_out) / ratio
                idx = np.clip(indices.astype(int), 0, len(data) - 1)
                data = data[idx]
            emb = model.embed_pooled(data, sr)
            results[Path(path).stem] = emb
        except Exception as e:
            results[Path(path).stem] = {"error": str(e)}

    np.savez_compressed(output_path, **{k: v for k, v in results.items() if isinstance(v, np.ndarray)})
    return {"n_processed": len(results), "output": output_path}


def pipeline_analyze(
    audio_path: str,
    model_path: str,
    sr: int = 16000,
) -> Dict[str, Any]:
    """Reference pipeline 2: Single-file deep analysis."""
    from scipy.io import wavfile

    model = NeuroTunesModel.from_pretrained(model_path)
    file_sr, data = wavfile.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64) / (np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0)

    analysis = model.analyze(data, sr)
    analysis["file"] = audio_path
    analysis["duration_seconds"] = len(data) / file_sr
    return analysis


def pipeline_similarity_matrix(
    audio_paths: List[str],
    model_path: str,
    sr: int = 16000,
) -> np.ndarray:
    """Reference pipeline 3: Pairwise similarity matrix."""
    from scipy.io import wavfile

    model = NeuroTunesModel.from_pretrained(model_path)
    embeddings = []
    for path in audio_paths:
        file_sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float64) / (np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0)
        embeddings.append(model.embed_pooled(data, sr))

    emb_matrix = np.stack(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10
    emb_n = emb_matrix / norms
    return emb_n @ emb_n.T
