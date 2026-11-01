"""End-to-end integration test: raw audio → features → domain analysis → training → inference.

This is the proof that the foundational-frontier model works as a complete system.
"""
import pytest
import numpy as np

from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
from models.music_foundation.inference import MusicInferencePipeline, InferenceConfig
from models.music_foundation.training import Trainer, TrainingConfig
from models.music_foundation.pretraining import PretrainingObjective
from models.music_pipeline.feature_pipeline import FeaturePipeline


def _generate_tone(freq: float, dur: float = 1.0, sr: int = 22050) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _generate_chord(freqs: list[float], dur: float = 1.0, sr: int = 22050) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    signal = sum(0.3 * np.sin(2 * np.pi * f * t) for f in freqs)
    return signal.astype(np.float32)


class TestEndToEnd:
    """Full pipeline: audio → extraction → encoding → similarity."""

    @pytest.fixture
    def pipeline(self):
        cfg = InferenceConfig(
            encoder=EncoderConfig(latent_dim=64, hidden_dim=128, audio_dim=64, semantic_dim=32),
            n_genres=5, n_moods=4, enable_domain=True,
        )
        return MusicInferencePipeline(cfg)

    def test_single_track_inference(self, pipeline):
        audio = _generate_tone(440.0, 2.0)
        result = pipeline.infer(audio, 22050)
        assert result.embedding.shape == (64,)
        assert result.genre_logits.shape == (5,)
        assert result.mood_logits.shape == (4,)
        assert result.vad.shape == (3,)
        assert not np.any(np.isnan(result.embedding))

    def test_distinct_inputs_produce_distinct_embeddings(self, pipeline):
        a440 = _generate_tone(440.0, 2.0)
        a110 = _generate_tone(110.0, 2.0)
        r440 = pipeline.infer(a440, 22050)
        r110 = pipeline.infer(a110, 22050)
        assert not np.allclose(r440.embedding, r110.embedding, atol=1e-3)

    def test_chord_vs_tone(self, pipeline):
        tone = _generate_tone(440.0, 2.0)
        chord = _generate_chord([261.63, 329.63, 392.00], 2.0)
        r_tone = pipeline.infer(tone, 22050)
        r_chord = pipeline.infer(chord, 22050)
        assert r_tone.embedding.shape == r_chord.embedding.shape
        sim = r_tone.similarity(r_chord)
        assert -1.0 <= sim <= 1.0

    def test_domain_analysis_integrated(self, pipeline):
        audio = _generate_tone(440.0, 2.0)
        result = pipeline.infer(audio, 22050)
        da = result.domain_analysis
        assert da is not None
        assert isinstance(da.cognition.consonance_mean, float)
        assert 0.0 <= da.emotion.valence <= 1.0
        assert da.theory.tempo_bpm >= 0


class TestTrainThenInfer:
    """Train on synthetic data, then verify inference still works and embeddings changed."""

    def test_training_changes_embeddings(self):
        enc_cfg = EncoderConfig(audio_dim=16, semantic_dim=8, hidden_dim=32, latent_dim=16, n_heads=2)
        model = MusicFoundationModel(enc_cfg)

        rng = np.random.RandomState(42)
        audio_data = rng.randn(32, 16)
        semantic_data = rng.randn(32, 8)

        emb_before = model.encode(audio_data[:4], semantic_data[:4]).copy()

        train_cfg = TrainingConfig(batch_size=16, epochs=2, lr=1e-2, warmup_steps=2, total_steps=4)
        trainer = Trainer(model, train_cfg)
        metrics = trainer.fit(audio_data, semantic_data)

        emb_after = model.encode(audio_data[:4], semantic_data[:4])
        assert not np.allclose(emb_before, emb_after, atol=1e-6)
        assert metrics.epoch == 2
        assert len(metrics.loss_history) > 0

    def test_loss_is_finite(self):
        enc_cfg = EncoderConfig(audio_dim=16, semantic_dim=8, hidden_dim=32, latent_dim=16, n_heads=2)
        model = MusicFoundationModel(enc_cfg)
        rng = np.random.RandomState(0)
        audio = rng.randn(16, 16)
        semantic = rng.randn(16, 8)
        train_cfg = TrainingConfig(batch_size=8, epochs=1, lr=1e-3, warmup_steps=1, total_steps=2)
        trainer = Trainer(model, train_cfg)
        metrics = trainer.fit(audio, semantic)
        assert all(np.isfinite(l) for l in metrics.loss_history)


class TestFeatureToEmbeddingRoundtrip:
    """Verify that FeaturePipeline output feeds cleanly into the encoder."""

    def test_pipeline_features_encode(self):
        fp = FeaturePipeline()
        audio = _generate_tone(440.0, 2.0)
        feats = fp.extract(audio, 22050)
        vec = feats.to_vector()

        enc_cfg = EncoderConfig(audio_dim=len(vec), semantic_dim=8, hidden_dim=64, latent_dim=32)
        model = MusicFoundationModel(enc_cfg)
        emb = model.encode_audio(vec)
        assert emb.ndim == 2
        assert emb.shape[-1] == 32
        assert not np.any(np.isnan(emb))


class TestMultiTrackBatch:
    """Batch processing of multiple tracks through the full pipeline."""

    def test_batch_consistency(self):
        cfg = InferenceConfig(
            encoder=EncoderConfig(latent_dim=32, hidden_dim=64, audio_dim=32, semantic_dim=16),
            enable_domain=False,
        )
        pipe = MusicInferencePipeline(cfg)
        freqs = [220, 330, 440, 554, 660]
        audios = [_generate_tone(f, 1.0) for f in freqs]

        results = pipe.infer_batch(audios, 22050)
        assert len(results) == 5

        mat = pipe.embedding_matrix(audios, 22050)
        assert mat.shape == (5, 32)

        for i, r in enumerate(results):
            np.testing.assert_allclose(mat[i], r.embedding, atol=1e-10)
