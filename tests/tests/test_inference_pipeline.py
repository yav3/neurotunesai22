"""Tests for the unified MusicInferencePipeline: audio → embedding + predictions."""
import pytest
import numpy as np

from models.music_foundation.inference import (
    MusicInferencePipeline, InferenceConfig, InferenceResult,
    GenreHead, MoodHead, VADHead,
)
from models.music_foundation.encoder import EncoderConfig


def _tone(freq: float = 440.0, dur: float = 2.0, sr: int = 22050) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestTaskHeads:
    def test_genre_head_shape(self):
        head = GenreHead(128, 10)
        x = np.random.randn(1, 128)
        out = head.forward(x)
        assert out.shape == (1, 10)

    def test_mood_head_shape(self):
        head = MoodHead(128, 8)
        x = np.random.randn(1, 128)
        out = head.forward(x)
        assert out.shape == (1, 8)

    def test_vad_head_bounded(self):
        head = VADHead(128)
        x = np.random.randn(1, 128) * 10
        out = head.forward(x)
        assert out.shape == (1, 3)
        assert np.all(out >= 0) and np.all(out <= 1)


class TestInferenceResult:
    def test_similarity_identical(self):
        emb = np.random.randn(128)
        r1 = InferenceResult(embedding=emb)
        r2 = InferenceResult(embedding=emb)
        assert abs(r1.similarity(r2) - 1.0) < 1e-6

    def test_similarity_orthogonal(self):
        a = np.zeros(128); a[0] = 1.0
        b = np.zeros(128); b[1] = 1.0
        r1 = InferenceResult(embedding=a)
        r2 = InferenceResult(embedding=b)
        assert abs(r1.similarity(r2)) < 1e-6


class TestMusicInferencePipeline:
    @pytest.fixture
    def pipeline(self):
        cfg = InferenceConfig(
            encoder=EncoderConfig(latent_dim=64, hidden_dim=128, audio_dim=64, semantic_dim=32),
            n_genres=5,
            n_moods=4,
            enable_domain=True,
        )
        return MusicInferencePipeline(cfg)

    @pytest.fixture
    def tone_audio(self):
        return _tone(440.0, 2.0, 22050)

    def test_infer_returns_result(self, pipeline, tone_audio):
        result = pipeline.infer(tone_audio, 22050)
        assert isinstance(result, InferenceResult)

    def test_embedding_shape(self, pipeline, tone_audio):
        result = pipeline.infer(tone_audio, 22050)
        assert result.embedding.ndim == 1
        assert result.embedding.shape[0] == 64

    def test_genre_logits_shape(self, pipeline, tone_audio):
        result = pipeline.infer(tone_audio, 22050)
        assert result.genre_logits.shape == (5,)

    def test_mood_logits_shape(self, pipeline, tone_audio):
        result = pipeline.infer(tone_audio, 22050)
        assert result.mood_logits.shape == (4,)

    def test_vad_bounded(self, pipeline, tone_audio):
        result = pipeline.infer(tone_audio, 22050)
        assert result.vad.shape == (3,)
        assert np.all(result.vad >= 0) and np.all(result.vad <= 1)

    def test_domain_analysis_present(self, pipeline, tone_audio):
        result = pipeline.infer(tone_audio, 22050)
        assert result.domain_analysis is not None

    def test_tempo_positive(self, pipeline, tone_audio):
        result = pipeline.infer(tone_audio, 22050)
        assert result.tempo_bpm >= 0

    def test_metadata_populated(self, pipeline, tone_audio):
        result = pipeline.infer(tone_audio, 22050)
        assert result.metadata["sr"] == 22050
        assert result.metadata["n_samples"] == len(tone_audio)

    def test_no_nans(self, pipeline, tone_audio):
        result = pipeline.infer(tone_audio, 22050)
        assert not np.any(np.isnan(result.embedding))
        assert not np.any(np.isnan(result.genre_logits))
        assert not np.any(np.isnan(result.vad))

    def test_stereo_handled(self, pipeline):
        sr = 22050
        stereo = np.random.randn(2 * sr, 2).astype(np.float32) * 0.1
        result = pipeline.infer(stereo, sr)
        assert isinstance(result, InferenceResult)
        assert result.embedding.ndim == 1

    def test_different_tones_different_embeddings(self, pipeline):
        a = _tone(220.0, 1.0, 22050)
        b = _tone(880.0, 1.0, 22050)
        ra = pipeline.infer(a, 22050)
        rb = pipeline.infer(b, 22050)
        assert not np.allclose(ra.embedding, rb.embedding, atol=1e-3)

    def test_similarity_self_is_one(self, pipeline, tone_audio):
        r = pipeline.infer(tone_audio, 22050)
        assert abs(r.similarity(r) - 1.0) < 1e-6


class TestBatchInference:
    def test_infer_batch(self):
        cfg = InferenceConfig(
            encoder=EncoderConfig(latent_dim=64, hidden_dim=128, audio_dim=64, semantic_dim=32),
            enable_domain=False,
        )
        pipe = MusicInferencePipeline(cfg)
        audios = [_tone(f, 1.0) for f in [220, 440, 880]]
        results = pipe.infer_batch(audios, 22050)
        assert len(results) == 3
        for r in results:
            assert r.embedding.shape == (64,)

    def test_embedding_matrix(self):
        cfg = InferenceConfig(
            encoder=EncoderConfig(latent_dim=64, hidden_dim=128, audio_dim=64, semantic_dim=32),
            enable_domain=False,
        )
        pipe = MusicInferencePipeline(cfg)
        audios = [_tone(f, 1.0) for f in [220, 440]]
        mat = pipe.embedding_matrix(audios, 22050)
        assert mat.shape == (2, 64)


class TestDomainDisabled:
    def test_no_domain_still_works(self):
        cfg = InferenceConfig(
            encoder=EncoderConfig(latent_dim=64, hidden_dim=128, audio_dim=64, semantic_dim=32),
            enable_domain=False,
        )
        pipe = MusicInferencePipeline(cfg)
        audio = _tone(440.0, 1.0, 22050)
        result = pipe.infer(audio, 22050)
        assert result.domain_analysis is None
        assert result.embedding.shape == (64,)
        assert not np.any(np.isnan(result.embedding))
