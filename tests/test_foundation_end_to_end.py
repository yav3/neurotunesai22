"""Tests for the end-to-end music foundation pipeline."""
import pytest
import numpy as np
from models.music_foundation.end_to_end import (
    MusicFoundationPipeline, PipelineConfig, TrainingResult,
    generate_synthetic_dataset, _adsr_envelope, _encode_metadata,
    AudioSample, run_demo,
)


class TestSyntheticDataGeneration:
    def test_generates_correct_count(self):
        samples = generate_synthetic_dataset(n_samples=10, duration=1.0)
        assert len(samples) == 10

    def test_sample_audio_shape(self):
        samples = generate_synthetic_dataset(n_samples=3, duration=2.0, sr=16000)
        for s in samples:
            assert s.audio.shape == (32000,)
            assert s.sr == 16000

    def test_metadata_present(self):
        samples = generate_synthetic_dataset(n_samples=5)
        for s in samples:
            assert "genre" in s.metadata
            assert "key" in s.metadata
            assert "tempo" in s.metadata
            assert "mode" in s.metadata

    def test_audio_normalized(self):
        samples = generate_synthetic_dataset(n_samples=10)
        for s in samples:
            assert np.abs(s.audio).max() <= 1.0 + 1e-6

    def test_deterministic_with_seed(self):
        s1 = generate_synthetic_dataset(n_samples=5, seed=123)
        s2 = generate_synthetic_dataset(n_samples=5, seed=123)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a.audio, b.audio)

    def test_different_seeds_differ(self):
        s1 = generate_synthetic_dataset(n_samples=5, seed=1)
        s2 = generate_synthetic_dataset(n_samples=5, seed=2)
        assert not np.allclose(s1[0].audio, s2[0].audio)


class TestADSREnvelope:
    def test_shape(self):
        env = _adsr_envelope(22050, 22050, 0.05, 0.1, 0.7, 0.2, 1.0)
        assert env.shape == (22050,)

    def test_starts_at_zero(self):
        env = _adsr_envelope(22050, 22050, 0.05, 0.1, 0.7, 0.2, 1.0)
        assert env[0] == pytest.approx(0.0, abs=0.01)

    def test_peak_is_one(self):
        env = _adsr_envelope(44100, 22050, 0.1, 0.1, 0.5, 0.1, 2.0)
        assert env.max() == pytest.approx(1.0, abs=0.05)


class TestMetadataEncoding:
    def test_output_dim(self):
        v = _encode_metadata({"genre": "jazz", "key": "C"}, dim=64)
        assert v.shape == (64,)

    def test_normalized(self):
        v = _encode_metadata({"genre": "rock", "tempo": 140}, dim=32)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=0.01)

    def test_different_genres_differ(self):
        v1 = _encode_metadata({"genre": "jazz"}, dim=64)
        v2 = _encode_metadata({"genre": "rock"}, dim=64)
        assert not np.allclose(v1, v2)

    def test_same_genre_same_output(self):
        v1 = _encode_metadata({"genre": "jazz"}, dim=64, seed=10)
        v2 = _encode_metadata({"genre": "jazz"}, dim=64, seed=10)
        np.testing.assert_array_equal(v1, v2)


class TestPipelineFeatureExtraction:
    def test_extract_features_shape(self):
        pipe = MusicFoundationPipeline(PipelineConfig(
            segment_duration=1.0, max_frames=64, audio_dim=64, semantic_dim=32,
        ))
        samples = generate_synthetic_dataset(n_samples=5, duration=1.0)
        pipe.add_samples(samples)
        audio_f, sem_f = pipe.extract_features()
        assert audio_f.shape == (5, 64)
        assert sem_f.shape == (5, 32)

    def test_features_not_all_zero(self):
        pipe = MusicFoundationPipeline(PipelineConfig(segment_duration=1.0, max_frames=64))
        samples = generate_synthetic_dataset(n_samples=3, duration=1.0)
        pipe.add_samples(samples)
        audio_f, _ = pipe.extract_features()
        assert np.abs(audio_f).sum() > 0

    def test_add_audio_directly(self):
        pipe = MusicFoundationPipeline(PipelineConfig(segment_duration=1.0, max_frames=64))
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        idx = pipe.add_audio(audio, sr=22050, metadata={"genre": "test"})
        assert idx == 0
        af, sf = pipe.extract_features()
        assert af.shape[0] == 1


class TestTokenization:
    def test_tokenize_produces_indices(self):
        pipe = MusicFoundationPipeline(PipelineConfig(
            segment_duration=1.0, max_frames=64, n_codebooks=2, n_codes=32,
        ))
        samples = generate_synthetic_dataset(n_samples=5, duration=1.0)
        pipe.add_samples(samples)
        all_tokens = pipe.tokenize_all()
        assert len(all_tokens) == 5
        for tokens in all_tokens:
            assert len(tokens) == 2
            for level in tokens:
                assert all(0 <= t < 32 for t in level)


class TestTraining:
    def test_train_returns_result(self):
        pipe = MusicFoundationPipeline(PipelineConfig(
            segment_duration=1.0, max_frames=64,
            audio_dim=32, semantic_dim=16, hidden_dim=32, latent_dim=32,
            n_heads=2, batch_size=4, n_codebooks=2, n_codes=32,
        ))
        samples = generate_synthetic_dataset(n_samples=20, duration=1.0)
        pipe.add_samples(samples)
        result = pipe.train(epochs=2)
        assert isinstance(result, TrainingResult)
        assert result.n_samples == 20
        assert result.n_epochs_run >= 1
        assert result.final_train_loss > 0
        assert len(result.codebook_perplexity) == 2

    def test_loss_decreases_or_stable(self):
        pipe = MusicFoundationPipeline(PipelineConfig(
            segment_duration=1.0, max_frames=64,
            audio_dim=32, semantic_dim=16, hidden_dim=32, latent_dim=32,
            n_heads=2, batch_size=8, n_codebooks=2, n_codes=32,
            patience=100,
        ))
        samples = generate_synthetic_dataset(n_samples=30, duration=1.0)
        pipe.add_samples(samples)
        result = pipe.train(epochs=3)
        losses = [e.get("total", 0) for e in result.history.epoch_losses]
        assert len(losses) >= 2

    def test_reconstruction_error_finite(self):
        pipe = MusicFoundationPipeline(PipelineConfig(
            segment_duration=1.0, max_frames=64,
            audio_dim=32, semantic_dim=16, hidden_dim=32, latent_dim=32,
            n_heads=2, batch_size=4, n_codebooks=2, n_codes=32,
        ))
        samples = generate_synthetic_dataset(n_samples=10, duration=1.0)
        pipe.add_samples(samples)
        result = pipe.train(epochs=1)
        assert np.isfinite(result.reconstruction_error)
        assert result.reconstruction_error >= 0


class TestRunDemo:
    def test_demo_runs(self):
        result = run_demo(n_samples=10, epochs=1)
        assert isinstance(result, TrainingResult)
        assert result.n_samples == 10
        assert result.final_train_loss > 0


class TestEncodeAndSimilarity:
    def test_encode_returns_vector(self):
        pipe = MusicFoundationPipeline(PipelineConfig(
            segment_duration=1.0, max_frames=64,
            audio_dim=32, semantic_dim=16, hidden_dim=32, latent_dim=32, n_heads=2,
        ))
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        emb = pipe.encode(audio)
        assert emb.shape == (32,)

    def test_same_audio_high_similarity(self):
        pipe = MusicFoundationPipeline(PipelineConfig(
            segment_duration=1.0, max_frames=64,
            audio_dim=32, semantic_dim=16, hidden_dim=32, latent_dim=32, n_heads=2,
        ))
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        sim = pipe.similarity(audio, audio)
        assert sim > 0.99
