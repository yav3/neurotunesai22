"""Tests for the end-to-end music pipeline: audio → features → encoder → predictions."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models.foundation.audio_feature_extractor import (
    AudioFeatureExtractor,
    ExtractedAudioFeatures,
    FOUNDATION_FEATURE_NAMES,
    _key_to_index,
    _key_to_camelot_number,
    _band_energy,
    _estimate_danceability,
)
from models.pipeline.end_to_end import (
    MusicPipeline,
    PipelineConfig,
    PipelinePrediction,
    MOOD_LABELS,
    VAD_LABELS,
)
from models.music_features.spectral import SpectralFeatures


# ---------------------------------------------------------------------------
# AudioFeatureExtractor
# ---------------------------------------------------------------------------

class TestAudioFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return AudioFeatureExtractor(sr=22050)

    @pytest.fixture
    def sine_440(self):
        sr = 22050
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        return np.sin(2 * np.pi * 440 * t)

    @pytest.fixture
    def white_noise(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(22050 * 2)

    def test_extract_returns_correct_shape(self, extractor, sine_440):
        result = extractor.extract(sine_440)
        assert isinstance(result, ExtractedAudioFeatures)
        assert result.features.shape == (16,)
        assert result.features.dtype == np.float32

    def test_extract_feature_names(self, extractor, sine_440):
        result = extractor.extract(sine_440)
        assert result.feature_names == FOUNDATION_FEATURE_NAMES
        assert len(result.feature_names) == 16

    def test_extract_metadata_keys(self, extractor, sine_440):
        result = extractor.extract(sine_440)
        assert "key" in result.metadata
        assert "mode" in result.metadata
        assert "key_confidence" in result.metadata
        assert "bpm_confidence" in result.metadata
        assert "duration_s" in result.metadata

    def test_extract_duration(self, extractor, sine_440):
        result = extractor.extract(sine_440)
        assert abs(result.metadata["duration_s"] - 2.0) < 0.01

    def test_extract_no_nans(self, extractor, sine_440):
        result = extractor.extract(sine_440)
        assert not np.any(np.isnan(result.features))

    def test_extract_no_infs(self, extractor, sine_440):
        result = extractor.extract(sine_440)
        assert not np.any(np.isinf(result.features))

    def test_extract_stereo_to_mono(self, extractor):
        sr = 22050
        stereo = np.random.default_rng(0).standard_normal((sr, 2))
        result = extractor.extract(stereo)
        assert result.features.shape == (16,)

    def test_extract_custom_sr(self, extractor, sine_440):
        result = extractor.extract(sine_440, sr=44100)
        assert result.metadata["duration_s"] == pytest.approx(1.0, abs=0.01)

    def test_extract_batch(self, extractor, sine_440, white_noise):
        batch = extractor.extract_batch([sine_440, white_noise])
        assert batch.shape == (2, 16)
        assert batch.dtype == np.float32

    def test_extract_short_audio(self, extractor):
        short = np.sin(np.linspace(0, 1, 1000))
        result = extractor.extract(short)
        assert result.features.shape == (16,)

    def test_extract_silence(self, extractor):
        silence = np.zeros(22050)
        result = extractor.extract(silence)
        assert result.features.shape == (16,)
        assert not np.any(np.isnan(result.features))

    def test_valence_arousal_dominance_bounded(self, extractor, sine_440):
        result = extractor.extract(sine_440)
        v_idx = FOUNDATION_FEATURE_NAMES.index("valence")
        a_idx = FOUNDATION_FEATURE_NAMES.index("arousal")
        d_idx = FOUNDATION_FEATURE_NAMES.index("dominance")
        for idx in [v_idx, a_idx, d_idx]:
            assert 0.0 <= result.features[idx] <= 1.0

    def test_key_is_valid_index(self, extractor, sine_440):
        result = extractor.extract(sine_440)
        key_idx = FOUNDATION_FEATURE_NAMES.index("key")
        assert 0 <= result.features[key_idx] < 12

    def test_scale_is_binary(self, extractor, sine_440):
        result = extractor.extract(sine_440)
        scale_idx = FOUNDATION_FEATURE_NAMES.index("scale")
        assert result.features[scale_idx] in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_key_to_index_basic(self):
        assert _key_to_index("C") == 0
        assert _key_to_index("A") == 9
        assert _key_to_index("B") == 11

    def test_key_to_index_case_insensitive(self):
        assert _key_to_index("c") == 0
        assert _key_to_index("a") == 9

    def test_key_to_index_sharp(self):
        assert _key_to_index("C#") == 1
        assert _key_to_index("F#") == 6

    def test_key_to_index_unknown(self):
        assert _key_to_index("X") == 0

    def test_key_to_camelot_number(self):
        val = _key_to_camelot_number("A", "minor")
        assert isinstance(val, int)
        assert val > 0

    def test_key_to_camelot_number_invalid(self):
        val = _key_to_camelot_number("ZZZ", "unknown")
        assert val == 0

    def test_band_energy_bass(self):
        spec = SpectralFeatures(chroma=[1.0] * 12)
        assert _band_energy(spec, "bass") == pytest.approx(1.0)

    def test_band_energy_mid(self):
        spec = SpectralFeatures(chroma=[0.0] * 4 + [0.5] * 4 + [0.0] * 4)
        assert _band_energy(spec, "mid") == pytest.approx(0.5)

    def test_band_energy_treble(self):
        spec = SpectralFeatures(chroma=[0.0] * 8 + [0.8] * 4)
        assert _band_energy(spec, "treble") == pytest.approx(0.8)

    def test_band_energy_empty_chroma(self):
        spec = SpectralFeatures(chroma=[])
        assert _band_energy(spec, "bass") == 0.0

    def test_estimate_danceability_range(self):
        from models.music_features.rhythm import RhythmFeatures
        from models.music_features.dynamics import DynamicsFeatures
        rhythm = RhythmFeatures(bpm=120.0, beat_regularity=0.8)
        dyn = DynamicsFeatures(rms_mean=0.5)
        d = _estimate_danceability(rhythm, dyn)
        assert 0.0 <= d <= 1.0

    def test_estimate_danceability_high(self):
        from models.music_features.rhythm import RhythmFeatures
        from models.music_features.dynamics import DynamicsFeatures
        rhythm = RhythmFeatures(bpm=120.0, beat_regularity=1.0)
        dyn = DynamicsFeatures(rms_mean=1.0)
        d = _estimate_danceability(rhythm, dyn)
        assert d > 0.7

    def test_estimate_danceability_zero_bpm(self):
        from models.music_features.rhythm import RhythmFeatures
        from models.music_features.dynamics import DynamicsFeatures
        rhythm = RhythmFeatures(bpm=0.0, beat_regularity=0.0)
        dyn = DynamicsFeatures(rms_mean=0.0)
        d = _estimate_danceability(rhythm, dyn)
        assert d == 0.0


# ---------------------------------------------------------------------------
# MusicPipeline
# ---------------------------------------------------------------------------

class TestMusicPipeline:
    @pytest.fixture
    def pipeline(self):
        return MusicPipeline(PipelineConfig(use_frontier=False))

    @pytest.fixture
    def pipeline_with_frontier(self):
        return MusicPipeline(PipelineConfig(use_frontier=True))

    @pytest.fixture
    def sine_audio(self):
        sr = 22050
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        return np.sin(2 * np.pi * 440 * t)

    def test_pipeline_init(self, pipeline):
        assert pipeline.config.sr == 22050
        assert pipeline.config.hidden_dim == 128
        assert pipeline.model is not None
        assert pipeline.extractor is not None

    def test_predict_audio_returns_prediction(self, pipeline, sine_audio):
        pred = pipeline.predict_audio(sine_audio)
        assert isinstance(pred, PipelinePrediction)

    def test_predict_audio_mood_scores(self, pipeline, sine_audio):
        pred = pipeline.predict_audio(sine_audio)
        assert len(pred.mood_scores) == len(MOOD_LABELS)
        for label in MOOD_LABELS:
            assert label in pred.mood_scores
            assert 0.0 <= pred.mood_scores[label] <= 1.0

    def test_predict_audio_vad_scores(self, pipeline, sine_audio):
        pred = pipeline.predict_audio(sine_audio)
        assert len(pred.vad_scores) == len(VAD_LABELS)
        for label in VAD_LABELS:
            assert label in pred.vad_scores

    def test_predict_audio_dominant_mood(self, pipeline, sine_audio):
        pred = pipeline.predict_audio(sine_audio)
        assert pred.dominant_mood in MOOD_LABELS
        assert pred.mood_scores[pred.dominant_mood] == max(pred.mood_scores.values())

    def test_predict_audio_features_dict(self, pipeline, sine_audio):
        pred = pipeline.predict_audio(sine_audio)
        assert len(pred.features) == 16
        for name in FOUNDATION_FEATURE_NAMES:
            assert name in pred.features

    def test_predict_audio_no_frontier_by_default(self, pipeline, sine_audio):
        pred = pipeline.predict_audio(sine_audio)
        assert pred.frontier_answer is None

    def test_predict_audio_with_frontier(self, pipeline_with_frontier, sine_audio):
        pred = pipeline_with_frontier.predict_audio(
            sine_audio,
            hypothesis="This track induces calm states",
        )
        assert pred.frontier_answer is not None

    def test_predict_audio_frontier_without_hypothesis(self, pipeline_with_frontier, sine_audio):
        pred = pipeline_with_frontier.predict_audio(sine_audio, hypothesis="")
        assert pred.frontier_answer is None

    def test_predict_features_raw(self, pipeline):
        features = np.random.default_rng(42).standard_normal(16).astype(np.float32)
        pred = pipeline.predict_features(features)
        assert isinstance(pred, PipelinePrediction)
        assert len(pred.mood_scores) == len(MOOD_LABELS)

    def test_predict_batch(self, pipeline, sine_audio):
        noise = np.random.default_rng(0).standard_normal(22050 * 2)
        preds = pipeline.predict_batch([sine_audio, noise])
        assert len(preds) == 2
        assert all(isinstance(p, PipelinePrediction) for p in preds)

    def test_predict_deterministic(self, pipeline, sine_audio):
        pred1 = pipeline.predict_audio(sine_audio)
        pred2 = pipeline.predict_audio(sine_audio)
        for label in MOOD_LABELS:
            assert pred1.mood_scores[label] == pytest.approx(
                pred2.mood_scores[label], abs=1e-5
            )

    def test_train_step_returns_losses(self, pipeline, sine_audio):
        noise = np.random.default_rng(0).standard_normal(22050 * 2)
        target_moods = np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32)
        target_vad = np.array([[0.8, 0.3, 0.5], [0.2, 0.1, 0.4]], dtype=np.float32)
        losses = pipeline.train_step([sine_audio, noise], target_moods, target_vad)
        assert "total_loss" in losses
        assert "mood_loss" in losses
        assert "vad_loss" in losses
        assert losses["total_loss"] > 0
        assert losses["total_loss"] == pytest.approx(
            losses["mood_loss"] + losses["vad_loss"], abs=1e-5
        )

    def test_pipeline_custom_config(self):
        cfg = PipelineConfig(hidden_dim=64, num_layers=2, num_heads=2)
        pipe = MusicPipeline(cfg)
        assert pipe.config.hidden_dim == 64
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 22050 * 2))
        pred = pipe.predict_audio(audio)
        assert isinstance(pred, PipelinePrediction)

    def test_pipeline_metadata_passthrough(self, pipeline, sine_audio):
        pred = pipeline.predict_audio(sine_audio)
        assert "key" in pred.metadata
        assert "mode" in pred.metadata
        assert pred.metadata["mode"] in ("major", "minor")


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.sr == 22050
        assert cfg.hidden_dim == 128
        assert cfg.num_layers == 4
        assert cfg.num_heads == 4
        assert cfg.use_frontier is True
        assert cfg.device == "cpu"

    def test_custom(self):
        cfg = PipelineConfig(sr=44100, hidden_dim=256, use_frontier=False)
        assert cfg.sr == 44100
        assert cfg.hidden_dim == 256
        assert cfg.use_frontier is False


# ---------------------------------------------------------------------------
# FOUNDATION_FEATURE_NAMES consistency
# ---------------------------------------------------------------------------

class TestFeatureNameConsistency:
    def test_foundation_names_length(self):
        assert len(FOUNDATION_FEATURE_NAMES) == 16

    def test_foundation_names_match_bridge(self):
        from models.foundation.music_frontier_bridge import AUDIO_FEATURE_NAMES
        assert FOUNDATION_FEATURE_NAMES == AUDIO_FEATURE_NAMES[:16]

    def test_mood_labels_count(self):
        assert len(MOOD_LABELS) == 5

    def test_vad_labels_count(self):
        assert len(VAD_LABELS) == 3
