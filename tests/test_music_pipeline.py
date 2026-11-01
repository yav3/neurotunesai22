"""Tests for music_pipeline module: feature extraction, annotation, and batch processing."""
import pytest
import numpy as np


def _sine(freq: float = 440.0, sr: int = 22050, dur: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


class TestSpectralFeatures:
    def test_basic_extraction(self):
        from models.music_pipeline.feature_pipeline import extract_spectral_features, PipelineConfig
        signal = _sine(440.0)
        cfg = PipelineConfig()
        feats = extract_spectral_features(signal, 22050, cfg)
        assert "centroid" in feats
        assert "rolloff" in feats
        assert "zcr" in feats
        assert feats["centroid"] > 0

    def test_silent_signal(self):
        from models.music_pipeline.feature_pipeline import extract_spectral_features, PipelineConfig
        signal = np.zeros(4096)
        feats = extract_spectral_features(signal, 22050, PipelineConfig())
        assert feats["centroid"] == 0.0

    def test_short_signal(self):
        from models.music_pipeline.feature_pipeline import extract_spectral_features, PipelineConfig
        signal = np.array([0.1, 0.2, 0.3])
        feats = extract_spectral_features(signal, 22050, PipelineConfig())
        assert feats["centroid"] == 0.0


class TestRhythmFeatures:
    def test_rhythmic_signal(self):
        from models.music_pipeline.feature_pipeline import extract_rhythm_features, PipelineConfig
        sr = 22050
        t = np.arange(sr * 2) / sr
        beat_freq = 2.0
        signal = np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t))
        feats = extract_rhythm_features(signal, sr, PipelineConfig())
        assert feats["tempo_estimate"] >= 0
        assert feats["beat_strength"] >= 0

    def test_empty_signal(self):
        from models.music_pipeline.feature_pipeline import extract_rhythm_features, PipelineConfig
        feats = extract_rhythm_features(np.zeros(512), 22050, PipelineConfig())
        assert feats["tempo_estimate"] == 0.0


class TestHarmonicFeatures:
    def test_chroma_extraction(self):
        from models.music_pipeline.feature_pipeline import extract_harmonic_features, PipelineConfig
        signal = _sine(440.0)
        feats = extract_harmonic_features(signal, 22050, PipelineConfig())
        assert len([k for k in feats if k.startswith("chroma_")]) == 12
        assert "estimated_key_idx" in feats
        assert "key_confidence" in feats


class TestTimbralFeatures:
    def test_brightness(self):
        from models.music_pipeline.feature_pipeline import extract_timbral_features, PipelineConfig
        signal = _sine(6000.0)
        feats = extract_timbral_features(signal, 22050, PipelineConfig())
        assert feats["brightness"] > 0

    def test_warmth(self):
        from models.music_pipeline.feature_pipeline import extract_timbral_features, PipelineConfig
        signal = _sine(200.0)
        feats = extract_timbral_features(signal, 22050, PipelineConfig())
        assert feats["warmth"] > 0


class TestEmotionFeatures:
    def test_vad_range(self):
        from models.music_pipeline.feature_pipeline import extract_emotion_features
        spectral = {"centroid": 2000.0, "key_confidence": 0.5}
        rhythm = {"tempo_estimate": 120.0, "beat_strength": 0.5, "pulse_clarity": 0.5}
        timbral = {"brightness": 0.5}
        feats = extract_emotion_features(spectral, rhythm, timbral)
        assert 0 <= feats["valence"] <= 1
        assert 0 <= feats["arousal"] <= 1
        assert 0 <= feats["dominance"] <= 1


class TestFeaturePipeline:
    def test_full_extraction(self):
        from models.music_pipeline.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        signal = _sine(440.0)
        result = pipe.extract(signal)
        assert result.spectral
        assert result.rhythm
        assert result.harmonic
        assert result.emotion
        assert result.metadata["duration_s"] > 0

    def test_to_vector(self):
        from models.music_pipeline.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        signal = _sine(440.0)
        result = pipe.extract(signal)
        vec = result.to_vector()
        assert isinstance(vec, np.ndarray)
        assert len(vec) > 0
        assert np.all(np.isfinite(vec))

    def test_dimension_names(self):
        from models.music_pipeline.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        result = pipe.extract(_sine())
        names = result.dimension_names()
        vec = result.to_vector()
        assert len(names) == len(vec)

    def test_batch(self):
        from models.music_pipeline.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        signals = [_sine(f) for f in [220, 440, 880]]
        results = pipe.extract_batch(signals)
        assert len(results) == 3

    def test_feature_matrix(self):
        from models.music_pipeline.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        signals = [_sine(f) for f in [220, 440]]
        mat = pipe.feature_matrix(signals)
        assert mat.shape[0] == 2
        assert mat.shape[1] > 0

    def test_selective_features(self):
        from models.music_pipeline.feature_pipeline import FeaturePipeline, PipelineConfig
        cfg = PipelineConfig(enable_rhythm=False, enable_emotion=False)
        pipe = FeaturePipeline(cfg)
        result = pipe.extract(_sine())
        assert result.spectral
        assert not result.rhythm
        assert not result.emotion


class TestAnnotationPipeline:
    def test_annotate(self):
        from models.music_pipeline.annotation import AnnotationPipeline
        pipe = AnnotationPipeline()
        signal = _sine(440.0)
        ann = pipe.annotate(signal)
        assert ann.key in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        assert ann.mood_quadrant in ["happy", "angry", "sad", "relaxed", "neutral"]
        assert isinstance(ann.tags, list)
        assert len(ann.tags) > 0

    def test_annotate_to_dict(self):
        from models.music_pipeline.annotation import AnnotationPipeline
        pipe = AnnotationPipeline()
        ann = pipe.annotate(_sine())
        d = ann.to_dict()
        assert "valence" in d
        assert "arousal" in d
        assert "tags" in d

    def test_batch_annotate(self):
        from models.music_pipeline.annotation import AnnotationPipeline
        pipe = AnnotationPipeline()
        signals = [_sine(f) for f in [220, 440, 880]]
        anns = pipe.annotate_batch(signals)
        assert len(anns) == 3

    def test_mood_quadrant_classification(self):
        from models.music_pipeline.annotation import classify_mood_quadrant
        assert classify_mood_quadrant(0.8, 0.8) == "happy"
        assert classify_mood_quadrant(0.2, 0.8) == "angry"
        assert classify_mood_quadrant(0.2, 0.2) == "sad"
        assert classify_mood_quadrant(0.8, 0.2) == "relaxed"

    def test_derive_tags(self):
        from models.music_pipeline.annotation import derive_tags, TrackAnnotation
        ann = TrackAnnotation(tempo_bpm=140, energy=0.8, brightness=0.6, danceability=0.7, valence=0.7, mood_quadrant="happy")
        tags = derive_tags(ann)
        assert "fast" in tags
        assert "high_energy" in tags
        assert "bright" in tags
        assert "danceable" in tags
        assert "positive" in tags
