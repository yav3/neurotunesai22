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


# --- Signal quality ---

class TestPeakAmplitude:
    def test_sine(self):
        from models.music_pipeline.quality import peak_amplitude
        t = np.arange(1000) / 1000.0
        sig = 0.8 * np.sin(2 * np.pi * 10 * t)
        assert abs(peak_amplitude(sig) - 0.8) < 0.01

    def test_empty(self):
        from models.music_pipeline.quality import peak_amplitude
        assert peak_amplitude(np.array([])) == 0.0


class TestRmsLevel:
    def test_constant(self):
        from models.music_pipeline.quality import rms_level
        sig = np.ones(100) * 0.5
        assert abs(rms_level(sig) - 0.5) < 1e-6

    def test_silence(self):
        from models.music_pipeline.quality import rms_level
        assert rms_level(np.zeros(100)) == 0.0


class TestRmsToDb:
    def test_unity(self):
        from models.music_pipeline.quality import rms_to_db
        assert abs(rms_to_db(1.0)) < 1e-6

    def test_half(self):
        from models.music_pipeline.quality import rms_to_db
        db = rms_to_db(0.5)
        assert abs(db - (-6.02)) < 0.1

    def test_silence(self):
        from models.music_pipeline.quality import rms_to_db
        assert rms_to_db(0.0) == -100.0


class TestClippingRatio:
    def test_no_clipping(self):
        from models.music_pipeline.quality import clipping_ratio
        sig = np.ones(100) * 0.5
        assert clipping_ratio(sig) == 0.0

    def test_all_clipped(self):
        from models.music_pipeline.quality import clipping_ratio
        sig = np.ones(100) * 1.0
        assert clipping_ratio(sig) == 1.0


class TestSilenceRatio:
    def test_loud_signal(self):
        from models.music_pipeline.quality import silence_ratio
        sig = np.ones(4096) * 0.5
        assert silence_ratio(sig, sr=22050, frame_length=1024) == 0.0

    def test_silent_signal(self):
        from models.music_pipeline.quality import silence_ratio
        sig = np.zeros(4096)
        assert silence_ratio(sig, sr=22050, frame_length=1024) == 1.0


class TestEstimateSnr:
    def test_signal_with_pauses(self):
        from models.music_pipeline.quality import estimate_snr
        sig = np.zeros(22050)
        sig[:11025] = 0.5 * np.sin(np.arange(11025) * 0.1)
        snr = estimate_snr(sig, frame_length=1024)
        assert snr > 10

    def test_returns_float(self):
        from models.music_pipeline.quality import estimate_snr
        rng = np.random.RandomState(42)
        sig = rng.randn(22050)
        snr = estimate_snr(sig)
        assert isinstance(snr, float)


class TestDcOffset:
    def test_zero_mean(self):
        from models.music_pipeline.quality import dc_offset
        t = np.arange(1000) / 1000.0
        sig = np.sin(2 * np.pi * 10 * t)
        assert abs(dc_offset(sig)) < 0.01

    def test_offset(self):
        from models.music_pipeline.quality import dc_offset
        sig = np.ones(100) * 0.3
        assert abs(dc_offset(sig) - 0.3) < 1e-6


class TestCrestFactor:
    def test_square_wave(self):
        from models.music_pipeline.quality import crest_factor
        sig = np.ones(100) * 0.5
        assert abs(crest_factor(sig) - 1.0) < 1e-6

    def test_impulsive(self):
        from models.music_pipeline.quality import crest_factor
        sig = np.zeros(1000)
        sig[500] = 1.0
        assert crest_factor(sig) > 10


class TestDynamicRange:
    def test_constant(self):
        from models.music_pipeline.quality import dynamic_range
        sig = np.ones(4096) * 0.5
        assert dynamic_range(sig) == 0.0

    def test_varied(self):
        from models.music_pipeline.quality import dynamic_range
        sig = np.concatenate([np.ones(2048) * 0.01, np.ones(2048) * 0.5])
        dr = dynamic_range(sig, frame_length=512)
        assert dr > 10


class TestAssessQuality:
    def test_clean_sine(self):
        from models.music_pipeline.quality import assess_quality
        sig = np.zeros(22050)
        sig[:11025] = 0.5 * np.sin(np.arange(11025) * 0.1)
        report = assess_quality(sig, sr=22050)
        assert report.clipping_ratio == 0.0
        assert report.duration_s == 1.0
        assert report.peak_amplitude > 0

    def test_clipped_signal(self):
        from models.music_pipeline.quality import assess_quality
        sig = np.ones(22050)
        report = assess_quality(sig, sr=22050)
        assert "clipping" in report.issues
        assert not report.is_valid

    def test_silent_signal(self):
        from models.music_pipeline.quality import assess_quality
        sig = np.zeros(22050)
        report = assess_quality(sig, sr=22050)
        assert "mostly_silent" in report.issues


# --- Normalization ---

class TestZScoreNormalizer:
    def test_fit_transform(self):
        from models.music_pipeline.normalization import ZScoreNormalizer
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5) * 3 + 10
        norm = ZScoreNormalizer()
        Z = norm.fit_transform(X)
        assert abs(Z.mean()) < 0.1
        assert abs(Z.std() - 1.0) < 0.1

    def test_inverse(self):
        from models.music_pipeline.normalization import ZScoreNormalizer
        rng = np.random.RandomState(42)
        X = rng.randn(30, 4) * 2 + 5
        norm = ZScoreNormalizer()
        Z = norm.fit_transform(X)
        X_back = norm.inverse_transform(Z)
        assert np.allclose(X, X_back, atol=1e-10)

    def test_not_fitted_raises(self):
        from models.music_pipeline.normalization import ZScoreNormalizer
        norm = ZScoreNormalizer()
        with pytest.raises(RuntimeError):
            norm.transform(np.ones((5, 3)))

    def test_statistics(self):
        from models.music_pipeline.normalization import ZScoreNormalizer
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3)
        norm = ZScoreNormalizer()
        norm.fit(X)
        assert norm.statistics is not None
        assert norm.statistics.n_features == 3
        assert norm.statistics.n_samples == 20


class TestMinMaxNormalizer:
    def test_default_range(self):
        from models.music_pipeline.normalization import MinMaxNormalizer
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        norm = MinMaxNormalizer()
        Z = norm.fit_transform(X)
        assert Z.min() >= 0.0 - 1e-10
        assert Z.max() <= 1.0 + 1e-10

    def test_custom_range(self):
        from models.music_pipeline.normalization import MinMaxNormalizer
        X = np.array([[0, 10], [5, 20], [10, 30]], dtype=float)
        norm = MinMaxNormalizer(feature_range=(-1.0, 1.0))
        Z = norm.fit_transform(X)
        assert Z.min() >= -1.0 - 1e-10
        assert Z.max() <= 1.0 + 1e-10

    def test_inverse(self):
        from models.music_pipeline.normalization import MinMaxNormalizer
        X = np.array([[2, 3], [4, 5], [6, 7]], dtype=float)
        norm = MinMaxNormalizer()
        Z = norm.fit_transform(X)
        X_back = norm.inverse_transform(Z)
        assert np.allclose(X, X_back, atol=1e-10)


class TestRobustNormalizer:
    def test_median_centered(self):
        from models.music_pipeline.normalization import RobustNormalizer
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3) * 5 + 100
        norm = RobustNormalizer()
        Z = norm.fit_transform(X)
        assert abs(np.median(Z[:, 0])) < 0.1

    def test_outlier_robust(self):
        from models.music_pipeline.normalization import RobustNormalizer
        X = np.array([[1], [2], [3], [4], [5], [1000]], dtype=float)
        norm = RobustNormalizer()
        Z = norm.fit_transform(X)
        assert abs(Z[5, 0]) > 100


class TestFeatureSelector:
    def test_removes_constant(self):
        from models.music_pipeline.normalization import FeatureSelector
        X = np.array([[1, 5, 3], [1, 6, 4], [1, 7, 5]], dtype=float)
        sel = FeatureSelector(variance_threshold=0.01)
        Z = sel.fit_transform(X)
        assert Z.shape[1] == 2

    def test_n_selected(self):
        from models.music_pipeline.normalization import FeatureSelector
        rng = np.random.RandomState(42)
        X = np.column_stack([rng.randn(20), np.zeros(20), rng.randn(20)])
        sel = FeatureSelector()
        sel.fit(X)
        assert sel.n_selected == 2

    def test_correlation_prune(self):
        from models.music_pipeline.normalization import FeatureSelector
        rng = np.random.RandomState(42)
        base = rng.randn(50)
        X = np.column_stack([base, base + 0.001 * rng.randn(50), rng.randn(50)])
        result = FeatureSelector.correlation_prune(X, threshold=0.95)
        assert result.shape[1] == 2
