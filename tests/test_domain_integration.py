"""Tests for models.foundation.domain_integration — the unified domain analysis layer."""
import pytest
import numpy as np

from models.foundation.domain_integration import (
    CognitionAnalysis, EmotionAnalysis, TheoryAnalysis,
    AcousticsAnalysis, StructureAnalysis, DomainAnalysis,
    DomainIntegrator,
    cognition_to_array, emotion_to_array, theory_to_array,
    acoustics_to_array, structure_to_array,
    DOMAIN_FEATURE_DIM,
)


class TestDataclasses:
    def test_cognition_defaults(self):
        c = CognitionAnalysis()
        assert c.consonance_mean == 0.0
        assert c.familiarity == 0.0

    def test_emotion_defaults(self):
        e = EmotionAnalysis()
        assert e.valence == 0.5
        assert e.primary_emotion == "neutral"

    def test_theory_defaults(self):
        t = TheoryAnalysis()
        assert t.key == "C"
        assert t.time_signature == (4, 4)

    def test_acoustics_defaults(self):
        a = AcousticsAnalysis()
        assert a.crest_factor == 1.0

    def test_structure_defaults(self):
        s = StructureAnalysis()
        assert s.form_name == "unknown"


class TestArrayConversions:
    def test_cognition_to_array_shape(self):
        c = CognitionAnalysis(consonance_mean=0.8, tension_max=0.6)
        arr = cognition_to_array(c)
        assert arr.shape == (10,)
        assert arr.dtype == np.float32
        assert arr[0] == pytest.approx(0.8)

    def test_emotion_to_array_shape(self):
        e = EmotionAnalysis(valence=0.7, arousal=0.3, dominance=0.5)
        arr = emotion_to_array(e)
        assert arr.shape == (3,)
        assert arr[0] == pytest.approx(0.7)

    def test_theory_to_array_shape(self):
        t = TheoryAnalysis(key_confidence=0.9, tempo_bpm=140.0)
        arr = theory_to_array(t)
        assert arr.shape == (5,)
        assert arr[0] == pytest.approx(0.9)
        assert arr[1] == pytest.approx(140.0 / 200.0)

    def test_acoustics_to_array_shape(self):
        a = AcousticsAnalysis(spectral_centroid=2000.0)
        arr = acoustics_to_array(a)
        assert arr.shape == (8,)
        assert arr[0] == pytest.approx(2000.0 / 8000.0)

    def test_structure_to_array_shape(self):
        s = StructureAnalysis(n_segments=5, form_complexity=0.7)
        arr = structure_to_array(s)
        assert arr.shape == (5,)
        assert arr[0] == pytest.approx(0.5)
        assert arr[1] == pytest.approx(0.7)


class TestDomainAnalysis:
    def test_to_flat_dict_numeric_only(self):
        da = DomainAnalysis()
        d = da.to_flat_dict()
        assert isinstance(d, dict)
        assert all(isinstance(v, float) for v in d.values())
        assert len(d) > 0

    def test_to_flat_dict_contains_expected_keys(self):
        da = DomainAnalysis()
        d = da.to_flat_dict()
        assert "cog_consonance_mean" in d
        assert "emo_valence" in d
        assert "theory_key_confidence" in d
        assert "acou_spectral_centroid" in d
        assert "struct_n_segments" in d

    def test_to_flat_dict_skips_strings(self):
        da = DomainAnalysis()
        d = da.to_flat_dict()
        for k in d:
            assert "primary_emotion" not in k
            assert "form_name" not in k
            assert "key" not in k or "key_confidence" in k

    def test_to_numpy(self):
        da = DomainAnalysis()
        arr = da.to_numpy()
        assert arr.dtype == np.float32
        assert arr.ndim == 1

    def test_to_flat_dict_tuple_expansion(self):
        da = DomainAnalysis(theory=TheoryAnalysis(time_signature=(3, 4)))
        d = da.to_flat_dict()
        assert "theory_time_signature_0" in d
        assert d["theory_time_signature_0"] == 3.0
        assert d["theory_time_signature_1"] == 4.0


class TestDomainFeatureDim:
    def test_constant(self):
        assert DOMAIN_FEATURE_DIM == 191

    def test_array_sum_matches(self):
        total = (
            cognition_to_array(CognitionAnalysis()).shape[0]
            + emotion_to_array(EmotionAnalysis()).shape[0]
            + theory_to_array(TheoryAnalysis()).shape[0]
            + acoustics_to_array(AcousticsAnalysis()).shape[0]
            + structure_to_array(StructureAnalysis()).shape[0]
        )
        assert total == 31  # domain-specific dims (191 - 160 base)


class TestDomainIntegrator:
    @pytest.fixture
    def integrator(self):
        return DomainIntegrator(sr=22050)

    @pytest.fixture
    def tone_audio(self):
        sr = 22050
        t = np.linspace(0, 2, 2 * sr, endpoint=False)
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr

    def test_init(self, integrator):
        assert integrator.sr == 22050
        assert integrator.feature_extractor is not None

    def test_analyze_returns_domain_analysis(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        assert isinstance(result, DomainAnalysis)

    def test_analyze_feature_vector_not_none(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        assert result.feature_vector is not None
        assert result.feature_vector.ndim == 1

    def test_analyze_metadata(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        assert result.metadata["sr"] == sr
        assert result.metadata["n_samples"] == len(audio)

    def test_analyze_cognition_populated(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        c = result.cognition
        assert isinstance(c.consonance_mean, float)
        assert isinstance(c.groove_density, float)

    def test_analyze_emotion_ranges(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        e = result.emotion
        assert 0.0 <= e.valence <= 1.0
        assert 0.0 <= e.arousal <= 1.0
        assert 0.0 <= e.dominance <= 1.0
        assert isinstance(e.primary_emotion, str)

    def test_analyze_theory_key(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        assert result.theory.key_confidence >= 0.0
        assert result.theory.tempo_bpm > 0

    def test_analyze_acoustics_populated(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        a = result.acoustics
        assert a.spectral_centroid >= 0.0
        assert a.dynamic_range_db >= 0.0

    def test_analyze_structure_populated(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        s = result.structure
        assert s.n_segments >= 1

    def test_stereo_input_handled(self, integrator):
        sr = 22050
        stereo = np.random.randn(2 * sr, 2).astype(np.float32) * 0.1
        result = integrator.analyze(stereo, sr)
        assert isinstance(result, DomainAnalysis)

    def test_analyze_base_features_present(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        assert result.base_features is not None

    def test_to_numpy_roundtrip(self, integrator, tone_audio):
        audio, sr = tone_audio
        result = integrator.analyze(audio, sr)
        arr = result.to_numpy()
        assert arr.ndim == 1
        assert not np.any(np.isnan(arr))
