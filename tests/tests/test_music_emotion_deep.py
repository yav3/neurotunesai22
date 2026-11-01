"""Tests for dimensional emotion models and mood classifier."""
import numpy as np
import pytest

from models.music_emotion.dimensional_models import (
    DimensionalEmotionModel,
    MusicMoodClassifier,
    PLUTCHIK_EMOTIONS,
    MOOD_LABELS,
)


class TestDimensionalEmotionModel:
    @pytest.fixture
    def dem(self):
        return DimensionalEmotionModel()

    def test_vad_range(self, dem):
        feats = {"tempo": 140, "mode": 1, "spectral_centroid": 3000, "loudness": -5}
        v, a, d = dem.vad_from_features(feats)
        assert -1 <= v <= 1
        assert -1 <= a <= 1
        assert -1 <= d <= 1

    def test_vad_major_vs_minor(self, dem):
        major = {"tempo": 120, "mode": 1, "spectral_centroid": 2000, "loudness": -15}
        minor = {"tempo": 120, "mode": 0, "spectral_centroid": 2000, "loudness": -15}
        v_maj, _, _ = dem.vad_from_features(major)
        v_min, _, _ = dem.vad_from_features(minor)
        assert v_maj > v_min

    def test_circumplex_q1(self, dem):
        assert dem.circumplex_quadrant(0.5, 0.5) == "happy/excited"

    def test_circumplex_q2(self, dem):
        assert dem.circumplex_quadrant(-0.5, 0.5) == "angry/tense"

    def test_circumplex_q3(self, dem):
        assert dem.circumplex_quadrant(-0.5, -0.5) == "sad/depressed"

    def test_circumplex_q4(self, dem):
        assert dem.circumplex_quadrant(0.5, -0.5) == "calm/relaxed"

    def test_emotion_trajectory_shape(self, dem):
        seq = np.random.default_rng(0).uniform(-1, 1, (20, 3))
        result = dem.emotion_trajectory(seq)
        assert result["smoothed"].shape == seq.shape
        assert len(result["arcs"]) <= 3

    def test_arousal_energy_correlation_perfect(self, dem):
        a = np.linspace(0, 1, 50)
        e = a * 2
        r = dem.arousal_energy_correlation(a, e)
        assert abs(r - 1.0) < 0.01

    def test_arousal_energy_correlation_short(self, dem):
        assert dem.arousal_energy_correlation(np.array([1]), np.array([2])) == 0.0

    def test_valence_mode_major_positive(self, dem):
        v = dem.valence_mode_mapping("C", "major")
        assert v > 0

    def test_valence_mode_minor_negative(self, dem):
        v = dem.valence_mode_mapping("A", "minor")
        assert v < 0

    def test_dominance_from_dynamics_range(self, dem):
        d = dem.dominance_from_dynamics(30, 150, 3000)
        assert -1 <= d <= 1

    def test_emotion_distance_symmetry(self, dem):
        a = (0.5, 0.3, 0.1)
        b = (-0.2, 0.8, -0.4)
        assert abs(dem.emotion_distance(a, b) - dem.emotion_distance(b, a)) < 1e-10

    def test_emotion_distance_identical_zero(self, dem):
        a = (0.5, 0.3, 0.1)
        assert dem.emotion_distance(a, a) < 1e-10

    def test_plutchik_returns_valid_emotion(self, dem):
        emotion, intensity = dem.plutchik_from_vad(0.8, 0.6, 0.5)
        assert emotion in PLUTCHIK_EMOTIONS
        assert intensity in ("mild", "moderate", "intense")

    def test_plutchik_joy(self, dem):
        emotion, _ = dem.plutchik_from_vad(0.9, 0.7, 0.7)
        assert emotion == "joy"

    def test_plutchik_sadness(self, dem):
        emotion, _ = dem.plutchik_from_vad(-0.8, -0.5, -0.4)
        assert emotion == "sadness"


class TestMusicMoodClassifier:
    @pytest.fixture
    def mmc(self):
        return MusicMoodClassifier()

    def test_mood_vocabulary_length(self, mmc):
        assert len(mmc.mood_vocabulary()) == 8

    def test_feature_to_mood_scores_shape(self, mmc):
        features = np.random.default_rng(0).standard_normal(20)
        scores = mmc.feature_to_mood_scores(features)
        assert len(scores) == 8

    def test_feature_to_mood_scores_range(self, mmc):
        features = np.random.default_rng(0).standard_normal(20)
        scores = mmc.feature_to_mood_scores(features)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_mood_from_tempo_mode_happy(self, mmc):
        assert mmc.mood_from_tempo_mode(140, True, 0.7) == "happy"

    def test_mood_from_tempo_mode_calm(self, mmc):
        assert mmc.mood_from_tempo_mode(60, True, 0.2) == "calm"

    def test_mood_transition_sums_to_one(self, mmc):
        probs = mmc.mood_transition_probability("happy", 0.5)
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_mood_consistency_constant(self, mmc):
        scores = np.tile([1, 0, 0, 0, 0, 0, 0, 0], (10, 1)).astype(float)
        c = mmc.mood_consistency(scores)
        assert c > 0.5

    def test_mix_moods_same(self, mmc):
        assert mmc.mix_moods("happy", 0.8, "happy", 0.6) == "happy"

    def test_mix_moods_different(self, mmc):
        result = mmc.mix_moods("happy", 0.8, "sad", 0.3)
        assert "happy" in result and "sad" in result

    def test_mood_to_color_valid_rgb(self, mmc):
        r, g, b = mmc.mood_to_color("happy")
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255

    def test_mood_to_color_unknown(self, mmc):
        r, g, b = mmc.mood_to_color("nonexistent")
        assert r == 128 and g == 128 and b == 128
