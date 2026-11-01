"""Tests for emotion dynamics and sentiment lexicon."""
import numpy as np
import pytest
from models.music_emotion.emotion_dynamics import (
    EmotionTrajectory, track_emotion_over_time, emotion_volatility,
    emotion_arc_type, emotional_surprise, valence_arousal_path_length,
    emotional_centroid,
)
from models.music_emotion.sentiment_lexicon import (
    MUSIC_SENTIMENT_LEXICON, score_text_sentiment, emotional_distance,
    closest_emotion_word, sentiment_polarity, arousal_level,
)


class TestEmotionDynamics:
    def _simple_trajectory(self, n=20):
        return EmotionTrajectory(
            timestamps=np.arange(n) * 5.0,
            valence_curve=np.linspace(-0.5, 0.8, n),
            arousal_curve=np.linspace(0.3, 0.7, n),
            dominance_curve=np.ones(n) * 0.5,
        )

    def test_track_from_features(self):
        feats = [{"valence": 0.5, "arousal": 0.3, "dominance": 0.4}] * 5
        traj = track_emotion_over_time(feats)
        assert len(traj.timestamps) == 5

    def test_track_empty(self):
        traj = track_emotion_over_time([])
        assert len(traj.timestamps) == 0

    def test_volatility_nonneg(self):
        traj = self._simple_trajectory()
        assert emotion_volatility(traj) >= 0

    def test_volatility_zero_for_constant(self):
        traj = EmotionTrajectory(
            np.arange(10) * 1.0,
            np.ones(10) * 0.5, np.ones(10) * 0.5, np.ones(10) * 0.5)
        assert emotion_volatility(traj) < 1e-6

    def test_arc_type_rising(self):
        traj = self._simple_trajectory()
        assert emotion_arc_type(traj) in ("rising", "peak", "flat")

    def test_arc_type_valid_string(self):
        traj = self._simple_trajectory()
        valid = {"rising", "falling", "peak", "valley", "flat", "oscillating"}
        assert emotion_arc_type(traj) in valid

    def test_surprise_nonneg(self):
        traj = self._simple_trajectory()
        s = emotional_surprise(traj)
        assert np.all(s >= 0)

    def test_surprise_length(self):
        traj = self._simple_trajectory(10)
        s = emotional_surprise(traj)
        assert len(s) == 10

    def test_path_length_nonneg(self):
        traj = self._simple_trajectory()
        assert valence_arousal_path_length(traj) >= 0

    def test_path_length_zero_for_constant(self):
        traj = EmotionTrajectory(
            np.arange(5) * 1.0,
            np.ones(5), np.ones(5), np.ones(5))
        assert valence_arousal_path_length(traj) < 1e-6

    def test_centroid_bounds(self):
        traj = self._simple_trajectory()
        v, a, d = emotional_centroid(traj)
        assert -1.0 <= v <= 1.0
        assert 0.0 <= a <= 1.0


class TestSentimentLexicon:
    def test_lexicon_size(self):
        assert len(MUSIC_SENTIMENT_LEXICON) >= 80

    def test_score_known_word(self):
        v, a, d = score_text_sentiment("triumphant")
        assert v > 0.5
        assert a > 0.5

    def test_score_unknown_returns_zero(self):
        v, a, d = score_text_sentiment("xyzzy")
        assert v == 0.0

    def test_melancholy_negative_valence(self):
        v, a, d = score_text_sentiment("melancholy")
        assert v < 0

    def test_emotional_distance_zero(self):
        assert emotional_distance((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) == 0.0

    def test_emotional_distance_symmetric(self):
        d1 = emotional_distance((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
        d2 = emotional_distance((0.4, 0.5, 0.6), (0.1, 0.2, 0.3))
        assert abs(d1 - d2) < 1e-10

    def test_closest_word_returns_string(self):
        w = closest_emotion_word((0.9, 0.9, 0.8))
        assert isinstance(w, str)
        assert w in MUSIC_SENTIMENT_LEXICON

    def test_sentiment_polarity_range(self):
        p = sentiment_polarity("jubilant triumphant")
        assert -1.0 <= p <= 1.0

    def test_arousal_level_range(self):
        a = arousal_level("energetic aggressive")
        assert 0.0 <= a <= 1.0

    def test_arousal_high_for_energetic(self):
        a = arousal_level("energetic")
        assert a > 0.7
