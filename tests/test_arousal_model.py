"""Tests for arousal-valence model."""
import numpy as np
import pytest
from models.music_emotion.arousal_model import (
    AVPoint, AVTrajectory, ArousalValenceModel,
    spectral_features, rhythm_features, extract_features,
    quadrant_distribution, emotional_arc_similarity,
)

SR = 44100

def _tone(freq=440, dur=1.0, amp=0.5):
    t = np.arange(int(SR * dur)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


class TestAVTrajectory:
    def test_empty(self):
        t = AVTrajectory()
        assert t.mean_valence() == 0.0
        assert t.emotional_range() == 0.0

    def test_single_point(self):
        t = AVTrajectory(points=[AVPoint(0.5, 0.8)])
        assert t.mean_valence() == 0.5
        assert t.mean_arousal() == 0.8

    def test_emotional_range(self):
        t = AVTrajectory(points=[AVPoint(-1, -1), AVPoint(1, 1)])
        assert t.emotional_range() > 2.0


class TestFeatures:
    def test_spectral(self):
        s = _tone(1000, 0.5)
        feats = spectral_features(s, SR)
        assert feats.shape[1] == 4
        assert feats[0, 0] > 500  # centroid near 1kHz

    def test_rhythm(self):
        s = _tone(440, 1.0)
        r = rhythm_features(s, SR)
        assert len(r) == 3

    def test_extract(self):
        s = _tone(440, 1.0)
        f = extract_features(s, SR)
        assert len(f) >= 11


class TestModel:
    def test_fit_predict(self):
        n = 20
        X = np.random.randn(n, 11)
        v = np.random.uniform(-1, 1, n)
        a = np.random.uniform(-1, 1, n)
        model = ArousalValenceModel()
        model.fit(X, v, a)
        vp, ap = model.predict(X[0])
        assert -1 <= vp <= 1
        assert -1 <= ap <= 1

    def test_unfit_returns_zero(self):
        model = ArousalValenceModel()
        v, a = model.predict(np.zeros(11))
        assert v == 0.0
        assert a == 0.0


class TestQuadrant:
    def test_distribution(self):
        pts = [AVPoint(0.5, 0.5), AVPoint(-0.5, 0.5),
               AVPoint(-0.5, -0.5), AVPoint(0.5, -0.5)]
        d = quadrant_distribution(pts)
        assert abs(d["happy"] - 0.25) < 1e-6
        assert abs(d["angry"] - 0.25) < 1e-6


class TestSimilarity:
    def test_identical(self):
        t = AVTrajectory(points=[AVPoint(0, 0), AVPoint(1, 1)])
        assert emotional_arc_similarity(t, t) > 0.9

    def test_opposite(self):
        t1 = AVTrajectory(points=[AVPoint(-1, -1), AVPoint(1, 1)])
        t2 = AVTrajectory(points=[AVPoint(1, 1), AVPoint(-1, -1)])
        sim = emotional_arc_similarity(t1, t2)
        assert sim < 0.5
