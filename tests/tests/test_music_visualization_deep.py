"""Deep tests for chromagram computation and Tonnetz projection."""
import numpy as np
import pytest

from models.music_visualization.chromagram_display import (
    compute_chromagram, hpcp, chroma_to_tonnetz,
    chroma_correlation, key_profile_correlation,
    Chromagram, TonnetzCoords, PITCH_CLASSES,
)

SR = 16000


def _sine(freq=440.0, dur=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)


class TestComputeChromagram:
    def test_shape(self):
        sig = _sine(dur=1.0)
        cg = compute_chromagram(sig, SR)
        assert isinstance(cg, Chromagram)
        assert cg.chroma.shape[0] == 12
        assert cg.chroma.shape[1] > 0

    def test_times(self):
        cg = compute_chromagram(_sine(dur=1.0), SR, hop=2048)
        assert len(cg.times) == cg.chroma.shape[1]
        assert cg.times[0] == pytest.approx(0.0)

    def test_nonneg(self):
        cg = compute_chromagram(_sine(), SR)
        assert np.all(cg.chroma >= 0)

    def test_norm_l1(self):
        cg = compute_chromagram(_sine(), SR, norm="l1")
        col_sums = cg.chroma.sum(axis=0)
        for s in col_sums:
            if s > 1e-10:
                assert s == pytest.approx(1.0, abs=0.01)

    def test_norm_max(self):
        cg = compute_chromagram(_sine(), SR, norm="max")
        col_max = cg.chroma.max(axis=0)
        for m in col_max:
            if m > 1e-10:
                assert m == pytest.approx(1.0, abs=0.01)


class TestHPCP:
    def test_shape(self):
        p = hpcp(_sine(dur=1.0), SR)
        assert p.shape[0] == 12
        assert p.shape[1] > 0

    def test_nonneg(self):
        p = hpcp(_sine(), SR)
        assert np.all(p >= 0)

    def test_max_one(self):
        p = hpcp(_sine(), SR)
        for t in range(p.shape[1]):
            if p[:, t].max() > 0:
                assert p[:, t].max() == pytest.approx(1.0, abs=0.01)


class TestTonnetz:
    def test_shape(self):
        chroma = np.random.default_rng(0).random((12, 10))
        tn = chroma_to_tonnetz(chroma)
        assert isinstance(tn, TonnetzCoords)
        assert len(tn.fifths) == 10
        assert len(tn.minor_thirds) == 10
        assert len(tn.major_thirds) == 10

    def test_wrong_bins(self):
        with pytest.raises(ValueError):
            chroma_to_tonnetz(np.zeros((7, 5)))

    def test_silence_zero(self):
        chroma = np.zeros((12, 3))
        tn = chroma_to_tonnetz(chroma)
        assert np.all(tn.fifths == 0)


class TestChromaCorrelation:
    def test_diagonal_one(self):
        rng = np.random.default_rng(42)
        chroma = rng.random((12, 50))
        corr = chroma_correlation(chroma)
        for i in range(12):
            assert corr[i, i] == pytest.approx(1.0, abs=0.01)

    def test_symmetric(self):
        rng = np.random.default_rng(0)
        corr = chroma_correlation(rng.random((12, 20)))
        assert np.allclose(corr, corr.T, atol=1e-10)


class TestKeyProfile:
    def test_returns_24_keys(self):
        chroma = np.random.default_rng(0).random(12)
        results = key_profile_correlation(chroma)
        assert len(results) == 24

    def test_sorted_descending(self):
        chroma = np.random.default_rng(1).random(12)
        results = key_profile_correlation(chroma)
        for i in range(len(results) - 1):
            assert results[i][2] >= results[i + 1][2]

    def test_c_major_dominant(self):
        c_major = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        results = key_profile_correlation(c_major)
        assert results[0][0] == "C"
        assert results[0][1] == "major"
