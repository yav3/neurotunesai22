"""Tests for ensemble dynamics analysis."""
import numpy as np
import pytest

from models.music_collaboration.ensemble_dynamics import (
    compute_onset_synchrony, compute_tempo_agreement, entrainment_index,
    turn_taking_analysis, blend_score, analyze_ensemble, EnsembleSynchrony,
)


class TestOnsetSynchrony:
    def test_perfect_sync(self):
        onsets = [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])]
        assert compute_onset_synchrony(onsets) == pytest.approx(1.0)

    def test_poor_sync(self):
        onsets = [np.array([0.0, 0.5, 1.0]), np.array([0.25, 0.75, 1.25])]
        s = compute_onset_synchrony(onsets, tolerance_sec=0.3)
        assert 0.0 <= s <= 1.0

    def test_single_part(self):
        assert compute_onset_synchrony([np.array([0.0, 1.0])]) == 1.0

    def test_empty(self):
        assert compute_onset_synchrony([np.array([]), np.array([])]) == 1.0


class TestTempoAgreement:
    def test_identical_tempi(self):
        tempi = [np.array([120.0, 120.0]), np.array([120.0, 120.0])]
        assert compute_tempo_agreement(tempi) == pytest.approx(1.0)

    def test_different_tempi(self):
        tempi = [np.array([120.0]), np.array([60.0])]
        a = compute_tempo_agreement(tempi)
        assert 0.0 <= a < 1.0

    def test_single_part(self):
        assert compute_tempo_agreement([np.array([100.0])]) == 1.0


class TestEntrainmentIndex:
    def test_correlated_ioi(self):
        ioi = np.array([0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6])
        e = entrainment_index(ioi, ioi)
        assert e == pytest.approx(1.0, abs=0.1)

    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        e = entrainment_index(a, b)
        assert e < 0.5

    def test_short(self):
        assert entrainment_index(np.array([1.0]), np.array([1.0])) == 0.0


class TestTurnTaking:
    def test_no_overlap(self):
        o1 = np.array([0.0, 0.1, 0.2])
        o2 = np.array([0.5, 0.6, 0.7])
        result = turn_taking_analysis([o1, o2], ["A", "B"])
        assert result["overlap_ratio"] == pytest.approx(0.0, abs=0.05)

    def test_full_overlap(self):
        o = np.array([0.0, 0.1, 0.2])
        result = turn_taking_analysis([o, o], ["A", "B"])
        assert result["overlap_ratio"] > 0.5

    def test_empty(self):
        result = turn_taking_analysis([], [])
        assert result["overlap_ratio"] == 0.0


class TestBlendScore:
    def test_identical_spectra(self):
        s = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        assert blend_score([s, s]) == pytest.approx(1.0)

    def test_disjoint_spectra(self):
        s1 = np.array([1.0, 0.0, 0.0, 0.0])
        s2 = np.array([0.0, 0.0, 0.0, 1.0])
        assert blend_score([s1, s2]) == pytest.approx(0.0)

    def test_single(self):
        assert blend_score([np.array([1.0])]) == 1.0


class TestAnalyzeEnsemble:
    def test_basic(self):
        onsets = [np.array([0.0, 0.5]), np.array([0.0, 0.5])]
        tempi = [np.array([120.0]), np.array([120.0])]
        result = analyze_ensemble(onsets, tempi)
        assert isinstance(result, EnsembleSynchrony)
        assert result.n_parts == 2
        assert result.onset_synchrony == pytest.approx(1.0)
