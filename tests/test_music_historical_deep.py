"""Deep tests for harmonic language evolution across musical eras."""
import numpy as np
import pytest

from models.music_historical.harmonic_language_evolution import (
    ROMAN_NUMERALS, ERA_TRANSITION_MATRICES,
    harmonic_entropy, vocabulary_usage, dominant_resolution_rate,
    analyze_progression, profile_era, era_entropy_trajectory,
    generate_progression, transition_divergence,
    HarmonicProfile, ProgressionAnalysis,
)
from models.music_historical.periods import MusicalPeriod


class TestHarmonicEntropy:
    def test_nonnegative(self):
        for T in ERA_TRANSITION_MATRICES.values():
            assert harmonic_entropy(T) >= 0.0

    def test_uniform_max(self):
        n = len(ROMAN_NUMERALS)
        T_uniform = np.ones((n, n)) / n
        H = harmonic_entropy(T_uniform)
        assert H == pytest.approx(np.log2(n), abs=0.01)

    def test_deterministic_zero(self):
        n = len(ROMAN_NUMERALS)
        T = np.zeros((n, n))
        T[:, 0] = 1.0
        H = harmonic_entropy(T)
        assert H == pytest.approx(0.0, abs=0.01)


class TestVocabularyUsage:
    def test_classical_uses_all(self):
        v = vocabulary_usage(ERA_TRANSITION_MATRICES[MusicalPeriod.CLASSICAL])
        assert v >= 5

    def test_threshold_effect(self):
        T = ERA_TRANSITION_MATRICES[MusicalPeriod.CLASSICAL]
        v_low = vocabulary_usage(T, threshold=0.001)
        v_high = vocabulary_usage(T, threshold=0.5)
        assert v_low >= v_high


class TestDominantResolution:
    def test_classical_strong(self):
        r = dominant_resolution_rate(ERA_TRANSITION_MATRICES[MusicalPeriod.CLASSICAL])
        assert r > 0.3

    def test_modern_weaker(self):
        r_cl = dominant_resolution_rate(ERA_TRANSITION_MATRICES[MusicalPeriod.CLASSICAL])
        r_mod = dominant_resolution_rate(ERA_TRANSITION_MATRICES[MusicalPeriod.MODERN])
        assert r_cl >= r_mod


class TestAnalyzeProgression:
    def test_basic(self):
        result = analyze_progression(["I", "IV", "V", "I"])
        assert isinstance(result, ProgressionAnalysis)
        assert result.log_likelihood < 0
        assert result.perplexity >= 1.0
        assert len(result.surprise_curve) == 3

    def test_single_chord(self):
        result = analyze_progression(["I"])
        assert result.perplexity == 1.0

    def test_unknown_chords_filtered(self):
        result = analyze_progression(["I", "XXX", "V", "I"])
        assert len(result.chords) == 3

    def test_surprise_location(self):
        result = analyze_progression(["I", "IV", "V", "I", "vii°", "iii"])
        assert result.most_surprising_idx >= 0


class TestProfileEra:
    def test_returns_profile(self):
        p = profile_era(MusicalPeriod.CLASSICAL)
        assert isinstance(p, HarmonicProfile)
        assert p.period == MusicalPeriod.CLASSICAL
        assert p.vocabulary_size > 0

    def test_entropy_ordering(self):
        cl = profile_era(MusicalPeriod.CLASSICAL).entropy
        mod = profile_era(MusicalPeriod.MODERN).entropy
        assert mod > cl


class TestEraTrajectory:
    def test_default_periods(self):
        traj = era_entropy_trajectory()
        assert len(traj) == 4
        assert all(isinstance(h, float) for _, h in traj)

    def test_custom_periods(self):
        traj = era_entropy_trajectory([MusicalPeriod.BAROQUE, MusicalPeriod.ROMANTIC])
        assert len(traj) == 2


class TestGenerateProgression:
    def test_length(self):
        prog = generate_progression(length=12, seed=0)
        assert len(prog) == 12

    def test_valid_chords(self):
        prog = generate_progression(seed=1)
        assert all(c in ROMAN_NUMERALS for c in prog)

    def test_deterministic(self):
        a = generate_progression(seed=42)
        b = generate_progression(seed=42)
        assert a == b


class TestTransitionDivergence:
    def test_self_zero(self):
        d = transition_divergence(MusicalPeriod.CLASSICAL, MusicalPeriod.CLASSICAL)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_positive(self):
        d = transition_divergence(MusicalPeriod.CLASSICAL, MusicalPeriod.MODERN)
        assert d > 0
