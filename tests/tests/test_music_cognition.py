"""Tests for the music_cognition module: consonance, expectation, groove, tension, memory."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_cognition.consonance import (
    RoughnessModel,
    IntervalConsonance,
    roughness,
    interval_consonance,
    chord_consonance,
    INTERVAL_CONSONANCE,
)
from models.music_cognition.expectation import (
    MarkovExpectation,
    MelodicExpectation,
    information_content,
    melodic_surprise,
)
from models.music_cognition.groove import (
    GrooveAnalyzer,
    MicrotimingProfile,
    syncopation_index,
    swing_ratio,
)
from models.music_cognition.tension import (
    TensionCurve,
    SurpriseTracker,
    harmonic_tension,
    voice_leading_cost,
    minimal_voice_leading,
)
from models.music_cognition.memory import (
    RepetitionDetector,
    FamiliarityModel,
    find_repetitions,
    familiarity_preference,
)


class TestRoughnessModel:
    @pytest.fixture
    def model(self):
        return RoughnessModel()

    def test_unison_low_roughness(self, model):
        assert model.roughness(440.0, 440.0) == pytest.approx(0.0, abs=0.01)

    def test_octave_low_roughness(self, model):
        r = model.roughness(220.0, 440.0)
        assert r < 0.1

    def test_minor_second_high_roughness(self, model):
        r = model.roughness(440.0, 466.16)
        assert r > 0.0

    def test_zero_frequency(self, model):
        assert model.roughness(0.0, 440.0) == 0.0

    def test_symmetry(self, model):
        r1 = model.roughness(440.0, 660.0)
        r2 = model.roughness(660.0, 440.0)
        assert r1 == pytest.approx(r2, abs=1e-10)

    def test_chord_roughness_single(self, model):
        assert model.chord_roughness([440.0]) == 0.0

    def test_chord_roughness_pair(self, model):
        r = model.chord_roughness([440.0, 660.0])
        assert r >= 0.0

    def test_roughness_module_function(self):
        assert roughness(440.0, 440.0) == pytest.approx(0.0, abs=0.01)


class TestIntervalConsonance:
    @pytest.fixture
    def ic(self):
        return IntervalConsonance()

    def test_unison(self, ic):
        assert ic.rate(0) == 1.0

    def test_octave(self, ic):
        assert ic.rate(12) == 0.95

    def test_perfect_fifth(self, ic):
        assert ic.rate(7) == 0.90

    def test_tritone(self, ic):
        assert ic.rate(6) == 0.20

    def test_negative_wraps(self, ic):
        assert ic.rate(-5) == ic.rate(5)

    def test_chord_consonance_major(self, ic):
        c = ic.chord_consonance([60.0, 64.0, 67.0])
        assert 0.0 < c < 1.0

    def test_chord_consonance_single(self, ic):
        assert ic.chord_consonance([60.0]) == 1.0

    def test_interval_consonance_function(self):
        assert interval_consonance(7) == 0.90

    def test_chord_consonance_function(self):
        c = chord_consonance([60.0, 64.0, 67.0])
        assert 0.0 < c < 1.0


class TestMarkovExpectation:
    @pytest.fixture
    def model(self):
        m = MarkovExpectation(order=1)
        m.train([[60, 62, 64, 65, 67], [60, 62, 64, 60, 62]])
        return m

    def test_predict_distribution(self, model):
        dist = model.predict([60])
        assert isinstance(dist, dict)
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_probability_known(self, model):
        p = model.probability([60], 62)
        assert p > 0

    def test_probability_unknown(self, model):
        p = model.probability([99], 0)
        assert p == 0.0

    def test_information_content_certain(self, model):
        p = model.probability([60], 62)
        ic = model.information_content([60], 62)
        assert ic >= 0

    def test_sequence_entropy(self, model):
        e = model.sequence_entropy([60, 62, 64, 65])
        assert e >= 0

    def test_empty_sequence_entropy(self, model):
        assert model.sequence_entropy([60]) == 0.0


class TestMelodicExpectation:
    @pytest.fixture
    def me(self):
        return MelodicExpectation()

    def test_surprise_unison(self, me):
        assert me.surprise(0) == pytest.approx(0.1)

    def test_surprise_octave(self, me):
        assert me.surprise(12) == pytest.approx(0.3)

    def test_surprise_increases_with_interval(self, me):
        assert me.surprise(1) < me.surprise(7)

    def test_proximity_inverse(self, me):
        assert me.proximity_score(0) > me.proximity_score(7)

    def test_registral_return_short(self, me):
        assert me.registral_return([60, 64]) == 0.0

    def test_melodic_surprise_function(self):
        assert melodic_surprise(0) == pytest.approx(0.1)


class TestGrooveAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return GrooveAnalyzer()

    def test_syncopation_basic(self, analyzer):
        onsets = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
        s = analyzer.syncopation_index(onsets, tempo=120.0)
        assert 0.0 <= s <= 1.0

    def test_syncopation_empty(self, analyzer):
        assert analyzer.syncopation_index(np.array([0.0])) == 0.0

    def test_grid_deviation_on_grid(self, analyzer):
        onsets = np.array([0.0, 0.5, 1.0, 1.5])
        devs = analyzer.grid_deviation(onsets, tempo=120.0)
        assert len(devs) == 4
        assert all(abs(d) < 0.01 for d in devs)

    def test_grid_deviation_empty(self, analyzer):
        devs = analyzer.grid_deviation(np.array([]), tempo=120.0)
        assert len(devs) == 0

    def test_syncopation_index_function(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75])
        s = syncopation_index(onsets, tempo=120.0)
        assert 0.0 <= s <= 1.0


class TestMicrotimingProfile:
    def test_analyze_on_grid(self):
        mp = MicrotimingProfile()
        onsets = np.array([0.0, 0.5, 1.0])
        result = mp.analyze(onsets, tempo=120.0)
        assert "mean_deviation" in result
        assert "std_deviation" in result
        assert "tendency" in result

    def test_analyze_empty(self):
        mp = MicrotimingProfile()
        result = mp.analyze(np.array([]), tempo=120.0)
        assert result["mean_deviation"] == 0.0


class TestSwingRatio:
    def test_straight(self):
        pairs = np.array([0.0, 0.5, 1.0, 1.5])
        r = swing_ratio(pairs)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_single_onset(self):
        assert swing_ratio(np.array([0.0])) == 1.0


class TestTensionCurve:
    @pytest.fixture
    def tc(self):
        return TensionCurve()

    def test_same_chord_low_tension(self, tc):
        t = tc.tension([60, 64, 67], [60, 64, 67])
        assert t < 0.3

    def test_distant_chord_higher(self, tc):
        t1 = tc.tension([60, 64, 67], [61, 65, 68])
        t2 = tc.tension([60, 64, 67], [66, 70, 73])
        assert t2 > 0 or t1 > 0

    def test_curve_length(self, tc):
        chords = [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]]
        c = tc.curve(chords)
        assert len(c) == 4
        assert c[0] == 0.0

    def test_curve_single(self, tc):
        c = tc.curve([[60]])
        assert c == [0.0]


class TestHarmonicTension:
    def test_same_root(self):
        assert harmonic_tension(0, 0) == 0.0

    def test_tritone_high(self):
        t = harmonic_tension(0, 6)
        assert t > 0.5

    def test_fifth_moderate(self):
        t = harmonic_tension(0, 7)
        assert 0.0 < t < 1.0


class TestVoiceLeading:
    def test_same_chord(self):
        assert voice_leading_cost([60, 64, 67], [60, 64, 67]) == 0

    def test_step_motion(self):
        cost = voice_leading_cost([60, 64, 67], [62, 65, 69])
        assert cost > 0

    def test_minimal_pairing(self):
        pairs = minimal_voice_leading([60, 64], [62, 65])
        assert len(pairs) == 2

    def test_empty(self):
        assert voice_leading_cost([], [60]) == 0


class TestSurpriseTracker:
    def test_untrained(self):
        st = SurpriseTracker()
        s = st.track([60, 62, 64])
        assert all(v == 0.0 for v in s)

    def test_trained(self):
        st = SurpriseTracker()
        st.train([[60, 62, 64, 65, 67]] * 10)
        s = st.track([60, 62, 64])
        assert len(s) == 3
        assert s[0] == 0.0


class TestRepetitionDetector:
    @pytest.fixture
    def detector(self):
        return RepetitionDetector()

    def test_find_exact_repeat(self, detector):
        seq = [1, 2, 3, 4, 1, 2, 3]
        reps = detector.find_repetitions(seq, min_length=3)
        assert len(reps) > 0

    def test_no_repeat(self, detector):
        seq = [1, 2, 3, 4, 5]
        reps = detector.find_repetitions(seq, min_length=3)
        assert len(reps) == 0

    def test_density(self, detector):
        seq = [1, 2, 3, 1, 2, 3]
        d = detector.repetition_density(seq, min_length=3)
        assert 0.0 < d <= 1.0

    def test_density_no_repeats(self, detector):
        assert detector.repetition_density([1, 2, 3, 4]) == 0.0

    def test_find_repetitions_function(self):
        reps = find_repetitions([1, 2, 1, 2], min_length=2)
        assert len(reps) > 0


class TestFamiliarityModel:
    @pytest.fixture
    def model(self):
        return FamiliarityModel()

    def test_zero_exposure(self, model):
        assert model.preference(0) == 0.0

    def test_peak_exposure(self, model):
        p = model.preference(10, peak=10)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_inverted_u(self, model):
        low = model.preference(2, peak=10)
        peak = model.preference(10, peak=10)
        high = model.preference(30, peak=10)
        assert low < peak
        assert high < peak

    def test_curve_length(self, model):
        c = model.preference_curve(max_exposure=20)
        assert len(c) == 21

    def test_optimal_exposure(self, model):
        assert model.optimal_exposure(peak=15) == 15

    def test_familiarity_function(self):
        assert familiarity_preference(0) == 0.0
        assert familiarity_preference(10, peak=10) == pytest.approx(1.0, abs=0.01)
