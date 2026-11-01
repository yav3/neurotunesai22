"""Tests for the music_historical module."""

import pytest
from models.music_historical.periods import (
    MusicalPeriod, PeriodProfile, PERIOD_PROFILES,
    classify_period_by_year, period_harmonic_signature, style_distance,
)
from models.music_historical.evolution import (
    StyleFeatureVector, style_evolution_trajectory,
    innovation_score, influence_graph,
)
from models.music_historical.form_evolution import (
    MusicalForm, form_period_distribution, form_complexity_score, form_lineage,
)
from models.music_historical.instrumentation_history import (
    InstrumentEra, INSTRUMENT_TIMELINE,
    available_instruments, orchestra_size_by_period, instrument_family_balance,
)


class TestPeriods:
    def test_all_periods_have_profiles(self):
        for p in MusicalPeriod:
            assert p in PERIOD_PROFILES

    def test_classify_medieval(self):
        assert classify_period_by_year(800) == MusicalPeriod.MEDIEVAL

    def test_classify_baroque(self):
        assert classify_period_by_year(1700) == MusicalPeriod.BAROQUE

    def test_classify_classical(self):
        assert classify_period_by_year(1800) == MusicalPeriod.CLASSICAL

    def test_classify_romantic(self):
        assert classify_period_by_year(1850) == MusicalPeriod.ROMANTIC

    def test_classify_modern(self):
        assert classify_period_by_year(1950) == MusicalPeriod.MODERN

    def test_classify_future(self):
        assert classify_period_by_year(2100) == MusicalPeriod.CONTEMPORARY

    def test_harmonic_signature(self):
        sig = period_harmonic_signature(MusicalPeriod.BAROQUE)
        assert "basso_continuo" in sig
        assert sig["basso_continuo"] == 1.0

    def test_style_distance_identical(self):
        p = PERIOD_PROFILES[MusicalPeriod.CLASSICAL]
        assert style_distance(p, p) == 0.0

    def test_style_distance_different(self):
        a = PERIOD_PROFILES[MusicalPeriod.MEDIEVAL]
        b = PERIOD_PROFILES[MusicalPeriod.MODERN]
        d = style_distance(a, b)
        assert 0.0 < d <= 1.0


class TestEvolution:
    def test_trajectory_length(self):
        periods = [MusicalPeriod.BAROQUE, MusicalPeriod.CLASSICAL, MusicalPeriod.ROMANTIC]
        traj = style_evolution_trajectory(periods)
        assert len(traj) == 3

    def test_trajectory_values(self):
        traj = style_evolution_trajectory([MusicalPeriod.ROMANTIC])
        sfv = traj[0]
        assert sfv.dissonance_level == 0.3
        assert sfv.dynamic_range == 0.9

    def test_innovation_score_zero_at_norm(self):
        sfv = StyleFeatureVector(0.2, 0.5, 8, 4.0, 4, 0.5)
        score = innovation_score(sfv, MusicalPeriod.BAROQUE)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_innovation_score_positive(self):
        sfv = StyleFeatureVector(0.9, 0.9, 12, 2.0, 6, 1.0)
        score = innovation_score(sfv, MusicalPeriod.MEDIEVAL)
        assert score > 0.5

    def test_influence_graph_structure(self):
        composers = {
            "Bach": MusicalPeriod.BAROQUE,
            "Mozart": MusicalPeriod.CLASSICAL,
            "Beethoven": MusicalPeriod.CLASSICAL,
            "Chopin": MusicalPeriod.ROMANTIC,
        }
        graph = influence_graph(composers)
        assert "Bach" in graph
        assert "Mozart" in graph["Bach"]

    def test_influence_graph_no_distant_links(self):
        composers = {
            "Gregorian": MusicalPeriod.MEDIEVAL,
            "Stravinsky": MusicalPeriod.MODERN,
        }
        graph = influence_graph(composers)
        assert "Stravinsky" not in graph["Gregorian"]
        assert "Gregorian" not in graph["Stravinsky"]


class TestFormEvolution:
    def test_distribution_completeness(self):
        dist = form_period_distribution()
        assert len(dist) == len(MusicalForm)

    def test_complexity_fugue_high(self):
        score = form_complexity_score(MusicalForm.FUGUE)
        assert score >= 0.8

    def test_complexity_plainchant_low(self):
        score = form_complexity_score(MusicalForm.PLAINCHANT)
        assert score <= 0.2

    def test_lineage_plainchant_is_root(self):
        chain = form_lineage(MusicalForm.PLAINCHANT)
        assert chain == [MusicalForm.PLAINCHANT]

    def test_lineage_symphony_chain(self):
        chain = form_lineage(MusicalForm.SYMPHONY)
        assert chain[0] == MusicalForm.PLAINCHANT
        assert chain[-1] == MusicalForm.SYMPHONY
        assert MusicalForm.SONATA in chain

    def test_lineage_twelve_tone(self):
        chain = form_lineage(MusicalForm.TWELVE_TONE_ROW)
        assert MusicalForm.FUGUE in chain
        assert MusicalForm.MOTET in chain


class TestInstrumentation:
    def test_timeline_count(self):
        assert len(INSTRUMENT_TIMELINE) == 20

    def test_medieval_instruments(self):
        instruments = available_instruments(MusicalPeriod.MEDIEVAL)
        assert "organ" in instruments
        assert "lute" in instruments
        assert "violin" not in instruments

    def test_classical_includes_earlier(self):
        instruments = available_instruments(MusicalPeriod.CLASSICAL)
        assert "organ" in instruments
        assert "violin" in instruments
        assert "piano" in instruments
        assert "synthesizer" not in instruments

    def test_modern_includes_all(self):
        instruments = available_instruments(MusicalPeriod.MODERN)
        assert "synthesizer" in instruments
        assert len(instruments) == 20

    def test_orchestra_sizes(self):
        assert orchestra_size_by_period(MusicalPeriod.MEDIEVAL) == 5
        assert orchestra_size_by_period(MusicalPeriod.ROMANTIC) == 80

    def test_family_balance(self):
        instruments = ["violin", "viola", "cello", "oboe", "piano"]
        balance = instrument_family_balance(instruments)
        assert balance["string"] == pytest.approx(3 / 5)
        assert balance["woodwind"] == pytest.approx(1 / 5)
        assert balance["keyboard"] == pytest.approx(1 / 5)

    def test_family_balance_unknown(self):
        balance = instrument_family_balance(["theremin"])
        assert "unknown" in balance


class TestPackageExports:
    def test_all_exports(self):
        import models.music_historical as mh
        assert hasattr(mh, "MusicalPeriod")
        assert hasattr(mh, "MusicalForm")
        assert hasattr(mh, "InstrumentEra")
        assert hasattr(mh, "StyleFeatureVector")
        assert hasattr(mh, "form_lineage")
        assert hasattr(mh, "available_instruments")
