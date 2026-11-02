"""Tests for expansion round 7: rights management, music selection, auditory attention, form evolution."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_industry.rights_management import (
    RightType, LicenseType, OwnershipShare, License, RightsBundle,
    validate_splits, detect_conflicts, RoyaltyWaterfall,
    compute_waterfall, license_value_estimate, territory_overlap,
)
from models.music_therapy.music_selection import (
    TherapeuticIntent, TrackProfile, TherapeuticPlaylist,
    intent_fitness, rank_tracks, build_therapeutic_playlist,
    iso_principle_sequence, session_energy_profile,
)
from models.music_cognition.auditory_attention import (
    SalienceMap, spectral_salience, onset_salience, compute_salience_map,
    AuditoryStream, stream_segregation, attention_decay, surprise_score,
)
from models.music_historical.form_evolution import (
    MusicalForm, FormType, FormAnalysis,
    form_period_distribution, form_complexity_score, form_lineage,
    section_similarity, detect_form, repetition_rate, analyze_form,
    period_from_year,
)
from models.music_historical.periods import MusicalPeriod


class TestRightsManagement:
    def test_ownership_revenue(self):
        s = OwnershipShare(holder="alice", share_pct=50.0, right_type=RightType.COMPOSITION)
        assert abs(s.revenue(1000.0) - 500.0) < 0.01

    def test_rights_bundle_total(self):
        rb = RightsBundle(
            work_id="W001", title="Song A",
            shares=[
                OwnershipShare("alice", 60.0, RightType.COMPOSITION),
                OwnershipShare("bob", 40.0, RightType.COMPOSITION),
                OwnershipShare("alice", 100.0, RightType.MASTER),
            ],
        )
        assert abs(rb.total_share(RightType.COMPOSITION) - 100.0) < 0.01
        assert abs(rb.total_share(RightType.MASTER) - 100.0) < 0.01
        assert rb.is_fully_allocated(RightType.COMPOSITION)
        assert {"alice", "bob"} == rb.holders()

    def test_validate_splits_correct(self):
        shares = [
            OwnershipShare("a", 60.0, RightType.COMPOSITION),
            OwnershipShare("b", 40.0, RightType.COMPOSITION),
        ]
        assert validate_splits(shares) == []

    def test_validate_splits_over(self):
        shares = [
            OwnershipShare("a", 60.0, RightType.COMPOSITION),
            OwnershipShare("b", 50.0, RightType.COMPOSITION),
        ]
        issues = validate_splits(shares)
        assert len(issues) == 1
        assert "over-allocated" in issues[0]

    def test_validate_splits_under(self):
        shares = [
            OwnershipShare("a", 30.0, RightType.MASTER),
        ]
        issues = validate_splits(shares)
        assert len(issues) == 1
        assert "under-allocated" in issues[0]

    def test_detect_conflicts_exclusive(self):
        rb = RightsBundle(
            work_id="W002", title="Song B",
            licenses=[
                License(LicenseType.EXCLUSIVE, "pub1", "label1"),
                License(LicenseType.EXCLUSIVE, "pub2", "label2"),
            ],
        )
        conflicts = detect_conflicts([rb])
        assert any("multiple exclusive" in c for c in conflicts)

    def test_royalty_waterfall(self):
        wf = RoyaltyWaterfall(stages=[("admin", 15.0), ("writer", 50.0), ("publisher", 35.0)])
        dist = wf.distribute(1000.0)
        assert abs(dist["admin"] - 150.0) < 0.01
        total_dist = sum(dist.values())
        assert total_dist < 1000.01

    def test_compute_waterfall(self):
        shares = [
            OwnershipShare("writer", 60.0, RightType.COMPOSITION),
            OwnershipShare("co_writer", 40.0, RightType.COMPOSITION),
        ]
        wf = compute_waterfall(shares, admin_fee_pct=10.0, admin_holder="pub")
        dist = wf.distribute(1000.0)
        assert "pub" in dist
        assert dist["pub"] > 0

    def test_license_value_fee(self):
        lic = License(LicenseType.SYNC, "owner", "licensee", fee=5000.0)
        assert license_value_estimate(lic, monthly_streams=100000) == 5000.0

    def test_license_value_revenue_share(self):
        lic = License(LicenseType.NON_EXCLUSIVE, "owner", "licensee",
                      revenue_share_pct=20.0, duration_months=12)
        val = license_value_estimate(lic, monthly_streams=100000, per_stream_rate=0.004)
        assert val > 0

    def test_territory_overlap(self):
        shares = [
            OwnershipShare("a", 50.0, RightType.COMPOSITION, territory="US"),
            OwnershipShare("b", 50.0, RightType.COMPOSITION, territory="US"),
            OwnershipShare("c", 100.0, RightType.COMPOSITION, territory="EU"),
        ]
        overlap = territory_overlap(shares)
        assert len(overlap["US"]) == 2
        assert len(overlap["EU"]) == 1


class TestMusicSelection:
    def _tracks(self):
        return [
            TrackProfile("calm1", 65, 0.4, 0.2, acousticness=0.8),
            TrackProfile("energy1", 130, 0.8, 0.9),
            TrackProfile("focus1", 100, 0.5, 0.3, instrumentalness=0.9),
            TrackProfile("sleep1", 60, 0.3, 0.1, acousticness=0.9, instrumentalness=0.7),
        ]

    def test_intent_fitness_calming(self):
        t = TrackProfile("t1", 70, 0.4, 0.2, acousticness=0.8)
        score = intent_fitness(t, TherapeuticIntent.CALMING)
        assert score > 0.7

    def test_intent_fitness_energizing(self):
        t = TrackProfile("t2", 130, 0.8, 0.9)
        score = intent_fitness(t, TherapeuticIntent.ENERGIZING)
        assert score > 0.7

    def test_rank_tracks_order(self):
        tracks = self._tracks()
        ranked = rank_tracks(tracks, TherapeuticIntent.CALMING)
        assert ranked[0][0] in ("calm1", "sleep1")

    def test_build_playlist(self):
        tracks = self._tracks()
        pl = build_therapeutic_playlist(tracks, TherapeuticIntent.FOCUS, target_duration_min=10)
        assert isinstance(pl, TherapeuticPlaylist)
        assert pl.intent == TherapeuticIntent.FOCUS
        assert pl.mean_fitness > 0

    def test_iso_principle_sequence(self):
        tracks = [
            TrackProfile(f"t{i}", 100, 0.5, e)
            for i, e in enumerate([0.2, 0.4, 0.5, 0.6, 0.8, 0.9])
        ]
        seq = iso_principle_sequence(tracks, start_energy=0.8, target_energy=0.3, n_steps=4)
        assert len(seq) <= 4
        assert len(set(seq)) == len(seq)

    def test_session_energy_profile(self):
        tracks = self._tracks()
        ids = ["calm1", "energy1"]
        profile = session_energy_profile(tracks, ids)
        assert len(profile) == 2
        assert profile[0] < profile[1]

    def test_intent_fitness_unknown(self):
        t = TrackProfile("t", 100, 0.5, 0.5)
        score = intent_fitness(t, TherapeuticIntent.SOCIAL_BONDING)
        assert score == 0.5


class TestAuditoryAttention:
    def test_spectral_salience_shape(self):
        mag = np.random.default_rng(0).random((50, 128))
        sal = spectral_salience(mag)
        assert len(sal) == 50
        assert sal[0] == 0.0
        assert sal[-1] == 0.0

    def test_spectral_salience_normalized(self):
        mag = np.random.default_rng(1).random((30, 64))
        sal = spectral_salience(mag)
        assert np.max(sal) <= 1.0 + 1e-9

    def test_onset_salience_shape(self):
        onset = np.random.default_rng(2).random(40)
        sal = onset_salience(onset, window=3)
        assert len(sal) == 40
        assert np.all(sal >= 0)

    def test_compute_salience_map_combined(self):
        times = np.linspace(0, 5, 30)
        spec = np.random.default_rng(3).random(30)
        ons = np.random.default_rng(4).random(30)
        sm = compute_salience_map(times, spectral_sal=spec, onset_sal=ons)
        assert isinstance(sm, SalienceMap)
        assert len(sm.salience) == 30
        assert isinstance(sm.mean_salience, float)

    def test_salience_map_peaks(self):
        times = np.arange(10, dtype=float)
        spec = np.array([0, 0.2, 0.8, 0.3, 0.1, 0.2, 0.9, 0.3, 0.1, 0])
        sm = compute_salience_map(times, spectral_sal=spec)
        for pt in sm.peak_times:
            assert 0 <= pt <= 9

    def test_stream_segregation_empty(self):
        result = stream_segregation(np.array([]), np.array([]))
        assert result == []

    def test_stream_segregation_two_streams(self):
        freqs = np.array([440, 445, 880, 885, 440, 880])
        times = np.array([0.0, 0.1, 0.15, 0.25, 0.3, 0.35])
        streams = stream_segregation(freqs, times, freq_threshold=100)
        assert len(streams) >= 2
        for s in streams:
            assert isinstance(s, AuditoryStream)
            assert s.coherence >= 0

    def test_attention_decay_shape(self):
        sal = np.random.default_rng(5).random(20)
        att = attention_decay(sal, decay_rate=0.9)
        assert len(att) == 20

    def test_attention_decay_smoothing(self):
        sal = np.zeros(20)
        sal[5] = 1.0
        att = attention_decay(sal, decay_rate=0.95)
        assert att[5] > att[4]
        assert att[6] > att[7]

    def test_surprise_score_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        s = surprise_score(x, x)
        np.testing.assert_allclose(s, 0.0, atol=1e-9)

    def test_surprise_score_deviation(self):
        expected = np.array([1.0, 1.0, 1.0])
        observed = np.array([2.0, 1.0, 0.5])
        s = surprise_score(expected, observed)
        assert s[0] > 0
        assert abs(s[1]) < 1e-9
        assert s[2] > 0


class TestFormEvolution:
    def test_form_period_distribution(self):
        dist = form_period_distribution()
        assert MusicalForm.FUGUE in dist
        assert MusicalPeriod.BAROQUE in dist[MusicalForm.FUGUE]

    def test_form_complexity_score(self):
        assert form_complexity_score(MusicalForm.PLAINCHANT) < form_complexity_score(MusicalForm.FUGUE)
        assert form_complexity_score(MusicalForm.TWELVE_TONE_ROW) > 0.9

    def test_form_lineage_plainchant(self):
        chain = form_lineage(MusicalForm.PLAINCHANT)
        assert chain == [MusicalForm.PLAINCHANT]

    def test_form_lineage_symphony(self):
        chain = form_lineage(MusicalForm.SYMPHONY)
        assert chain[0] == MusicalForm.PLAINCHANT
        assert chain[-1] == MusicalForm.SYMPHONY
        assert len(chain) >= 3

    def test_section_similarity_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        sim = section_similarity(a, a)
        assert abs(sim - 1.0) < 1e-6

    def test_section_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = section_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_detect_form_ternary(self):
        assert detect_form(["A", "B", "A"]) == FormType.TERNARY

    def test_detect_form_binary(self):
        assert detect_form(["A", "B"]) == FormType.BINARY

    def test_detect_form_strophic(self):
        assert detect_form(["A", "A", "A"]) == FormType.STROPHIC

    def test_detect_form_through_composed(self):
        assert detect_form(["A", "B", "C", "D"]) == FormType.THROUGH_COMPOSED

    def test_detect_form_rondo(self):
        assert detect_form(["A", "B", "A", "C", "A"]) == FormType.RONDO

    def test_repetition_rate_no_repeats(self):
        assert repetition_rate(["A", "B", "C"]) == 0.0

    def test_repetition_rate_all_same(self):
        rate = repetition_rate(["A", "A", "A", "A"])
        assert rate == 0.75

    def test_analyze_form_ternary(self):
        fa = analyze_form(["A", "B", "A"])
        assert isinstance(fa, FormAnalysis)
        assert fa.detected_form == FormType.TERNARY
        assert fa.unique_sections == 2
        assert fa.symmetry_score == 1.0

    def test_period_from_year(self):
        assert period_from_year(1300) == MusicalPeriod.MEDIEVAL
        assert period_from_year(1550) == MusicalPeriod.RENAISSANCE
        assert period_from_year(1700) == MusicalPeriod.BAROQUE
        assert period_from_year(1800) == MusicalPeriod.CLASSICAL
        assert period_from_year(1870) == MusicalPeriod.ROMANTIC
        assert period_from_year(1910) == MusicalPeriod.IMPRESSIONIST
        assert period_from_year(1960) == MusicalPeriod.MODERN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
