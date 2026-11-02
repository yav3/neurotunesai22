"""Tests for expansion round 10: voice arrangement, sync licensing, phrase generation, mood ontology."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_arrangement.voice_arrangement import (
    VoiceArrangement, voicing_spread, voicing_density,
    detect_doubling, arrangement_register, voice_crossing_count,
    spacing_quality, optimal_doubling, register_balance, open_close_voicing,
)
from models.music_industry.sync_licensing import (
    MediaType, UsageType, SyncPlacement, CueSheet,
    estimate_sync_fee, build_cue_sheet, placement_value_score,
    territory_multiplier, catalog_sync_potential,
)
from models.music_improvisation.phrase_generation import (
    PhraseContour, classify_contour, generate_tension_arc,
    rhythmic_variation, pitch_displacement, phrase_density,
    intervallic_variety, build_phrase,
)
from models.music_therapy.mood_ontology import (
    MoodCategory, MoodState, MoodTransition, MOOD_VA_COORDINATES,
    mood_distance, classify_mood, mood_trajectory, plan_mood_transition,
    mood_music_features, mood_consistency,
)


class TestVoiceArrangement:
    def test_voicing_spread(self):
        assert voicing_spread([60, 64, 67]) == 7
        assert voicing_spread([60]) == 0

    def test_voicing_density(self):
        d = voicing_density([60, 64, 67])
        assert d > 0

    def test_detect_doubling(self):
        voices = [[60, 64], [64, 67]]
        result = detect_doubling(voices)
        assert len(result) == 1
        assert result[0] == (0, 1, 64)

    def test_arrangement_register(self):
        assert arrangement_register([40, 42, 44]) == "bass"
        assert arrangement_register([72, 76, 79]) == "soprano"
        assert arrangement_register([60, 64, 67]) == "alto"

    def test_voice_crossing_count(self):
        voices = [[72, 60], [60, 72]]
        crossings = voice_crossing_count(voices)
        assert crossings == 1

    def test_spacing_quality_even(self):
        q = spacing_quality([60, 64, 68, 72])
        assert q > 0.9

    def test_spacing_quality_uneven(self):
        q = spacing_quality([60, 61, 72])
        assert q < 0.7

    def test_optimal_doubling(self):
        result = optimal_doubling([60, 64, 67], 5)
        assert len(result) == 5
        assert result.count(60) >= 2

    def test_register_balance(self):
        voices = [[40, 42], [60, 64], [72, 76]]
        bal = register_balance(voices)
        assert "bass" in bal
        assert "soprano" in bal

    def test_open_close_voicing(self):
        assert open_close_voicing([60, 64, 67]) == "close"
        assert open_close_voicing([48, 67, 84]) == "open"

    def test_voice_arrangement_dataclass(self):
        va = VoiceArrangement(voices=[[60, 64], [67, 72]], n_voices=2)
        assert len(va.all_pitches) == 4


class TestSyncLicensing:
    def test_estimate_sync_fee_film_featured(self):
        fee = estimate_sync_fee(MediaType.FILM, UsageType.FEATURED, 60.0)
        assert fee > 10000

    def test_estimate_sync_fee_podcast(self):
        fee = estimate_sync_fee(MediaType.PODCAST, UsageType.BACKGROUND, 30.0)
        assert fee < 1000

    def test_territory_multiplier_worldwide(self):
        assert territory_multiplier("worldwide") == 1.5

    def test_territory_multiplier_unknown(self):
        assert territory_multiplier("Mars") == 1.0

    def test_build_cue_sheet(self):
        placements = [
            SyncPlacement(MediaType.TV_SERIES, UsageType.BACKGROUND, 30.0, fee_usd=2000),
            SyncPlacement(MediaType.TV_SERIES, UsageType.END_CREDITS, 45.0, fee_usd=5000),
        ]
        cs = build_cue_sheet("Episode 1", placements)
        assert isinstance(cs, CueSheet)
        assert cs.total_fee == 7000
        assert cs.total_duration == 75.0

    def test_placement_value_score_at_market(self):
        fee = estimate_sync_fee(MediaType.FILM, UsageType.FEATURED, 60.0)
        p = SyncPlacement(MediaType.FILM, UsageType.FEATURED, 60.0, fee_usd=fee)
        score = placement_value_score(p)
        assert abs(score - 0.5) < 0.01

    def test_placement_value_score_zero_fee(self):
        p = SyncPlacement(MediaType.FILM, UsageType.FEATURED, 60.0, fee_usd=0)
        assert placement_value_score(p) == 0.0

    def test_catalog_sync_potential(self):
        placements = [
            SyncPlacement(MediaType.COMMERCIAL, UsageType.FEATURED, 30.0, fee_usd=80000),
            SyncPlacement(MediaType.PODCAST, UsageType.THEME, 60.0, fee_usd=1500),
        ]
        pot = catalog_sync_potential(placements)
        assert pot["total_fee"] == 81500
        assert pot["mean_value_score"] > 0

    def test_catalog_sync_potential_empty(self):
        pot = catalog_sync_potential([])
        assert pot["total_fee"] == 0.0


class TestPhraseGeneration:
    def test_classify_contour_ascending(self):
        assert classify_contour([60, 62, 64, 66, 68, 70]) == "ascending"

    def test_classify_contour_descending(self):
        assert classify_contour([72, 70, 68, 66, 64, 62]) == "descending"

    def test_classify_contour_static(self):
        assert classify_contour([60]) == "static"

    def test_generate_tension_arc_shape(self):
        arc = generate_tension_arc(20, peak_position=0.7)
        assert len(arc) == 20
        assert np.all(arc >= 0)
        assert np.all(arc <= 1.0)
        peak_idx = np.argmax(arc)
        assert 10 <= peak_idx <= 18

    def test_rhythmic_variation_length(self):
        base = [0.5, 0.5, 0.25, 0.25]
        varied = rhythmic_variation(base, 0.3, rng=np.random.default_rng(42))
        assert len(varied) == 4
        assert not all(a == b for a, b in zip(base, varied))

    def test_pitch_displacement(self):
        ref = [60, 64, 67]
        displaced = pitch_displacement(ref, target_pitch=72, gravity=0.5)
        assert all(d >= p for d, p in zip(displaced, ref))

    def test_phrase_density(self):
        assert phrase_density([60, 62, 64, 65], 2.0) == 2.0

    def test_intervallic_variety(self):
        v = intervallic_variety([60, 62, 64, 67, 72])
        assert 0 < v <= 1.0

    def test_intervallic_variety_constant(self):
        v = intervallic_variety([60, 62, 64, 66])
        assert abs(v - 1.0 / 3.0) < 1e-9

    def test_build_phrase_arch(self):
        scale = [60, 62, 64, 65, 67, 69, 71, 72]
        phrase = build_phrase(scale, n_notes=8, contour="arch", rng=np.random.default_rng(0))
        assert isinstance(phrase, PhraseContour)
        assert len(phrase.pitches) == 8
        assert phrase.contour_type == "arch"
        assert phrase.range > 0

    def test_build_phrase_ascending(self):
        scale = [48, 50, 52, 53, 55, 57, 59, 60]
        phrase = build_phrase(scale, n_notes=5, contour="ascending")
        assert phrase.pitches[0] <= phrase.pitches[-1]


class TestMoodOntology:
    def test_mood_distance_same(self):
        assert mood_distance(MoodCategory.HAPPY, MoodCategory.HAPPY) == 0.0

    def test_mood_distance_opposite(self):
        d = mood_distance(MoodCategory.HAPPY, MoodCategory.SAD)
        assert d > 1.0

    def test_classify_mood_happy(self):
        m = classify_mood(0.8, 0.7)
        assert m == MoodCategory.HAPPY

    def test_classify_mood_sad(self):
        m = classify_mood(-0.6, -0.4)
        assert m == MoodCategory.SAD

    def test_mood_trajectory_length(self):
        traj = mood_trajectory(MoodCategory.SAD, MoodCategory.HAPPY, n_steps=5)
        assert len(traj) == 5
        assert isinstance(traj[0], MoodState)

    def test_mood_trajectory_endpoints(self):
        traj = mood_trajectory(MoodCategory.CALM, MoodCategory.ENERGETIC, n_steps=3)
        v_calm, a_calm = MOOD_VA_COORDINATES[MoodCategory.CALM]
        assert abs(traj[0].valence - v_calm) < 1e-9
        assert abs(traj[0].arousal - a_calm) < 1e-9

    def test_plan_mood_transition(self):
        t = plan_mood_transition(MoodCategory.ANXIOUS, MoodCategory.PEACEFUL)
        assert isinstance(t, MoodTransition)
        assert t.recommended_steps >= 2
        assert t.distance > 0

    def test_mood_music_features(self):
        feat = mood_music_features(MoodCategory.ENERGETIC)
        assert feat["target_tempo"] > 100
        assert feat["target_energy"] > 0.5

    def test_mood_state_quadrant(self):
        s = MoodState(valence=0.5, arousal=0.5)
        assert s.quadrant == "Q1_happy_energetic"
        s2 = MoodState(valence=-0.5, arousal=-0.5)
        assert s2.quadrant == "Q3_sad_depressed"

    def test_mood_consistency_same(self):
        seq = [MoodCategory.HAPPY, MoodCategory.HAPPY, MoodCategory.HAPPY]
        assert mood_consistency(seq) == 1.0

    def test_mood_consistency_varied(self):
        seq = [MoodCategory.HAPPY, MoodCategory.SAD, MoodCategory.ANGRY, MoodCategory.CALM]
        c = mood_consistency(seq)
        assert 0 < c < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
