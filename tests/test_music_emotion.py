from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.music_emotion.plutchik import (
    EmotionIntensity,
    PlutchikEmotion,
    PlutchikWheel,
    primary_base,
    emotion_intensity,
)
from models.music_emotion.music_emotion_profile import (
    EmotionProfile,
    MusicEmotionMapper,
    MusicFeatureVector,
)
from models.music_emotion.russell_circumplex import RussellCircumplex, RussellEmotion
from models.music_emotion.gems import GEMSFactor, GEMSProfile, MusicToGEMS
from models.music_emotion.emotion_arc import EmotionArc, EmotionArcAnalyzer


# ── Plutchik ──────────────────────────────────────────────────────

class TestPlutchikWheel:
    def test_opposites_symmetric(self):
        for e, opp in PlutchikWheel.OPPOSITES.items():
            assert PlutchikWheel.OPPOSITES[opp] == e

    def test_opposites_distance_is_one(self):
        for e, opp in PlutchikWheel.OPPOSITES.items():
            assert PlutchikWheel.emotion_distance(e, opp) == pytest.approx(1.0)

    def test_same_emotion_distance_zero(self):
        assert PlutchikWheel.emotion_distance(
            PlutchikEmotion.JOY, PlutchikEmotion.JOY
        ) == 0.0

    def test_adjacent_distance(self):
        d = PlutchikWheel.emotion_distance(
            PlutchikEmotion.JOY, PlutchikEmotion.TRUST
        )
        assert d == pytest.approx(0.25)

    def test_intensity_variant_maps_to_base(self):
        assert primary_base(PlutchikEmotion.ECSTASY) == PlutchikEmotion.JOY
        assert primary_base(PlutchikEmotion.GRIEF) == PlutchikEmotion.SADNESS

    def test_intensity_levels(self):
        assert emotion_intensity(PlutchikEmotion.ECSTASY) == EmotionIntensity.HIGH
        assert emotion_intensity(PlutchikEmotion.JOY) == EmotionIntensity.MEDIUM
        assert emotion_intensity(PlutchikEmotion.SERENITY) == EmotionIntensity.LOW

    def test_dyad_components(self):
        love = PlutchikWheel.DYADS["love"]
        assert love.component_a == PlutchikEmotion.JOY
        assert love.component_b == PlutchikEmotion.TRUST

    def test_eight_primary_dyads(self):
        assert len(PlutchikWheel.DYADS) == 8

    def test_blend_single_emotion(self):
        result = PlutchikWheel.blend_emotions([(PlutchikEmotion.JOY, 1.0)])
        assert result["joy"] == 1.0
        assert result["sadness"] == 0.0

    def test_blend_two_emotions(self):
        result = PlutchikWheel.blend_emotions([
            (PlutchikEmotion.JOY, 0.6),
            (PlutchikEmotion.TRUST, 0.4),
        ])
        assert result["joy"] > result["trust"]
        assert result["joy"] > 0

    def test_blend_empty_returns_zeros(self):
        result = PlutchikWheel.blend_emotions([])
        assert all(v == 0.0 for v in result.values())

    def test_intensity_distance_uses_base(self):
        d = PlutchikWheel.emotion_distance(
            PlutchikEmotion.ECSTASY, PlutchikEmotion.GRIEF
        )
        assert d == pytest.approx(1.0)


# ── Music emotion mapping ────────────────────────────────────────

class TestMusicEmotionMapper:
    def test_joy_features_produce_high_valence(self):
        features = MusicFeatureVector(
            bpm=140, loudness_db=-3, danceability=0.85, acousticness=0.1,
            energy=0.9, mode=1.0, melodic_dissonance=0.1, percussiveness=0.8,
            isochrony_regularity=0.9, key_stability=0.9, brightness=0.8,
        )
        v, a, d = MusicEmotionMapper.features_to_vad(features)
        assert v > 0.6
        assert a > 0.5

    def test_sad_features_produce_low_valence_low_arousal(self):
        features = MusicFeatureVector(
            bpm=60, loudness_db=-18, danceability=0.2, acousticness=0.9,
            energy=0.15, mode=0.0, melodic_dissonance=0.15, percussiveness=0.1,
            brightness=0.2,
        )
        v, a, d = MusicEmotionMapper.features_to_vad(features)
        assert v < 0.4
        assert a < 0.4

    def test_anger_features_high_arousal_low_valence(self):
        features = MusicFeatureVector(
            bpm=170, loudness_db=-1, danceability=0.4, acousticness=0.05,
            energy=0.95, mode=0.0, melodic_dissonance=0.7, percussiveness=0.9,
            brightness=0.7,
        )
        v, a, d = MusicEmotionMapper.features_to_vad(features)
        assert a > 0.6
        assert v < 0.5

    def test_vad_to_plutchik_sums_to_one(self):
        scores = MusicEmotionMapper.vad_to_plutchik(0.5, 0.5, 0.5)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)

    def test_vad_to_plutchik_joy_centroid(self):
        scores = MusicEmotionMapper.vad_to_plutchik(0.9, 0.7, 0.7)
        assert scores[PlutchikEmotion.JOY] == max(scores.values())

    def test_vad_to_plutchik_sadness_centroid(self):
        scores = MusicEmotionMapper.vad_to_plutchik(0.15, 0.2, 0.25)
        assert scores[PlutchikEmotion.SADNESS] == max(scores.values())

    def test_full_pipeline_joy(self):
        features = MusicFeatureVector(
            bpm=140, loudness_db=-3, danceability=0.85, acousticness=0.1,
            energy=0.9, mode=1.0, brightness=0.8,
        )
        profile = MusicEmotionMapper.features_to_emotion_profile(features)
        assert profile.primary_emotion in (
            PlutchikEmotion.JOY, PlutchikEmotion.ANTICIPATION
        )
        assert profile.quadrant == "high_arousal_high_valence"

    def test_full_pipeline_sad(self):
        features = MusicFeatureVector(
            bpm=55, loudness_db=-18, danceability=0.15, acousticness=0.9,
            energy=0.1, mode=0.0, brightness=0.15,
        )
        profile = MusicEmotionMapper.features_to_emotion_profile(features)
        assert profile.primary_emotion in (
            PlutchikEmotion.SADNESS, PlutchikEmotion.DISGUST
        )
        assert profile.quadrant == "low_arousal_low_valence"

    def test_classify_quadrant(self):
        assert MusicEmotionMapper.classify_quadrant(0.8, 0.9) == "high_arousal_high_valence"
        assert MusicEmotionMapper.classify_quadrant(0.1, 0.1) == "low_arousal_low_valence"
        assert MusicEmotionMapper.classify_quadrant(0.1, 0.9) == "high_arousal_low_valence"
        assert MusicEmotionMapper.classify_quadrant(0.9, 0.1) == "low_arousal_high_valence"

    def test_vad_clamped_to_unit(self):
        features = MusicFeatureVector(bpm=250, energy=1.5, brightness=2.0, mode=1.0)
        v, a, d = MusicEmotionMapper.features_to_vad(features)
        assert 0.0 <= v <= 1.0
        assert 0.0 <= a <= 1.0
        assert 0.0 <= d <= 1.0

    def test_default_features_mid_range(self):
        features = MusicFeatureVector()
        v, a, d = MusicEmotionMapper.features_to_vad(features)
        assert 0.2 < v < 0.9
        assert 0.2 < a < 0.8


# ── Russell circumplex ────────────────────────────────────────────

class TestRussellCircumplex:
    def test_nearest_happy(self):
        rc = RussellCircumplex()
        e = rc.nearest_emotion(0.8, 0.6)
        assert e.name == "happy"

    def test_nearest_sad(self):
        rc = RussellCircumplex()
        e = rc.nearest_emotion(-0.7, -0.5)
        assert e.name == "sad"

    def test_emotion_vector_angle(self):
        angle, mag = RussellCircumplex.emotion_vector(1.0, 0.0)
        assert angle == pytest.approx(0.0)

    def test_emotion_vector_magnitude(self):
        _, mag = RussellCircumplex.emotion_vector(0.6, 0.8)
        assert mag == pytest.approx(1.0)

    def test_trajectory_stats(self):
        rc = RussellCircumplex()
        seq = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]
        stats = rc.emotional_trajectory(seq)
        assert stats["total_distance"] > 0
        assert stats["mean_valence"] == pytest.approx(0.5)
        assert stats["dominant_quadrant"] == "high_arousal_high_valence"

    def test_trajectory_empty(self):
        rc = RussellCircumplex()
        stats = rc.emotional_trajectory([])
        assert stats["total_distance"] == 0.0

    def test_emotional_distance_same(self):
        e = RussellEmotion("test", 0.5, 0.5, 45.0)
        assert RussellCircumplex.emotional_distance(e, e) == 0.0


# ── GEMS ──────────────────────────────────────────────────────────

class TestGEMS:
    def test_vad_to_gems_all_factors_present(self):
        profile = MusicToGEMS.vad_to_gems(0.5, 0.5, 0.5)
        assert len(profile.scores) == 9

    def test_dominant_joyful_activation(self):
        profile = MusicToGEMS.vad_to_gems(0.9, 0.8, 0.7)
        dominant = MusicToGEMS.gems_dominant(profile)
        assert dominant == GEMSFactor.JOYFUL_ACTIVATION

    def test_dominant_sadness(self):
        profile = MusicToGEMS.vad_to_gems(0.15, 0.2, 0.25)
        dominant = MusicToGEMS.gems_dominant(profile)
        assert dominant == GEMSFactor.SADNESS

    def test_gems_valence_positive_for_joy(self):
        profile = MusicToGEMS.vad_to_gems(0.9, 0.8, 0.7)
        assert MusicToGEMS.gems_valence(profile) > 0.3

    def test_gems_valence_negative_for_sad(self):
        profile = MusicToGEMS.vad_to_gems(0.15, 0.2, 0.25)
        assert MusicToGEMS.gems_valence(profile) < 0.2

    def test_gems_scores_normalised_max_one(self):
        profile = MusicToGEMS.vad_to_gems(0.5, 0.5, 0.5)
        assert max(profile.scores.values()) == pytest.approx(1.0, abs=0.05)


# ── Emotion arcs ──────────────────────────────────────────────────

class TestEmotionArc:
    def test_generate_ramp_up(self):
        arc = EmotionArcAnalyzer.generate_target_arc("ramp_up", 180.0, 50)
        assert len(arc.timestamps) == 50
        assert arc.arousal_curve[-1] > arc.arousal_curve[0]

    def test_generate_peak(self):
        arc = EmotionArcAnalyzer.generate_target_arc("peak", 200.0, 100)
        mid = len(arc.arousal_curve) // 2
        assert arc.arousal_curve[mid] > arc.arousal_curve[0]
        assert arc.arousal_curve[mid] > arc.arousal_curve[-1]

    def test_detect_ramp_up(self):
        arc = EmotionArcAnalyzer.generate_target_arc("ramp_up", 180.0, 80)
        shape, conf = EmotionArcAnalyzer.detect_arc_shape(arc)
        assert shape == "ramp_up"
        assert conf > 0.3

    def test_detect_flat(self):
        arc = EmotionArcAnalyzer.generate_target_arc("flat", 60.0, 50)
        shape, _ = EmotionArcAnalyzer.detect_arc_shape(arc)
        assert shape == "flat"

    def test_detect_oscillating(self):
        arc = EmotionArcAnalyzer.generate_target_arc("oscillating", 120.0, 100)
        shape, _ = EmotionArcAnalyzer.detect_arc_shape(arc)
        assert shape == "oscillating"

    def test_complexity_flat_low(self):
        arc = EmotionArcAnalyzer.generate_target_arc("flat", 60.0, 50)
        c = EmotionArcAnalyzer.arc_complexity(arc)
        assert c < 0.15

    def test_complexity_oscillating_high(self):
        arc = EmotionArcAnalyzer.generate_target_arc("oscillating", 120.0, 100)
        c = EmotionArcAnalyzer.arc_complexity(arc)
        assert c > 0.15

    def test_journey_distance_positive(self):
        arc = EmotionArcAnalyzer.generate_target_arc("peak", 120.0, 50)
        d = EmotionArcAnalyzer.emotional_journey_distance(arc)
        assert d > 0

    def test_journey_distance_flat_near_zero(self):
        arc = EmotionArcAnalyzer.generate_target_arc("flat", 60.0, 50)
        d = EmotionArcAnalyzer.emotional_journey_distance(arc)
        assert d < 0.01

    def test_arc_shapes_list(self):
        from models.music_emotion.emotion_arc import ARC_SHAPES
        assert "peak" in ARC_SHAPES
        assert "tension_release" in ARC_SHAPES
        assert len(ARC_SHAPES) == 7

    def test_short_arc_returns_flat(self):
        arc = EmotionArc(
            timestamps=[0.0, 1.0],
            valence_curve=[0.5, 0.5],
            arousal_curve=[0.5, 0.5],
        )
        shape, _ = EmotionArcAnalyzer.detect_arc_shape(arc)
        assert shape == "flat"


# ── Edge cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_bpm(self):
        f = MusicFeatureVector(bpm=0.0)
        v, a, d = MusicEmotionMapper.features_to_vad(f)
        assert 0.0 <= a <= 1.0

    def test_extreme_bpm(self):
        f = MusicFeatureVector(bpm=300.0)
        v, a, d = MusicEmotionMapper.features_to_vad(f)
        assert 0.0 <= a <= 1.0

    def test_all_zeros_feature_vector(self):
        f = MusicFeatureVector(
            bpm=0, loudness_db=-60, danceability=0, acousticness=0,
            energy=0, mode=0, melodic_dissonance=0, percussiveness=0,
            isochrony_regularity=0, key_stability=0, brightness=0,
            melodic_direction=0, dynamic_range=0, harmonic_complexity=0,
        )
        profile = MusicEmotionMapper.features_to_emotion_profile(f)
        assert 0.0 <= profile.valence <= 1.0
        assert profile.primary_emotion is not None
