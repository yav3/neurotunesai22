"""Tests for musical affect: circumplex, Thayer, Hevner, emotion arcs, analyzer."""
import math
import pytest

from models.music_semantics.musical_affect import (
    AffectPoint, EmotionArc, CircumplexModel, MusicalAffectAnalyzer,
    HEVNER_CLUSTERS, THAYER_QUADRANTS,
    affect_distance, arc_similarity,
)


# --- Hevner / Thayer constants ---

class TestHevnerClusters:
    def test_eight_clusters(self):
        assert len(HEVNER_CLUSTERS) == 8

    def test_cluster_ids(self):
        assert set(HEVNER_CLUSTERS.keys()) == set(range(1, 9))

    def test_all_have_adjectives(self):
        for cid, data in HEVNER_CLUSTERS.items():
            assert len(data["adjectives"]) >= 5, f"Cluster {cid} too few adjectives"

    def test_valence_arousal_bounded(self):
        for cid, data in HEVNER_CLUSTERS.items():
            assert 0.0 <= data["valence"] <= 1.0
            assert 0.0 <= data["arousal"] <= 1.0


class TestThayerQuadrants:
    def test_four_quadrants(self):
        assert len(THAYER_QUADRANTS) == 4

    def test_quadrant_names(self):
        assert set(THAYER_QUADRANTS.keys()) == {"contentment", "depression", "exuberance", "anxiety"}

    def test_ranges_cover_unit_square(self):
        for name, q in THAYER_QUADRANTS.items():
            elo, ehi = q["energy_range"]
            tlo, thi = q["tension_range"]
            assert 0.0 <= elo < ehi <= 1.0
            assert 0.0 <= tlo < thi <= 1.0


# --- AffectPoint ---

class TestAffectPoint:
    def test_defaults(self):
        ap = AffectPoint()
        assert ap.valence == 0.5
        assert ap.arousal == 0.5
        assert ap.dominance == 0.5

    def test_energy_equals_arousal(self):
        ap = AffectPoint(arousal=0.7)
        assert ap.energy == 0.7

    def test_tension_bounded(self):
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
                ap = AffectPoint(valence=v, arousal=a)
                assert 0.0 <= ap.tension <= 1.0

    def test_circumplex_angle_center(self):
        ap = AffectPoint(valence=0.5, arousal=0.5)
        assert ap.circumplex_angle() == 0.0

    def test_circumplex_angle_high_valence(self):
        ap = AffectPoint(valence=1.0, arousal=0.5)
        assert ap.circumplex_angle() == 0.0

    def test_circumplex_angle_high_arousal(self):
        ap = AffectPoint(valence=0.5, arousal=1.0)
        assert abs(ap.circumplex_angle() - math.pi / 2) < 1e-9

    def test_circumplex_radius_center(self):
        ap = AffectPoint(valence=0.5, arousal=0.5)
        assert ap.circumplex_radius() < 1e-9

    def test_circumplex_radius_corner(self):
        ap = AffectPoint(valence=1.0, arousal=1.0)
        assert abs(ap.circumplex_radius() - math.sqrt(0.5)) < 1e-9

    def test_thayer_contentment(self):
        ap = AffectPoint(valence=0.8, arousal=0.2)
        assert ap.thayer_quadrant() == "contentment"

    def test_thayer_exuberance(self):
        ap = AffectPoint(valence=0.9, arousal=0.8)
        assert ap.thayer_quadrant() == "exuberance"

    def test_nearest_hevner_happy(self):
        ap = AffectPoint(valence=0.9, arousal=0.7)
        cid = ap.nearest_hevner_cluster()
        assert cid == 6  # Happy/Joyous

    def test_nearest_hevner_sad(self):
        ap = AffectPoint(valence=0.15, arousal=0.25)
        cid = ap.nearest_hevner_cluster()
        assert cid == 2  # Sad/Mournful

    def test_nearest_hevner_serene(self):
        ap = AffectPoint(valence=0.7, arousal=0.1)
        cid = ap.nearest_hevner_cluster()
        assert cid == 4  # Serene/Tranquil


# --- EmotionArc ---

class TestEmotionArc:
    def test_empty_arc(self):
        arc = EmotionArc()
        assert arc.length == 0

    def test_add_points(self):
        arc = EmotionArc()
        arc.add(0.5, 0.5)
        arc.add(0.8, 0.3)
        assert arc.length == 2

    def test_valence_trajectory(self):
        arc = EmotionArc()
        arc.add(0.2, 0.5)
        arc.add(0.8, 0.5)
        assert arc.valence_trajectory() == [0.2, 0.8]

    def test_arousal_trajectory(self):
        arc = EmotionArc()
        arc.add(0.5, 0.1)
        arc.add(0.5, 0.9)
        assert arc.arousal_trajectory() == [0.1, 0.9]

    def test_mean_affect_empty(self):
        arc = EmotionArc()
        m = arc.mean_affect()
        assert m.valence == 0.5

    def test_mean_affect(self):
        arc = EmotionArc()
        arc.add(0.2, 0.4)
        arc.add(0.8, 0.6)
        m = arc.mean_affect()
        assert abs(m.valence - 0.5) < 1e-9
        assert abs(m.arousal - 0.5) < 1e-9

    def test_variance_single_point(self):
        arc = EmotionArc()
        arc.add(0.5, 0.5)
        assert arc.variance() == (0.0, 0.0)

    def test_variance_two_points(self):
        arc = EmotionArc()
        arc.add(0.0, 0.0)
        arc.add(1.0, 1.0)
        vv, va = arc.variance()
        assert vv > 0
        assert va > 0

    def test_emotional_range_single(self):
        arc = EmotionArc()
        arc.add(0.5, 0.5)
        assert arc.emotional_range() == 0.0

    def test_emotional_range(self):
        arc = EmotionArc()
        arc.add(0.0, 0.0)
        arc.add(1.0, 1.0)
        assert abs(arc.emotional_range() - math.sqrt(2)) < 1e-9

    def test_direction_changes_none(self):
        arc = EmotionArc()
        arc.add(0.1, 0.5)
        arc.add(0.5, 0.5)
        arc.add(0.9, 0.5)
        assert arc.direction_changes() == 0

    def test_direction_changes_one(self):
        arc = EmotionArc()
        arc.add(0.1, 0.5)
        arc.add(0.9, 0.5)
        arc.add(0.4, 0.5)
        assert arc.direction_changes() == 1

    def test_classify_flat(self):
        arc = EmotionArc()
        for _ in range(5):
            arc.add(0.5, 0.5)
        assert arc.classify_shape() == "flat"

    def test_classify_ascending(self):
        arc = EmotionArc()
        for i in range(6):
            arc.add(0.2 + i * 0.12, 0.5)
        assert arc.classify_shape() in ("ascending", "rags_to_riches")

    def test_classify_descending(self):
        arc = EmotionArc()
        for i in range(6):
            arc.add(0.9 - i * 0.12, 0.5)
        assert arc.classify_shape() in ("descending", "riches_to_rags")

    def test_classify_rags_to_riches(self):
        arc = EmotionArc()
        vals = [0.1, 0.1, 0.15, 0.5, 0.8, 0.9]
        for v in vals:
            arc.add(v, 0.5)
        shape = arc.classify_shape()
        assert shape in ("rags_to_riches", "ascending")

    def test_classify_short(self):
        arc = EmotionArc()
        arc.add(0.5, 0.5)
        arc.add(0.8, 0.3)
        assert arc.classify_shape() == "flat"


# --- CircumplexModel ---

class TestCircumplexModel:
    @pytest.fixture
    def model(self):
        return CircumplexModel()

    def test_classify_returns_list(self, model):
        result = model.classify(0.8, 0.7)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_classify_happy_top(self, model):
        result = model.classify(0.8, 0.7)
        assert result[0][0] == "happy"

    def test_classify_sad_top(self, model):
        result = model.classify(0.15, 0.25)
        assert result[0][0] == "sad"

    def test_classify_similarity_bounded(self, model):
        result = model.classify(0.5, 0.5)
        for _, sim in result:
            assert 0.0 <= sim <= 1.0

    def test_interpolate_midpoint(self, model):
        v, a = model.interpolate("happy", "sad", 0.5)
        assert 0.3 < v < 0.6
        assert 0.3 < a < 0.6

    def test_interpolate_start(self, model):
        v, a = model.interpolate("happy", "sad", 0.0)
        assert abs(v - 0.8) < 1e-9
        assert abs(a - 0.7) < 1e-9

    def test_interpolate_unknown(self, model):
        v, a = model.interpolate("nonexistent", "happy", 0.5)
        assert v == 0.5 and a == 0.5

    def test_angular_distance_same(self, model):
        d = model.angular_distance("happy", "happy")
        assert abs(d) < 1e-9

    def test_angular_distance_positive(self, model):
        d = model.angular_distance("happy", "sad")
        assert d > 0

    def test_angular_distance_bounded(self, model):
        d = model.angular_distance("happy", "sad")
        assert d <= math.pi

    def test_angular_distance_unknown(self, model):
        d = model.angular_distance("unknown", "happy")
        assert abs(d - math.pi) < 1e-9


# --- MusicalAffectAnalyzer ---

class TestMusicalAffectAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return MusicalAffectAnalyzer()

    def test_features_to_affect_returns_point(self, analyzer):
        ap = analyzer.features_to_affect(120, 1.0, -20, 2000, 0.8)
        assert isinstance(ap, AffectPoint)

    def test_features_to_affect_bounded(self, analyzer):
        ap = analyzer.features_to_affect(120, 1.0, -20, 2000, 0.8)
        assert 0.0 <= ap.valence <= 1.0
        assert 0.0 <= ap.arousal <= 1.0
        assert 0.0 <= ap.dominance <= 1.0

    def test_fast_tempo_high_arousal(self, analyzer):
        slow = analyzer.features_to_affect(60, 0.5, -20, 2000, 0.5)
        fast = analyzer.features_to_affect(180, 0.5, -20, 2000, 0.5)
        assert fast.arousal > slow.arousal

    def test_major_mode_higher_valence(self, analyzer):
        minor = analyzer.features_to_affect(120, 0.0, -20, 2000, 0.5)
        major = analyzer.features_to_affect(120, 1.0, -20, 2000, 0.5)
        assert major.valence > minor.valence

    def test_analyze_trajectory(self, analyzer):
        feats = [
            {"tempo": 80, "mode": 0.0, "loudness": -30},
            {"tempo": 120, "mode": 0.5, "loudness": -20},
            {"tempo": 160, "mode": 1.0, "loudness": -10},
        ]
        arc = analyzer.analyze_trajectory(feats)
        assert arc.length == 3
        vals = arc.valence_trajectory()
        assert vals[-1] > vals[0]

    def test_circumplex_classify(self, analyzer):
        result = analyzer.circumplex_classify(0.8, 0.7)
        assert len(result) == 3
        assert result[0][0] == "happy"

    def test_hevner_cluster_returns_dict(self, analyzer):
        result = analyzer.hevner_cluster(0.9, 0.7)
        assert "cluster_id" in result
        assert "adjectives" in result
        assert result["cluster_id"] == 6

    def test_transition_cost_zero(self, analyzer):
        assert analyzer.transition_cost(0.5, 0.5, 0.5, 0.5) == 0.0

    def test_transition_cost_positive(self, analyzer):
        cost = analyzer.transition_cost(0.0, 0.0, 1.0, 1.0)
        assert abs(cost - math.sqrt(2)) < 1e-9

    def test_smooth_trajectory(self, analyzer):
        arc = EmotionArc()
        arc.add(0.0, 0.0)
        arc.add(1.0, 1.0)
        arc.add(0.0, 0.0)
        arc.add(1.0, 1.0)
        arc.add(0.0, 0.0)
        smoothed = analyzer.smooth_trajectory(arc, window=3)
        assert smoothed.length == 5
        vv_orig, _ = arc.variance()
        vv_smooth, _ = smoothed.variance()
        assert vv_smooth < vv_orig

    def test_smooth_short_passthrough(self, analyzer):
        arc = EmotionArc()
        arc.add(0.5, 0.5)
        result = analyzer.smooth_trajectory(arc, window=3)
        assert result is arc


# --- Standalone functions ---

class TestStandaloneFunctions:
    def test_affect_distance_zero(self):
        a = AffectPoint(0.5, 0.5, 0.5)
        assert affect_distance(a, a) == 0.0

    def test_affect_distance_positive(self):
        a = AffectPoint(0.0, 0.0, 0.0)
        b = AffectPoint(1.0, 1.0, 1.0)
        assert abs(affect_distance(a, b) - math.sqrt(3)) < 1e-9

    def test_arc_similarity_identical(self):
        a = EmotionArc()
        a.add(0.5, 0.5)
        a.add(0.8, 0.3)
        assert abs(arc_similarity(a, a) - 1.0) < 1e-9

    def test_arc_similarity_empty(self):
        a = EmotionArc()
        b = EmotionArc()
        assert arc_similarity(a, b) == 0.0

    def test_arc_similarity_bounded(self):
        a = EmotionArc()
        a.add(0.0, 0.0)
        b = EmotionArc()
        b.add(1.0, 1.0)
        sim = arc_similarity(a, b)
        assert 0.0 <= sim <= 1.0


# --- Import from __init__ ---

class TestSemanticsExports:
    def test_import_affect(self):
        from models.music_semantics import (
            AffectPoint, EmotionArc, CircumplexModel, MusicalAffectAnalyzer,
            HEVNER_CLUSTERS, THAYER_QUADRANTS,
            affect_distance, arc_similarity,
        )
        assert callable(CircumplexModel)
        assert callable(MusicalAffectAnalyzer)

    def test_import_mood_tags(self):
        from models.music_semantics import (
            MoodCategory, MoodTag, MOOD_VAD_MAP, MUSIC_TAG_TAXONOMY,
            TagProfile, vad_to_mood, tag_similarity,
        )
        assert callable(vad_to_mood)
        assert isinstance(MOOD_VAD_MAP, dict)

    def test_import_knowledge_graph(self):
        from models.music_semantics import (
            EntityType, RelationType, Entity, Relation, MusicKnowledgeGraph,
        )
        assert callable(MusicKnowledgeGraph)
