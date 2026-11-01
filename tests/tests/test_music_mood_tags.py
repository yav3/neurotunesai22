"""Tests for mood tags: taxonomy, VAD mapping, tag profiles, similarity."""
import pytest
import numpy as np

from models.music_semantics.mood_tags import (
    MoodCategory, MoodTag, MOOD_VAD_MAP, MUSIC_TAG_TAXONOMY,
    TagProfile, vad_to_mood, tag_similarity,
)


class TestMoodCategory:
    def test_sixteen_moods(self):
        assert len(MoodCategory) == 16

    def test_values_lowercase(self):
        for m in MoodCategory:
            assert m.value == m.value.lower()


class TestMoodVADMap:
    def test_all_categories_mapped(self):
        for m in MoodCategory:
            assert m in MOOD_VAD_MAP

    def test_vad_bounded(self):
        for m, (v, a, d) in MOOD_VAD_MAP.items():
            assert 0.0 <= v <= 1.0, f"{m} valence out of range"
            assert 0.0 <= a <= 1.0, f"{m} arousal out of range"
            assert 0.0 <= d <= 1.0, f"{m} dominance out of range"

    def test_happy_high_valence(self):
        v, a, d = MOOD_VAD_MAP[MoodCategory.HAPPY]
        assert v > 0.7

    def test_sad_low_valence(self):
        v, a, d = MOOD_VAD_MAP[MoodCategory.SAD]
        assert v < 0.3

    def test_energetic_high_arousal(self):
        v, a, d = MOOD_VAD_MAP[MoodCategory.ENERGETIC]
        assert a > 0.8

    def test_calm_low_arousal(self):
        v, a, d = MOOD_VAD_MAP[MoodCategory.CALM]
        assert a < 0.2


class TestMoodTag:
    def test_frozen(self):
        tag = MoodTag(MoodCategory.HAPPY)
        with pytest.raises(AttributeError):
            tag.category = MoodCategory.SAD  # type: ignore[misc]

    def test_vad_full_intensity(self):
        tag = MoodTag(MoodCategory.HAPPY, intensity=1.0)
        base = MOOD_VAD_MAP[MoodCategory.HAPPY]
        assert tag.vad == base

    def test_vad_zero_intensity(self):
        tag = MoodTag(MoodCategory.HAPPY, intensity=0.0)
        v, a, d = tag.vad
        assert abs(v - 0.5) < 1e-9
        assert abs(a - 0.5) < 1e-9
        assert abs(d - 0.5) < 1e-9

    def test_vad_mid_intensity(self):
        tag = MoodTag(MoodCategory.SAD, intensity=0.5)
        v, a, d = tag.vad
        base_v = MOOD_VAD_MAP[MoodCategory.SAD][0]
        expected_v = base_v * 0.5 + 0.5 * 0.5
        assert abs(v - expected_v) < 1e-9


class TestVadToMood:
    def test_returns_list(self):
        result = vad_to_mood(0.8, 0.6, 0.7)
        assert isinstance(result, list)

    def test_top_k(self):
        result = vad_to_mood(0.8, 0.6, 0.7, top_k=5)
        assert len(result) == 5

    def test_happy_nearest(self):
        result = vad_to_mood(0.8, 0.6, 0.7, top_k=1)
        assert result[0][0] == MoodCategory.HAPPY

    def test_sad_nearest(self):
        result = vad_to_mood(0.2, 0.2, 0.3, top_k=1)
        assert result[0][0] == MoodCategory.SAD

    def test_similarity_bounded(self):
        result = vad_to_mood(0.5, 0.5, 0.5)
        for _, sim in result:
            assert 0.0 <= sim <= 1.0

    def test_sorted_descending(self):
        result = vad_to_mood(0.5, 0.5, 0.5, top_k=5)
        sims = [s for _, s in result]
        assert sims == sorted(sims, reverse=True)


class TestMusicTagTaxonomy:
    def test_categories_present(self):
        for cat in ("genre", "era", "instrument", "tempo", "mood", "use_case"):
            assert cat in MUSIC_TAG_TAXONOMY

    def test_genre_count(self):
        assert len(MUSIC_TAG_TAXONOMY["genre"]) >= 20

    def test_mood_matches_enum(self):
        mood_values = {m.value for m in MoodCategory}
        taxonomy_moods = set(MUSIC_TAG_TAXONOMY["mood"])
        assert taxonomy_moods == mood_values

    def test_tempo_ordered(self):
        assert MUSIC_TAG_TAXONOMY["tempo"] == [
            "very_slow", "slow", "moderate", "fast", "very_fast"
        ]


class TestTagProfile:
    def test_add_and_retrieve(self):
        tp = TagProfile()
        tp.add_tag("genre", "rock", 0.9)
        tp.add_tag("genre", "blues", 0.7)
        top = tp.top_tags("genre", k=1)
        assert top[0] == ("rock", 0.9)

    def test_top_tags_empty_category(self):
        tp = TagProfile()
        assert tp.top_tags("nonexistent") == []

    def test_all_tags_min_confidence(self):
        tp = TagProfile()
        tp.add_tag("genre", "rock", 0.9)
        tp.add_tag("genre", "pop", 0.3)
        result = tp.all_tags(min_confidence=0.5)
        assert len(result) == 1
        assert result[0][1] == "rock"

    def test_all_tags_sorted_by_confidence(self):
        tp = TagProfile()
        tp.add_tag("genre", "rock", 0.5)
        tp.add_tag("mood", "happy", 0.9)
        tp.add_tag("era", "1990s", 0.7)
        result = tp.all_tags(0.0)
        confs = [c for _, _, c in result]
        assert confs == sorted(confs, reverse=True)


class TestTagSimilarity:
    def test_identical_profiles(self):
        a = TagProfile()
        a.add_tag("genre", "rock", 0.9)
        assert tag_similarity(a, a) == 1.0

    def test_disjoint_profiles(self):
        a = TagProfile()
        a.add_tag("genre", "rock", 0.9)
        b = TagProfile()
        b.add_tag("genre", "jazz", 0.9)
        assert tag_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = TagProfile()
        a.add_tag("genre", "rock", 0.9)
        a.add_tag("mood", "happy", 0.8)
        b = TagProfile()
        b.add_tag("genre", "rock", 0.7)
        b.add_tag("mood", "sad", 0.6)
        sim = tag_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_both_empty(self):
        assert tag_similarity(TagProfile(), TagProfile()) == 1.0

    def test_one_empty(self):
        a = TagProfile()
        a.add_tag("genre", "rock", 0.9)
        assert tag_similarity(a, TagProfile()) == 0.0
