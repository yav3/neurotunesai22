"""Tests for music domain modules: theory, emotion, features, NLP, playlist."""
from __future__ import annotations

import math
import numpy as np
import pytest


# ── Music Theory ──────────────────────────────────────────────────────
class TestIntervals:
    def test_note_to_index(self):
        from models.music_theory.intervals import note_to_index
        assert note_to_index("C") == 0
        assert note_to_index("A") == 9
        assert note_to_index("C#") == 1
        assert note_to_index("Db") == 1

    def test_interval_between(self):
        from models.music_theory.intervals import interval_between
        assert interval_between("C", "G") == 7
        assert interval_between("C", "C") == 0
        assert interval_between("A", "C") == 3

    def test_interval_name(self):
        from models.music_theory.intervals import interval_name
        assert interval_name(0) == "unison"
        assert interval_name(7) == "perfect 5th"
        assert interval_name(12) == "unison"  # wraps mod 12


class TestScales:
    def test_major_scale_construction(self):
        from models.music_theory.scales import get_scale
        s = get_scale("C", "major")
        assert s.notes == ("C", "D", "E", "F", "G", "A", "B")

    def test_minor_scale(self):
        from models.music_theory.scales import get_scale
        s = get_scale("A", "natural_minor")
        assert s.notes == ("A", "B", "C", "D", "E", "F", "G")

    def test_pentatonic(self):
        from models.music_theory.scales import get_scale
        s = get_scale("C", "pentatonic_major")
        assert len(s.notes) == 5

    def test_unknown_scale_raises(self):
        from models.music_theory.scales import get_scale
        with pytest.raises(ValueError):
            get_scale("C", "nonexistent_scale")

    def test_relative_minor(self):
        from models.music_theory.scales import get_relative_minor
        assert get_relative_minor("C") == "A"

    def test_detect_scale(self):
        from models.music_theory.scales import detect_scale_from_notes
        results = detect_scale_from_notes(["C", "D", "E", "F", "G", "A", "B"])
        assert len(results) > 0
        assert results[0][0] == "C"
        assert results[0][2] > 0.8


class TestChords:
    def test_major_chord(self):
        from models.music_theory.chords import ChordLibrary
        c = ChordLibrary.build_chord("C", "major")
        assert "C" in c.notes
        assert "E" in c.notes
        assert "G" in c.notes

    def test_minor_chord(self):
        from models.music_theory.chords import ChordLibrary
        c = ChordLibrary.build_chord("A", "minor")
        assert len(c.notes) == 3


class TestProgressions:
    def test_common_progressions_exist(self):
        from models.music_theory.progressions import COMMON_PROGRESSIONS
        assert "I-IV-V-I" in COMMON_PROGRESSIONS or len(COMMON_PROGRESSIONS) > 0


# ── Music Emotion ─────────────────────────────────────────────────────
class TestRussellCircumplex:
    def test_nearest_emotion(self):
        from models.music_emotion.russell_circumplex import RussellCircumplex
        rc = RussellCircumplex()
        e = rc.nearest_emotion(0.8, 0.6)
        assert e.name == "happy"

    def test_nearest_sad(self):
        from models.music_emotion.russell_circumplex import RussellCircumplex
        rc = RussellCircumplex()
        e = rc.nearest_emotion(-0.7, -0.5)
        assert e.name == "sad"

    def test_emotional_trajectory(self):
        from models.music_emotion.russell_circumplex import RussellCircumplex
        rc = RussellCircumplex()
        traj = rc.emotional_trajectory([(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)])
        assert traj["total_distance"] > 0
        assert "dominant_quadrant" in traj

    def test_empty_trajectory(self):
        from models.music_emotion.russell_circumplex import RussellCircumplex
        rc = RussellCircumplex()
        traj = rc.emotional_trajectory([])
        assert traj["total_distance"] == 0.0


class TestPlutchik:
    def test_plutchik_wheel_has_opposites(self):
        from models.music_emotion.plutchik import PlutchikWheel
        pw = PlutchikWheel()
        assert len(pw.OPPOSITES) > 0

    def test_emotion_blend(self):
        from models.music_emotion.plutchik import PlutchikWheel, PlutchikEmotion
        pw = PlutchikWheel()
        blend = pw.blend_emotions([(PlutchikEmotion.JOY, 0.8), (PlutchikEmotion.TRUST, 0.6)])
        assert blend is not None
        assert len(blend) > 0


class TestMusicEmotionProfile:
    def test_profile_from_features(self):
        from models.music_emotion.music_emotion_profile import (
            MusicEmotionMapper, MusicFeatureVector,
        )
        mapper = MusicEmotionMapper()
        features = MusicFeatureVector(
            bpm=120.0,
            loudness_db=-8.0,
            energy=0.5,
            mode=1.0,
            brightness=0.7,
            danceability=0.6,
        )
        profile = mapper.features_to_emotion_profile(features)
        assert 0.0 <= profile.valence <= 1.0
        assert 0.0 <= profile.arousal <= 1.0


# ── Music Features ────────────────────────────────────────────────────
class TestSpectralExtractor:
    def test_extract_from_noise(self):
        from models.music_features.spectral import SpectralExtractor
        se = SpectralExtractor(sr=22050)
        audio = np.random.randn(22050).astype(np.float64)
        feats = se.extract(audio)
        assert feats.spectral_centroid > 0
        assert feats.zero_crossing_rate > 0
        assert len(feats.mfcc) == 13
        assert len(feats.chroma) == 12

    def test_extract_from_sine(self):
        from models.music_features.spectral import SpectralExtractor
        se = SpectralExtractor(sr=22050)
        t = np.linspace(0, 1, 22050)
        audio = np.sin(2 * np.pi * 440 * t)
        feats = se.extract(audio)
        assert feats.spectral_centroid > 0
        assert feats.spectral_flatness < 0.5  # pure tone → low flatness

    def test_extract_short_audio(self):
        from models.music_features.spectral import SpectralExtractor
        se = SpectralExtractor(sr=22050)
        audio = np.random.randn(100).astype(np.float64)
        feats = se.extract(audio)
        assert feats.spectral_centroid >= 0


class TestRhythmExtractor:
    def test_extract_rhythm(self):
        from models.music_features.rhythm import RhythmExtractor
        re_ = RhythmExtractor(sr=22050)
        audio = np.random.randn(22050 * 3).astype(np.float64)
        feats = re_.extract(audio)
        assert feats.bpm > 0


# ── Music NLP ─────────────────────────────────────────────────────────
class TestMultilingual:
    def test_detect_english(self):
        from models.music_nlp.multilingual import MultilingualProcessor
        mp = MultilingualProcessor()
        lang, conf = mp.detect_language("I want to hold your hand")
        assert lang == "en"
        assert conf > 0.0

    def test_detect_spanish(self):
        from models.music_nlp.multilingual import MultilingualProcessor
        mp = MultilingualProcessor()
        lang, _ = mp.detect_language("No puedo vivir sin tu amor")
        assert lang == "es"

    def test_detect_japanese_script(self):
        from models.music_nlp.multilingual import MultilingualProcessor
        mp = MultilingualProcessor()
        lang, _ = mp.detect_language("きらきら星がきれいです")
        assert lang == "ja"

    def test_tokenize_english(self):
        from models.music_nlp.multilingual import MultilingualProcessor
        mp = MultilingualProcessor()
        tokens = mp.tokenize("Hello world", "en")
        assert tokens == ["Hello", "world"]

    def test_remove_stopwords(self):
        from models.music_nlp.multilingual import MultilingualProcessor
        mp = MultilingualProcessor()
        tokens = ["I", "love", "the", "music"]
        filtered = mp.remove_stopwords(tokens, "en")
        assert "love" in filtered
        assert "music" in filtered
        assert "the" not in filtered

    def test_syllable_count(self):
        from models.music_nlp.multilingual import MultilingualProcessor
        mp = MultilingualProcessor()
        assert mp.syllable_count("love") == 1
        assert mp.syllable_count("music") == 2
        assert mp.syllable_count("beautiful") == 3


# ── Playlist / Transition ─────────────────────────────────────────────
class TestTransitionScorer:
    def test_same_key_high_score(self):
        from models.music_playlist.transition import TransitionScorer, TransitionType
        ts = TransitionScorer()
        score = ts.score(
            {"key": "C major", "bpm": 120, "energy": 0.7, "valence": 0.5, "arousal": 0.5},
            {"key": "C major", "bpm": 120, "energy": 0.7, "valence": 0.5, "arousal": 0.5},
        )
        assert score.overall > 0.9
        assert score.transition_type == TransitionType.SAME_KEY

    def test_adjacent_key(self):
        from models.music_playlist.transition import TransitionScorer, TransitionType
        ts = TransitionScorer()
        score = ts.score(
            {"key": "C major", "bpm": 120, "energy": 0.7, "valence": 0.5, "arousal": 0.5},
            {"key": "G major", "bpm": 122, "energy": 0.65, "valence": 0.5, "arousal": 0.5},
        )
        assert score.harmonic > 0.7
        assert score.transition_type == TransitionType.ADJACENT_KEY

    def test_incompatible_transition(self):
        from models.music_playlist.transition import TransitionScorer
        ts = TransitionScorer()
        score = ts.score(
            {"key": "C major", "bpm": 80, "energy": 0.2, "valence": 0.8, "arousal": 0.1},
            {"key": "F# minor", "bpm": 170, "energy": 0.9, "valence": 0.1, "arousal": 0.9},
        )
        assert score.overall < 0.5

    def test_rank_transitions(self):
        from models.music_playlist.transition import TransitionScorer
        ts = TransitionScorer()
        current = {"key": "C major", "bpm": 120, "energy": 0.6, "valence": 0.5, "arousal": 0.5}
        candidates = [
            {"key": "C major", "bpm": 122, "energy": 0.6, "valence": 0.5, "arousal": 0.5},
            {"key": "F# minor", "bpm": 170, "energy": 0.9, "valence": 0.1, "arousal": 0.9},
            {"key": "G major", "bpm": 118, "energy": 0.55, "valence": 0.5, "arousal": 0.5},
        ]
        ranked = ts.rank_transitions(current, candidates)
        assert ranked[0][1].overall >= ranked[-1][1].overall


class TestEnergyCurve:
    def test_ascending_curve(self):
        from models.music_playlist.energy_curve import EnergyCurveDesigner, EnergyCurveShape
        designer = EnergyCurveDesigner()
        curve = designer.design(EnergyCurveShape.LINEAR_UP, n_slots=8)
        assert len(curve.values) == 8
        assert curve.values[-1] > curve.values[0]

    def test_descending_curve(self):
        from models.music_playlist.energy_curve import EnergyCurveDesigner, EnergyCurveShape
        designer = EnergyCurveDesigner()
        curve = designer.design(EnergyCurveShape.LINEAR_DOWN, n_slots=8)
        assert curve.values[0] > curve.values[-1]
