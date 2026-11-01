"""Tests for music semantics: lyrics analysis and narrative arc."""
import numpy as np
import pytest

from models.music_semantics.lyrics_analysis import (
    syllable_count,
    syllable_density,
    rhyme_similarity,
    detect_rhyme_scheme,
    line_similarity_matrix,
    detect_repetition,
    sentiment_polarity,
    sentiment_trajectory,
    lyric_melody_alignment,
)
from models.music_semantics.narrative_arc import (
    tension_curve,
    smooth_tension,
    classify_arc,
    ArcType,
    emotional_trajectory,
    vad_distance,
    emotional_range,
    emotional_velocity,
    segment_entropy,
    surprise_curve,
    structural_surprise_score,
    VADPoint,
)


# ── Lyrics: syllables ──────────────────────────────────────────────────

class TestSyllables:
    def test_single_word(self):
        assert syllable_count("hello") == 2

    def test_monosyllable(self):
        assert syllable_count("cat") == 1

    def test_multisyllable(self):
        assert syllable_count("beautiful") == 3

    def test_sentence(self):
        assert syllable_count("I love you") == 3

    def test_density(self):
        lines = ["I love you", "Baby dance with me"]
        d = syllable_density(lines, beats_per_line=4.0)
        assert len(d) == 2
        assert d[0] == pytest.approx(3.0 / 4.0)


# ── Lyrics: rhyme ──────────────────────────────────────────────────────

class TestRhyme:
    def test_perfect_rhyme(self):
        assert rhyme_similarity("the cat sat on the mat", "I wore a hat") > 0.5

    def test_no_rhyme(self):
        assert rhyme_similarity("hello world", "goodbye moon") < 0.3

    def test_scheme_aabb(self):
        lines = ["I see the light", "everything is right",
                 "the moon above", "I feel the love"]
        scheme = detect_rhyme_scheme(lines)
        assert len(scheme) == 4
        assert scheme[0] == scheme[1]  # light/right
        assert scheme[2] == scheme[3]  # above/love

    def test_empty(self):
        assert detect_rhyme_scheme([]) == []


# ── Lyrics: repetition ─────────────────────────────────────────────────

class TestRepetition:
    def test_similarity_matrix_diagonal(self):
        lines = ["hello world", "goodbye moon"]
        mat = line_similarity_matrix(lines)
        assert mat[0, 0] == 1.0
        assert mat[1, 1] == 1.0

    def test_detect_chorus(self):
        lines = [
            "verse one is here",
            "something different now",
            "chorus sing along",
            "another verse goes",
            "chorus sing along",
            "final verse at last",
            "chorus sing along",
        ]
        result = detect_repetition(lines)
        assert result.repetition_score > 0
        chorus_indices = [i for i, l in enumerate(result.section_labels) if l == "chorus"]
        assert len(chorus_indices) >= 2

    def test_empty(self):
        r = detect_repetition([])
        assert r.section_labels == []
        assert r.repetition_score == 0.0


# ── Lyrics: sentiment ──────────────────────────────────────────────────

class TestSentiment:
    def test_positive(self):
        r = sentiment_polarity("I love this beautiful happy day")
        assert r.polarity > 0
        assert r.positive_count >= 3

    def test_negative(self):
        r = sentiment_polarity("pain and sorrow in the dark alone")
        assert r.polarity < 0
        assert r.negative_count >= 3

    def test_neutral(self):
        r = sentiment_polarity("the quick brown fox")
        assert r.polarity == 0.0

    def test_trajectory(self):
        lines = ["I love you baby", "pain and sorrow", "happy day"]
        t = sentiment_trajectory(lines)
        assert len(t) == 3
        assert t[0] > 0  # positive
        assert t[1] < 0  # negative


# ── Lyrics: alignment ──────────────────────────────────────────────────

class TestAlignment:
    def test_perfect(self):
        assert lyric_melody_alignment([4, 6, 3], [4, 6, 3]) == pytest.approx(1.0)

    def test_mismatch(self):
        score = lyric_melody_alignment([4, 6], [8, 12])
        assert 0 < score < 1.0

    def test_empty(self):
        assert lyric_melody_alignment([], []) == 0.0


# ── Narrative arc: tension ─────────────────────────────────────────────

class TestTension:
    def test_curve_shape(self):
        h = np.array([0.2, 0.5, 0.8, 0.6, 0.3])
        d = np.array([0.3, 0.4, 0.7, 0.5, 0.2])
        c = tension_curve(h, d)
        assert len(c) == 5
        assert c[2] > c[0]

    def test_smooth(self):
        c = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        s = smooth_tension(c, window=3)
        assert len(s) == 5
        assert np.std(s) < np.std(c)


# ── Narrative arc: classification ──────────────────────────────────────

class TestArcClassification:
    def test_ramp(self):
        c = np.linspace(0, 1, 20)
        arc = classify_arc(c)
        assert arc.arc_type == ArcType.RAMP

    def test_arch(self):
        c = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, 0, 10)])
        arc = classify_arc(c)
        assert arc.arc_type == ArcType.ARCH
        assert 0.4 <= arc.peak_position <= 0.6

    def test_flat(self):
        c = np.ones(20) * 0.5
        arc = classify_arc(c)
        assert arc.arc_type == ArcType.FLAT

    def test_short(self):
        arc = classify_arc(np.array([0.5]))
        assert arc.arc_type == ArcType.FLAT


# ── Emotional trajectory ───────────────────────────────────────────────

class TestEmotional:
    def test_trajectory(self):
        v = np.array([0.5, 0.8, 0.2])
        a = np.array([0.6, 0.9, 0.3])
        d = np.array([0.5, 0.5, 0.5])
        traj = emotional_trajectory(v, a, d)
        assert len(traj) == 3
        assert traj[1].valence == pytest.approx(0.8)

    def test_distance(self):
        a = VADPoint(0.0, 0.0, 0.0)
        b = VADPoint(1.0, 0.0, 0.0)
        assert vad_distance(a, b) == pytest.approx(1.0)

    def test_range(self):
        traj = [VADPoint(0, 0, 0), VADPoint(1, 1, 1)]
        assert emotional_range(traj) == pytest.approx(np.sqrt(3))

    def test_velocity(self):
        traj = [VADPoint(0, 0, 0), VADPoint(1, 0, 0), VADPoint(1, 0, 0)]
        vel = emotional_velocity(traj)
        assert len(vel) == 2
        assert vel[0] == pytest.approx(1.0)
        assert vel[1] == pytest.approx(0.0)


# ── Structural surprise ───────────────────────────────────────────────

class TestSurprise:
    def test_entropy_uniform(self):
        f = np.random.default_rng(42).uniform(0, 1, 100)
        h = segment_entropy(f)
        assert h > 0

    def test_entropy_constant(self):
        f = np.ones(100)
        h = segment_entropy(f)
        assert h == pytest.approx(0.0)

    def test_surprise_curve(self):
        seg1 = np.random.default_rng(1).uniform(0, 1, 50)
        seg2 = np.ones(50) * 0.5
        sc = surprise_curve([seg1, seg2])
        assert len(sc) == 1
        assert sc[0] > 0

    def test_structural_score(self):
        segs = [np.random.default_rng(i).uniform(0, 1, 50) for i in range(5)]
        score = structural_surprise_score(segs)
        assert score >= 0
