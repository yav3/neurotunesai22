"""Tests for query-by-humming module."""
import numpy as np
import pytest

from models.music_fingerprinting.query_by_humming import (
    MelodyContour, MatchResult,
    extract_pitch_contour, pitch_to_midi_cents, normalize_contour,
    dtw_distance, subsequence_dtw,
    build_melody_database, search_melody_database, contour_similarity,
)


SR = 16000


def _sine(freq, dur=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


class TestExtractPitchContour:
    def test_detects_pitch(self):
        sig = _sine(440, 0.5)
        contour = extract_pitch_contour(sig, SR)
        assert isinstance(contour, MelodyContour)
        voiced = contour.pitches_hz[contour.pitches_hz > 0]
        if len(voiced) > 0:
            assert abs(np.median(voiced) - 440) < 50

    def test_silence(self):
        sig = np.zeros(SR // 2)
        contour = extract_pitch_contour(sig, SR)
        assert np.all(contour.pitches_hz == 0)

    def test_hop_size(self):
        sig = _sine(300, 0.5)
        c1 = extract_pitch_contour(sig, SR, hop_size=256)
        c2 = extract_pitch_contour(sig, SR, hop_size=512)
        assert len(c1.pitches_hz) > len(c2.pitches_hz)


class TestPitchToMidiCents:
    def test_a4(self):
        result = pitch_to_midi_cents(np.array([440.0]))
        assert abs(result[0] - 69.0) < 0.01

    def test_zero(self):
        result = pitch_to_midi_cents(np.array([0.0]))
        assert result[0] == 0.0

    def test_octave(self):
        result = pitch_to_midi_cents(np.array([220.0, 440.0]))
        assert abs(result[1] - result[0] - 12.0) < 0.01


class TestNormalizeContour:
    def test_relative(self):
        c = np.array([60.0, 62.0, 64.0, 65.0])
        n = normalize_contour(c, "relative")
        assert n[0] == 0
        assert abs(n[1] - 2.0) < 0.01

    def test_mean_centered(self):
        c = np.array([60.0, 62.0, 64.0])
        n = normalize_contour(c, "mean_centered")
        assert abs(np.mean(n[n != 0])) < 0.01

    def test_z_score(self):
        c = np.array([60.0, 62.0, 64.0, 66.0])
        n = normalize_contour(c, "z_score")
        voiced = n[n != 0]
        assert abs(np.mean(voiced)) < 0.01
        assert abs(np.std(voiced) - 1.0) < 0.01

    def test_empty(self):
        n = normalize_contour(np.array([0.0, 0.0]), "relative")
        assert np.all(n == 0)


class TestDTWDistance:
    def test_identical(self):
        c = np.array([60, 62, 64, 65, 67])
        d = dtw_distance(c, c)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_different(self):
        a = np.array([60, 62, 64, 65, 67])
        b = np.array([72, 74, 76, 77, 79])
        d = dtw_distance(a, b)
        assert d > 0

    def test_empty(self):
        d = dtw_distance(np.array([0, 0]), np.array([60, 62]))
        assert d == float('inf')


class TestSubsequenceDTW:
    def test_finds_match(self):
        ref = np.array([60, 62, 64, 65, 67, 69, 71, 72])
        query = np.array([64, 65, 67])
        dist, offset = subsequence_dtw(query, ref)
        assert dist < 1.0
        assert offset >= 0

    def test_exact_match(self):
        ref = np.array([60, 62, 64, 65, 67])
        query = np.array([60, 62, 64, 65, 67])
        dist, offset = subsequence_dtw(query, ref)
        assert dist == pytest.approx(0.0, abs=1e-6)


class TestBuildMelodyDatabase:
    def test_build(self):
        melodies = [
            np.array([440, 494, 523, 587]),
            np.array([262, 294, 330, 349]),
        ]
        db = build_melody_database(melodies, normalize=True)
        assert len(db) == 2
        assert all(isinstance(m, np.ndarray) for m in db)


class TestSearchMelodyDatabase:
    def test_search(self):
        melodies = [
            np.array([440, 494, 523, 587, 659]),
            np.array([262, 294, 330, 349, 392]),
            np.array([523, 587, 659, 698, 784]),
        ]
        db = build_melody_database(melodies)
        query = MelodyContour(
            pitches_hz=np.array([440, 494, 523, 587, 659]),
            confidences=np.ones(5),
            hop_s=0.01,
            sr=SR,
        )
        results = search_melody_database(query, db, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, MatchResult) for r in results)
        assert results[0].distance <= results[-1].distance


class TestContourSimilarity:
    def test_identical_cosine(self):
        a = np.array([440, 494, 523, 587])
        s = contour_similarity(a, a, "cosine")
        assert s > 0.9

    def test_transposed_cosine(self):
        a = np.array([440, 494, 523, 587])
        b = np.array([220, 247, 262, 294])
        s = contour_similarity(a, b, "cosine")
        assert s > 0.5

    def test_correlation(self):
        a = np.array([440, 494, 523, 587])
        b = np.array([440, 494, 523, 587])
        s = contour_similarity(a, b, "correlation")
        assert s > 0.9

    def test_dtw_method(self):
        a = np.array([440, 494, 523])
        b = np.array([440, 500, 530])
        s = contour_similarity(a, b, "dtw")
        assert 0 < s <= 1.0
