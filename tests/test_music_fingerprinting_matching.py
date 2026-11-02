"""Tests for robust fingerprinting and matching."""
import numpy as np
import pytest
from models.music_fingerprinting.robust_matching import (
    SpectralPeak, find_spectral_peaks, generate_fingerprints,
    hash_fingerprint, FingerprintDB, time_offset_histogram, match_score,
)


def test_find_peaks_threshold():
    spec = np.zeros((32, 16))
    spec[10, 5] = 10.0
    spec[20, 10] = 8.0
    peaks = find_spectral_peaks(spec, threshold_db=-20, min_freq_distance=2, min_time_distance=2)
    assert len(peaks) >= 2


def test_find_peaks_empty():
    spec = np.zeros((16, 8))
    peaks = find_spectral_peaks(spec, threshold_db=0)
    assert len(peaks) == 0


def test_generate_fingerprints_fanout():
    peaks = [SpectralPeak(t, f * 10, -10.0) for t in range(10) for f in range(3)]
    fps = generate_fingerprints(peaks, fan_out=2, max_delta_t=5)
    assert len(fps) > 0
    for f1, f2, dt, t_anc in fps:
        assert dt > 0
        assert dt <= 5


def test_hash_deterministic():
    h1 = hash_fingerprint(100, 200, 10)
    h2 = hash_fingerprint(100, 200, 10)
    assert h1 == h2


def test_hash_differs():
    h1 = hash_fingerprint(100, 200, 10)
    h2 = hash_fingerprint(100, 201, 10)
    assert h1 != h2


def test_db_insert_query():
    db = FingerprintDB()
    fps = [(10, 20, 5, 0), (15, 25, 3, 2), (20, 30, 4, 5)]
    db.insert(0, fps)
    results = db.query(fps, min_matches=2)
    assert len(results) >= 1
    assert results[0][0] == 0


def test_db_no_match():
    db = FingerprintDB()
    db.insert(0, [(10, 20, 5, 0)])
    results = db.query([(100, 200, 50, 0)], min_matches=1)
    assert len(results) == 0


def test_db_multiple_tracks():
    db = FingerprintDB()
    fps_a = [(10, 20, 5, i) for i in range(10)]
    fps_b = [(50, 60, 3, i) for i in range(10)]
    db.insert(0, fps_a)
    db.insert(1, fps_b)
    results = db.query(fps_a, min_matches=3)
    assert results[0][0] == 0


def test_offset_histogram():
    matches = [(10, 5), (15, 10), (20, 15)]
    hist = time_offset_histogram(matches)
    assert hist[5] == 3


def test_match_score_perfect():
    hist = {5: 100}
    assert match_score(hist, 100) == 1.0


def test_match_score_empty():
    assert match_score({}, 10) == 0.0


def test_match_score_partial():
    hist = {5: 30, 10: 5}
    s = match_score(hist, 100)
    assert abs(s - 0.3) < 1e-6


def test_peak_ordering():
    spec = np.zeros((16, 16))
    spec[5, 3] = 100.0
    spec[5, 10] = 90.0
    peaks = find_spectral_peaks(spec, threshold_db=-20, min_freq_distance=1, min_time_distance=1)
    times = [p.time_bin for p in peaks]
    assert 3 in times and 10 in times
