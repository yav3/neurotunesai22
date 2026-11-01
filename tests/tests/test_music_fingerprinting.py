"""Tests for music_fingerprinting: chromaprint, landmarks, matching, LSH."""
import pytest
import numpy as np


@pytest.fixture
def synthetic_spectrogram():
    rng = np.random.RandomState(42)
    return rng.rand(128, 256)


@pytest.fixture
def spectrogram_with_peaks():
    S = np.zeros((64, 128))
    S[10, 20] = 5.0
    S[30, 50] = 4.0
    S[50, 80] = 3.0
    S[15, 100] = 6.0
    return S


class TestSpectralFingerprinter:
    def test_deterministic(self, synthetic_spectrogram):
        from models.music_fingerprinting.chromaprint import SpectralFingerprinter
        fp = SpectralFingerprinter()
        a = fp.fingerprint(synthetic_spectrogram)
        b = fp.fingerprint(synthetic_spectrogram)
        assert a == b

    def test_returns_bytes(self, synthetic_spectrogram):
        from models.music_fingerprinting.chromaprint import SpectralFingerprinter
        fp = SpectralFingerprinter()
        result = fp.fingerprint(synthetic_spectrogram)
        assert isinstance(result, bytes)
        assert len(result) == 32  # SHA-256

    def test_different_inputs_different_fp(self):
        from models.music_fingerprinting.chromaprint import SpectralFingerprinter
        fp = SpectralFingerprinter()
        a_spec = np.zeros((64, 100))
        a_spec[10, :] = 1.0
        b_spec = np.zeros((64, 100))
        b_spec[50, :] = 1.0
        a = fp.fingerprint(a_spec)
        b = fp.fingerprint(b_spec)
        assert a != b

    def test_empty_spectrogram(self):
        from models.music_fingerprinting.chromaprint import SpectralFingerprinter
        fp = SpectralFingerprinter()
        result = fp.fingerprint(np.zeros((64, 0)))
        assert result == b""

    def test_segments(self, synthetic_spectrogram):
        from models.music_fingerprinting.chromaprint import SpectralFingerprinter
        fp = SpectralFingerprinter()
        segs = fp.fingerprint_segments(synthetic_spectrogram, segment_size=64)
        assert len(segs) > 1
        assert all(isinstance(s, bytes) for s in segs)

    def test_sub_fingerprints(self, synthetic_spectrogram):
        from models.music_fingerprinting.chromaprint import SpectralFingerprinter
        fp = SpectralFingerprinter(window_size=8, hop_size=4)
        subs = fp.sub_fingerprints(synthetic_spectrogram)
        assert len(subs) > 0
        assert subs[0].offset == 0
        assert subs[1].offset == 4

    def test_sub_fingerprint_deterministic(self, synthetic_spectrogram):
        from models.music_fingerprinting.chromaprint import SpectralFingerprinter
        fp = SpectralFingerprinter()
        a = fp.sub_fingerprints(synthetic_spectrogram)
        b = fp.sub_fingerprints(synthetic_spectrogram)
        assert [s.hash_value for s in a] == [s.hash_value for s in b]


class TestGrayCode:
    def test_gray_code_zero(self):
        from models.music_fingerprinting.chromaprint import _gray_code
        assert _gray_code(0) == 0

    def test_gray_code_one(self):
        from models.music_fingerprinting.chromaprint import _gray_code
        assert _gray_code(1) == 1

    def test_gray_code_adjacent_differ_one_bit(self):
        from models.music_fingerprinting.chromaprint import _gray_code
        for n in range(100):
            diff = _gray_code(n) ^ _gray_code(n + 1)
            assert bin(diff).count("1") == 1


class TestPeakDetection:
    def test_finds_planted_peaks(self, spectrogram_with_peaks):
        from models.music_fingerprinting.landmarks import find_peaks
        peaks = find_peaks(spectrogram_with_peaks, threshold=1.0, neighborhood_size=5)
        peak_coords = {(p.time, p.freq) for p in peaks}
        assert (20, 10) in peak_coords
        assert (50, 30) in peak_coords

    def test_no_peaks_in_silence(self):
        from models.music_fingerprinting.landmarks import find_peaks
        S = np.zeros((64, 128))
        peaks = find_peaks(S, threshold=0.01)
        assert len(peaks) == 0

    def test_threshold_filters(self, spectrogram_with_peaks):
        from models.music_fingerprinting.landmarks import find_peaks
        low = find_peaks(spectrogram_with_peaks, threshold=1.0)
        high = find_peaks(spectrogram_with_peaks, threshold=5.5)
        assert len(low) > len(high)

    def test_peak_magnitudes_positive(self, spectrogram_with_peaks):
        from models.music_fingerprinting.landmarks import find_peaks
        peaks = find_peaks(spectrogram_with_peaks, threshold=1.0)
        assert all(p.magnitude > 0 for p in peaks)


class TestLandmarks:
    def test_form_pairs_respects_fan_out(self, spectrogram_with_peaks):
        from models.music_fingerprinting.landmarks import find_peaks, form_pairs
        peaks = find_peaks(spectrogram_with_peaks, threshold=1.0)
        landmarks = form_pairs(peaks, fan_out=2, time_delta_max=200)
        from collections import Counter
        anchor_counts = Counter(lm.anchor_time for lm in landmarks)
        assert all(c <= 2 for c in anchor_counts.values())

    def test_form_pairs_time_constraint(self, spectrogram_with_peaks):
        from models.music_fingerprinting.landmarks import find_peaks, form_pairs
        peaks = find_peaks(spectrogram_with_peaks, threshold=1.0)
        landmarks = form_pairs(peaks, fan_out=10, time_delta_max=10)
        assert all(lm.delta_time <= 10 for lm in landmarks)

    def test_hash_landmarks_produces_pairs(self, spectrogram_with_peaks):
        from models.music_fingerprinting.landmarks import find_peaks, form_pairs, hash_landmarks
        peaks = find_peaks(spectrogram_with_peaks, threshold=1.0)
        landmarks = form_pairs(peaks, fan_out=5)
        hashes = hash_landmarks(landmarks)
        assert all(isinstance(h, int) and isinstance(off, int) for h, off in hashes)

    def test_constellation_map(self, synthetic_spectrogram):
        from models.music_fingerprinting.landmarks import ConstellationMap
        cm = ConstellationMap(threshold=0.5, neighborhood_size=3)
        hashes = cm.extract(synthetic_spectrogram)
        assert len(hashes) > 0
        assert all(isinstance(h, int) and isinstance(off, int) for h, off in hashes)


class TestLandmarkIndex:
    def test_add_and_query(self, synthetic_spectrogram):
        from models.music_fingerprinting.landmarks import ConstellationMap, LandmarkIndex
        cm = ConstellationMap(threshold=0.5, neighborhood_size=3)
        hashes = cm.extract(synthetic_spectrogram)
        idx = LandmarkIndex()
        idx.add_track("track_1", hashes)
        assert idx.n_tracks == 1
        results = idx.query(hashes, min_matches=2)
        assert len(results) > 0
        assert results[0].track_id == "track_1"

    def test_self_match_highest(self, synthetic_spectrogram):
        from models.music_fingerprinting.landmarks import ConstellationMap, LandmarkIndex
        cm = ConstellationMap(threshold=0.5, neighborhood_size=3)
        rng = np.random.RandomState(99)
        other = rng.rand(128, 256)
        h1 = cm.extract(synthetic_spectrogram)
        h2 = cm.extract(other)
        idx = LandmarkIndex()
        idx.add_track("self", h1)
        idx.add_track("other", h2)
        results = idx.query(h1, min_matches=2)
        assert results[0].track_id == "self"

    def test_empty_query(self):
        from models.music_fingerprinting.landmarks import LandmarkIndex
        idx = LandmarkIndex()
        results = idx.query([], min_matches=1)
        assert results == []

    def test_min_matches_filter(self, synthetic_spectrogram):
        from models.music_fingerprinting.landmarks import ConstellationMap, LandmarkIndex
        cm = ConstellationMap(threshold=0.5, neighborhood_size=3)
        hashes = cm.extract(synthetic_spectrogram)
        idx = LandmarkIndex()
        idx.add_track("t1", hashes)
        high_min = idx.query(hashes[:1], min_matches=100)
        assert len(high_min) == 0


class TestAlignment:
    def test_perfect_alignment(self):
        from models.music_fingerprinting.matching import align_matches
        offsets = [5, 5, 5, 5, 5]
        result = align_matches(offsets)
        assert result.best_offset == 5
        assert result.confidence == 1.0
        assert result.n_aligned == 5

    def test_noisy_alignment(self):
        from models.music_fingerprinting.matching import align_matches
        offsets = [5, 5, 5, 6, 7, 5]
        result = align_matches(offsets)
        assert result.best_offset == 5
        assert result.confidence > 0.5

    def test_empty_alignment(self):
        from models.music_fingerprinting.matching import align_matches
        result = align_matches([])
        assert result.confidence == 0.0

    def test_score_match(self):
        from models.music_fingerprinting.matching import align_matches, score_match
        offsets = [10] * 50
        alignment = align_matches(offsets)
        s = score_match(alignment)
        assert 0 < s <= 1.0


class TestFingerprintMatcher:
    def test_match_self(self):
        from models.music_fingerprinting.matching import FingerprintMatcher
        query = [(100, 0), (200, 5), (300, 10), (400, 15)]
        db = {"track_a": [(100, 0), (200, 5), (300, 10), (400, 15)]}
        m = FingerprintMatcher(min_matches=2)
        results = m.match(query, db)
        assert len(results) > 0
        assert results[0].track_id == "track_a"

    def test_no_match_below_threshold(self):
        from models.music_fingerprinting.matching import FingerprintMatcher
        query = [(100, 0)]
        db = {"track_a": [(999, 0)]}
        m = FingerprintMatcher(min_matches=2)
        results = m.match(query, db)
        assert len(results) == 0

    def test_partial_match(self):
        from models.music_fingerprinting.matching import FingerprintMatcher
        query = [(i, i) for i in range(20)]
        db = {"full": [(i, i) for i in range(20)]}
        m = FingerprintMatcher(min_matches=2)
        results = m.match_partial(query, db, query_start=5, query_end=15)
        assert len(results) > 0


class TestMinHash:
    def test_identical_sets_similarity_1(self):
        from models.music_fingerprinting.hashing import MinHashFingerprint
        mh = MinHashFingerprint(n_hashes=256)
        s = {1, 2, 3, 4, 5}
        sig = mh.signature(s)
        assert mh.jaccard_estimate(sig, sig) == 1.0

    def test_disjoint_sets_low_similarity(self):
        from models.music_fingerprinting.hashing import MinHashFingerprint
        mh = MinHashFingerprint(n_hashes=256)
        a = mh.signature({1, 2, 3, 4, 5})
        b = mh.signature({100, 200, 300, 400, 500})
        assert mh.jaccard_estimate(a, b) < 0.2

    def test_approximates_exact(self):
        from models.music_fingerprinting.hashing import MinHashFingerprint
        mh = MinHashFingerprint(n_hashes=512)
        set_a = set(range(50))
        set_b = set(range(30, 80))
        exact = mh.jaccard_exact(set_a, set_b)
        sig_a = mh.signature(set_a)
        sig_b = mh.signature(set_b)
        approx = mh.jaccard_estimate(sig_a, sig_b)
        assert abs(approx - exact) < 0.15

    def test_empty_set(self):
        from models.music_fingerprinting.hashing import MinHashFingerprint
        mh = MinHashFingerprint()
        sig = mh.signature(set())
        assert len(sig) == mh.n_hashes


class TestLSHIndex:
    def test_insert_and_query(self):
        from models.music_fingerprinting.hashing import LSHIndex
        idx = LSHIndex(dim=16, n_tables=5, n_bits=4, seed=42)
        rng = np.random.RandomState(42)
        vec = rng.randn(16)
        idx.insert("item_0", vec)
        results = idx.query(vec)
        assert len(results) > 0
        assert results[0][0] == "item_0"
        assert results[0][1] > 0.99

    def test_similar_vectors_found(self):
        from models.music_fingerprinting.hashing import LSHIndex
        idx = LSHIndex(dim=32, n_tables=20, n_bits=8, seed=42)
        rng = np.random.RandomState(42)
        base = rng.randn(32)
        similar = base + 0.1 * rng.randn(32)
        different = rng.randn(32)
        idx.insert("base", base)
        idx.insert("different", different)
        results = idx.query(similar, threshold=0.5)
        found_ids = {r[0] for r in results}
        assert "base" in found_ids

    def test_n_items(self):
        from models.music_fingerprinting.hashing import LSHIndex
        idx = LSHIndex(dim=8)
        rng = np.random.RandomState(42)
        for i in range(10):
            idx.insert(f"item_{i}", rng.randn(8))
        assert idx.n_items == 10

    def test_wrong_dim_raises(self):
        from models.music_fingerprinting.hashing import LSHIndex
        idx = LSHIndex(dim=8)
        with pytest.raises(ValueError):
            idx.insert("x", np.zeros(16))

    def test_threshold_filters(self):
        from models.music_fingerprinting.hashing import LSHIndex
        idx = LSHIndex(dim=16, n_tables=10, n_bits=6, seed=42)
        rng = np.random.RandomState(42)
        for i in range(20):
            idx.insert(f"item_{i}", rng.randn(16))
        query = rng.randn(16)
        all_results = idx.query(query, threshold=-1.0)
        high_results = idx.query(query, threshold=0.8)
        assert len(all_results) >= len(high_results)


class TestRobustness:
    def test_noisy_spectrogram_similar_fp(self):
        from models.music_fingerprinting.chromaprint import SpectralFingerprinter
        rng = np.random.RandomState(42)
        original = rng.rand(64, 200) * 10
        noisy = original + rng.randn(64, 200) * 0.01
        fp = SpectralFingerprinter()
        fp_orig = fp.sub_fingerprints(original)
        fp_noisy = fp.sub_fingerprints(noisy)
        orig_hashes = {s.hash_value for s in fp_orig}
        noisy_hashes = {s.hash_value for s in fp_noisy}
        if orig_hashes and noisy_hashes:
            overlap = len(orig_hashes & noisy_hashes) / len(orig_hashes | noisy_hashes)
            assert overlap > 0.3
