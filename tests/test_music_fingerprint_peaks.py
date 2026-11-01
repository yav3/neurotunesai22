"""Tests for spectral peak fingerprinting module."""
import numpy as np
import pytest

from models.music_fingerprinting.spectral_peaks import (
    Peak, FingerprintHash, SpectralPeakExtractor,
)


class TestFindPeaks2D:
    def test_planted_peak(self):
        spec = np.zeros((32, 32))
        spec[16, 16] = 10.0
        peaks = SpectralPeakExtractor.find_peaks_2d(spec, neighborhood_size=3)
        found = any(p.freq_idx == 16 and p.time_idx == 16 for p in peaks)
        assert found

    def test_no_peaks_in_silence(self):
        spec = np.zeros((32, 32))
        peaks = SpectralPeakExtractor.find_peaks_2d(spec)
        assert len(peaks) == 0

    def test_small_spec(self):
        spec = np.ones((3, 3))
        peaks = SpectralPeakExtractor.find_peaks_2d(spec, neighborhood_size=5)
        assert len(peaks) == 0

    def test_multiple_peaks(self):
        spec = np.zeros((64, 64))
        spec[10, 10] = 5.0
        spec[30, 30] = 5.0
        spec[50, 50] = 5.0
        peaks = SpectralPeakExtractor.find_peaks_2d(spec, neighborhood_size=3)
        assert len(peaks) >= 3


class TestPeakPairs:
    def test_basic_pairing(self):
        peaks = [Peak(0, 10), Peak(5, 15), Peak(10, 20)]
        pairs = SpectralPeakExtractor.peak_pairs(
            peaks, target_zone_time=20, target_zone_freq=50
        )
        assert len(pairs) > 0

    def test_respects_time_zone(self):
        peaks = [Peak(0, 10), Peak(100, 15)]
        pairs = SpectralPeakExtractor.peak_pairs(
            peaks, target_zone_time=10, target_zone_freq=50
        )
        assert len(pairs) == 0

    def test_fan_out_limit(self):
        peaks = [Peak(0, 10)] + [Peak(i, 10 + i) for i in range(1, 20)]
        pairs = SpectralPeakExtractor.peak_pairs(
            peaks, target_zone_time=50, target_zone_freq=100, fan_out=3
        )
        anchor_0_pairs = [p for p in pairs if p[0].time_idx == 0]
        assert len(anchor_0_pairs) <= 3


class TestFingerprintHash:
    def test_deterministic(self):
        h1 = SpectralPeakExtractor.fingerprint_hash(100, 200, 5)
        h2 = SpectralPeakExtractor.fingerprint_hash(100, 200, 5)
        assert h1 == h2

    def test_different_inputs(self):
        h1 = SpectralPeakExtractor.fingerprint_hash(100, 200, 5)
        h2 = SpectralPeakExtractor.fingerprint_hash(100, 201, 5)
        assert h1 != h2

    def test_fits_32_bits(self):
        h = SpectralPeakExtractor.fingerprint_hash(1023, 1023, 1023)
        assert h < 2**32


class TestGenerateFingerprints:
    def test_tonal_signal(self):
        sr = 22050
        t = np.linspace(0, 2.0, sr * 2)
        signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        fps = SpectralPeakExtractor.generate_fingerprints(signal, sr)
        assert len(fps) > 0
        assert all(isinstance(fp, FingerprintHash) for fp in fps)

    def test_short_signal(self):
        fps = SpectralPeakExtractor.generate_fingerprints(np.zeros(100))
        assert fps == []


class TestMatchFingerprints:
    def test_self_match(self):
        sr = 22050
        t = np.linspace(0, 2.0, sr * 2)
        signal = np.sin(2 * np.pi * 440 * t)
        fps = SpectralPeakExtractor.generate_fingerprints(signal, sr)
        if not fps:
            pytest.skip("No fingerprints generated")
        db = {}
        song_id = 42
        for fp in fps:
            db.setdefault(fp.hash_val, []).append((song_id, fp.time_offset))
        results = SpectralPeakExtractor.match_fingerprints(fps, db)
        assert len(results) > 0
        assert results[0][0] == song_id
        assert results[0][2] > 0.5


class TestConstellationDensity:
    def test_empty_peaks(self):
        assert SpectralPeakExtractor.constellation_density([]) == 0.0

    def test_single_peak(self):
        d = SpectralPeakExtractor.constellation_density([Peak(5, 10)])
        assert d == 1.0

    def test_time_filter(self):
        peaks = [Peak(0, 5), Peak(10, 5), Peak(20, 5)]
        d = SpectralPeakExtractor.constellation_density(peaks, time_range=(0, 10))
        assert d > 0
