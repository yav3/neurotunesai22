"""Tests for expansion round 3: copyright, playlist, fingerprinting, audio dynamics."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_copyright.content_id import (
    AudioFingerprint, ContentIDResult, spectral_hash,
    extract_landmarks, match_landmarks, FingerprintDatabase,
)
from models.music_copyright.license_model import (
    LicenseType, LicenseInfo, LICENSE_PROPERTIES,
    check_compatibility, most_restrictive, attribution_text,
)
from models.music_playlist.diversity import (
    genre_diversity, tempo_diversity, artist_diversity,
    decade_spread, mood_coverage, compute_diversity, PlaylistDiversityReport,
)
from models.music_fingerprinting.robustness import (
    add_white_noise, time_stretch, pitch_shift_semitones,
    apply_lowpass, mp3_simulation, evaluate_robustness, DegradationResult,
)
from models.music_audio.dynamics import (
    compute_rms, compute_crest_factor, compute_dynamic_range,
    compute_loudness_envelope, detect_clipping,
    DynamicsCompressor, DynamicsLimiter,
)


class TestContentID:
    def test_spectral_hash_deterministic(self):
        spec = np.random.RandomState(42).randn(128, 100)
        h1 = spectral_hash(spec)
        h2 = spectral_hash(spec)
        assert h1 == h2
        assert len(h1) == 32

    def test_spectral_hash_different_for_different_input(self):
        rng = np.random.RandomState(42)
        h1 = spectral_hash(rng.randn(128, 100))
        h2 = spectral_hash(rng.randn(128, 100))
        assert h1 != h2

    def test_extract_landmarks_returns_triples(self):
        spec = np.random.RandomState(42).randn(64, 200)
        spec = np.abs(spec)
        lm = extract_landmarks(spec, n_peaks=10, neighborhood=3)
        assert isinstance(lm, list)
        for item in lm:
            assert len(item) == 3

    def test_extract_landmarks_empty_spec(self):
        lm = extract_landmarks(np.array([]))
        assert lm == []

    def test_match_landmarks_exact(self):
        lm = [(10, 20, 5), (30, 40, 3), (50, 60, 7)]
        db = {"song_a": lm}
        result = match_landmarks(lm, db, threshold=0.3)
        assert result.matched
        assert result.source_id == "song_a"
        assert result.similarity == 1.0

    def test_match_landmarks_empty_query(self):
        result = match_landmarks([], {"a": [(1, 2, 3)]})
        assert not result.matched

    def test_fingerprint_database(self):
        db = FingerprintDatabase()
        assert db.size == 0
        db.add("song_a", [(1, 2, 3), (4, 5, 6)])
        assert db.size == 1
        result = db.query([(1, 2, 3), (4, 5, 6)])
        assert result.matched
        assert db.remove("song_a")
        assert db.size == 0


class TestLicenseModel:
    def test_all_license_types_have_properties(self):
        for lt in LicenseType:
            assert lt in LICENSE_PROPERTIES

    def test_public_domain_allows_all(self):
        ok, _ = check_compatibility(LicenseType.PUBLIC_DOMAIN, "commercial")
        assert ok
        ok, _ = check_compatibility(LicenseType.PUBLIC_DOMAIN, "remix")
        assert ok

    def test_cc_by_nc_blocks_commercial(self):
        ok, msg = check_compatibility(LicenseType.CC_BY_NC, "commercial")
        assert not ok
        assert "commercial" in msg.lower()

    def test_cc_by_nd_blocks_derivative(self):
        ok, msg = check_compatibility(LicenseType.CC_BY_ND, "remix")
        assert not ok

    def test_most_restrictive_single(self):
        result = most_restrictive([LicenseType.CC_BY])
        assert result == LicenseType.CC_BY

    def test_most_restrictive_mixed(self):
        result = most_restrictive([LicenseType.CC_BY, LicenseType.CC_BY_NC_ND])
        assert result == LicenseType.CC_BY_NC_ND

    def test_most_restrictive_empty(self):
        assert most_restrictive([]) == LicenseType.PUBLIC_DOMAIN

    def test_attribution_text_cc_by(self):
        text = attribution_text(LicenseType.CC_BY, "Artist", "Song")
        assert "Artist" in text
        assert "Song" in text
        assert "CC BY" in text

    def test_attribution_text_public_domain(self):
        text = attribution_text(LicenseType.PUBLIC_DOMAIN, "Artist", "Song")
        assert text == ""


class TestPlaylistDiversity:
    def test_genre_diversity_uniform(self):
        genres = ["rock", "pop", "jazz", "classical"]
        d = genre_diversity(genres)
        assert d == 1.0

    def test_genre_diversity_single(self):
        genres = ["rock", "rock", "rock"]
        d = genre_diversity(genres)
        assert d == 0.0

    def test_genre_diversity_empty(self):
        assert genre_diversity([]) == 0.0

    def test_tempo_diversity(self):
        d = tempo_diversity([120.0, 120.0, 120.0])
        assert d == 0.0
        d2 = tempo_diversity([80.0, 120.0, 160.0])
        assert d2 > 0

    def test_artist_diversity(self):
        d = artist_diversity(["A", "B", "C"])
        assert d == 1.0
        d2 = artist_diversity(["A", "A", "A"])
        assert abs(d2 - 1 / 3) < 1e-10

    def test_decade_spread(self):
        d = decade_spread([1980, 1990, 2000, 2010, 2020])
        assert d == 1.0
        d2 = decade_spread([2020, 2021, 2022])
        assert d2 < 1.0

    def test_mood_coverage(self):
        d = mood_coverage(["happy", "sad", "energetic"])
        assert d == 0.3

    def test_compute_diversity(self):
        tracks = [
            {"genre": "rock", "tempo": 120, "artist": "A", "year": 2000, "mood": "happy"},
            {"genre": "jazz", "tempo": 90, "artist": "B", "year": 1980, "mood": "calm"},
        ]
        report = compute_diversity(tracks)
        assert isinstance(report, PlaylistDiversityReport)
        assert 0 <= report.overall_diversity <= 1


class TestFingerprintRobustness:
    def test_add_white_noise_shape(self):
        sig = np.sin(np.linspace(0, 2 * np.pi * 440, 22050))
        noisy = add_white_noise(sig, 20.0)
        assert noisy.shape == sig.shape
        assert not np.allclose(noisy, sig)

    def test_time_stretch_faster(self):
        sig = np.ones(1000)
        stretched = time_stretch(sig, 2.0)
        assert len(stretched) == 500

    def test_time_stretch_slower(self):
        sig = np.ones(1000)
        stretched = time_stretch(sig, 0.5)
        assert len(stretched) == 2000

    def test_time_stretch_identity(self):
        sig = np.random.randn(100)
        stretched = time_stretch(sig, 1.0)
        np.testing.assert_array_equal(stretched, sig)

    def test_pitch_shift_preserves_length(self):
        sig = np.sin(np.linspace(0, 2 * np.pi * 440, 22050))
        shifted = pitch_shift_semitones(sig, 2.0)
        assert len(shifted) == len(sig)

    def test_lowpass_removes_highs(self):
        sr = 22050
        t = np.linspace(0, 1, sr)
        sig = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 10000 * t)
        filtered = apply_lowpass(sig, cutoff_hz=500.0, sr=sr)
        spec_orig = np.abs(np.fft.rfft(sig))
        spec_filt = np.abs(np.fft.rfft(filtered))
        high_idx = int(5000 * len(t) / sr)
        assert np.mean(spec_filt[high_idx:]) < np.mean(spec_orig[high_idx:])

    def test_mp3_simulation(self):
        sig = np.random.randn(4096)
        compressed = mp3_simulation(sig, 0.5)
        assert compressed.shape == sig.shape

    def test_evaluate_robustness_runs(self):
        sig = np.sin(np.linspace(0, 2 * np.pi * 440, 4096))
        fp_fn = lambda s: np.mean(np.abs(np.fft.rfft(s))[:20])
        match_fn = lambda a, b: 1.0 - min(abs(a - b) / max(abs(a), 1e-10), 1.0)
        results = evaluate_robustness(sig, fp_fn, match_fn, threshold=0.5)
        assert len(results) == 11
        assert all(isinstance(r, DegradationResult) for r in results)


class TestAudioDynamics:
    def test_compute_rms_sine(self):
        t = np.linspace(0, 1, 22050)
        sig = np.sin(2 * np.pi * 440 * t)
        rms = compute_rms(sig, frame_length=2048, hop_length=512)
        assert rms.shape[0] > 0
        assert np.mean(rms) > 0.5

    def test_compute_crest_factor_sine(self):
        t = np.linspace(0, 1, 22050)
        sig = np.sin(2 * np.pi * 440 * t)
        cf = compute_crest_factor(sig)
        assert abs(cf - 3.01) < 0.5

    def test_compute_crest_factor_silence(self):
        assert compute_crest_factor(np.zeros(100)) == 0.0

    def test_compute_dynamic_range(self):
        sig = np.concatenate([np.ones(1000) * 0.1, np.ones(1000) * 0.9])
        dr = compute_dynamic_range(sig)
        assert dr > 0

    def test_loudness_envelope_shape(self):
        sig = np.random.randn(22050)
        env = compute_loudness_envelope(sig, sr=22050, frame_ms=50.0)
        assert len(env) > 0

    def test_detect_clipping(self):
        sig = np.array([0.5, 0.8, 1.0, 1.0, 0.7, 0.3, 1.0, 0.2])
        regions = detect_clipping(sig, threshold=0.99)
        assert len(regions) >= 1

    def test_detect_clipping_clean(self):
        sig = np.array([0.1, 0.2, 0.3, 0.4])
        regions = detect_clipping(sig)
        assert len(regions) == 0

    def test_compressor_reduces_peaks(self):
        sig = np.zeros(4410)
        sig[2000:2100] = 0.9
        sig[0:100] = 0.1
        comp = DynamicsCompressor(threshold_db=-10.0, ratio=4.0)
        out = comp.process(sig, sr=44100)
        assert np.max(np.abs(out)) <= np.max(np.abs(sig)) + 0.01

    def test_limiter_is_compressor(self):
        lim = DynamicsLimiter(threshold_db=-3.0)
        assert lim.ratio == 100.0
        assert lim.knee_db == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
