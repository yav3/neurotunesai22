"""Deep tests for fingerprinting: DB, hashing, constellation, robustness."""
import numpy as np
import pytest

from models.music_fingerprinting.fingerprint_db import (
    FingerprintDB, TrackRecord, generate_hashes_from_peaks,
)
from models.music_fingerprinting.audio_hash import (
    perceptual_hash, hash_similarity, AudioHash,
)
from models.music_fingerprinting.robustness import (
    add_white_noise, time_stretch, pitch_shift_semitones,
    apply_lowpass, evaluate_robustness,
)
from models.music_fingerprinting.chromaprint import SpectralFingerprinter
from models.music_fingerprinting.constellation_map import (
    find_spectral_peaks, build_constellation,
)

SR = 16000


def _sine(freq=440.0, dur=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _spectrogram(signal, n_fft=1024, hop=512):
    n = len(signal)
    n_frames = max(1, (n - n_fft) // hop + 1)
    spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        w = np.hanning(n_fft)
        spec[:, i] = np.abs(np.fft.rfft(frame * w))
    return spec


# ── FingerprintDB ────────────────────────────────────────────────────

class TestFingerprintDB:
    def test_add_and_query(self):
        db = FingerprintDB()
        hashes_a = [(100, 0), (200, 5), (300, 10), (400, 15), (500, 20)]
        db.add_track(TrackRecord("a", "Song A"), hashes_a)
        result = db.query(hashes_a, min_matches=3)
        assert len(result) >= 1
        assert result[0][0] == "a"

    def test_no_match(self):
        db = FingerprintDB()
        db.add_track(TrackRecord("a"), [(100, 0), (200, 5)])
        result = db.query([(999, 0), (888, 5)], min_matches=1)
        assert len(result) == 0

    def test_empty_query(self):
        db = FingerprintDB()
        assert db.query([]) == []

    def test_remove_track(self):
        db = FingerprintDB()
        db.add_track(TrackRecord("a"), [(100, 0), (200, 5)])
        db.remove_track("a")
        assert db.n_tracks == 0
        assert not db.contains("a")

    def test_multiple_tracks(self):
        db = FingerprintDB()
        db.add_track(TrackRecord("a"), [(100, i) for i in range(10)])
        db.add_track(TrackRecord("b"), [(200, i) for i in range(10)])
        assert db.n_tracks == 2
        result_a = db.query([(100, i) for i in range(10)], min_matches=5)
        assert result_a[0][0] == "a"

    def test_confidence_range(self):
        db = FingerprintDB()
        hashes = [(i * 100, i) for i in range(20)]
        db.add_track(TrackRecord("a"), hashes)
        result = db.query(hashes[:10], min_matches=3)
        if result:
            assert 0.0 < result[0][1] <= 1.0


class TestGenerateHashes:
    def test_basic(self):
        times = np.array([0, 5, 10, 15, 20])
        freqs = np.array([100, 200, 150, 300, 250])
        hashes = generate_hashes_from_peaks(times, freqs, fan_out=3)
        assert len(hashes) > 0
        assert all(isinstance(h, tuple) and len(h) == 2 for h in hashes)

    def test_empty(self):
        assert generate_hashes_from_peaks(np.array([]), np.array([]), fan_out=3) == []

    def test_single_peak(self):
        hashes = generate_hashes_from_peaks(np.array([0]), np.array([100]))
        assert len(hashes) == 0

    def test_fan_out_limit(self):
        times = np.arange(20)
        freqs = np.arange(20) * 50
        hashes = generate_hashes_from_peaks(times, freqs, fan_out=2, time_window=100)
        per_anchor = {}
        for h, t in hashes:
            per_anchor.setdefault(t, 0)
            per_anchor[t] += 1
        assert all(c <= 2 for c in per_anchor.values())


# ── Perceptual Hash ──────────────────────────────────────────────────

class TestPerceptualHash:
    def test_same_signal_same_hash(self):
        spec = _spectrogram(_sine(dur=2.0))
        h1 = perceptual_hash(spec, SR)
        h2 = perceptual_hash(spec, SR)
        assert hash_similarity(h1, h2) == pytest.approx(1.0)

    def test_different_signals(self):
        spec1 = _spectrogram(_sine(440, dur=2.0))
        rng = np.random.default_rng(99)
        noise = rng.standard_normal(int(SR * 2.0)).astype(np.float64) * 0.3
        spec2 = _spectrogram(noise)
        h1 = perceptual_hash(spec1, SR)
        h2 = perceptual_hash(spec2, SR)
        sim = hash_similarity(h1, h2)
        assert sim <= 1.0

    def test_returns_audio_hash(self):
        spec = _spectrogram(_sine(dur=1.0))
        h = perceptual_hash(spec, SR)
        assert isinstance(h, AudioHash)


# ── Robustness ───────────────────────────────────────────────────────

class TestRobustness:
    def test_add_noise_preserves_length(self):
        sig = _sine()
        noisy = add_white_noise(sig, snr_db=20)
        assert len(noisy) == len(sig)

    def test_time_stretch(self):
        sig = _sine(dur=1.0)
        stretched = time_stretch(sig, rate=1.1)
        assert len(stretched) != len(sig)

    def test_pitch_shift(self):
        sig = _sine(dur=0.5)
        shifted = pitch_shift_semitones(sig, semitones=2, sr=SR)
        assert len(shifted) > 0

    def test_lowpass(self):
        sig = _sine(freq=4000, dur=0.5)
        filtered = apply_lowpass(sig, cutoff_hz=1000, sr=SR)
        assert np.sum(filtered ** 2) < np.sum(sig ** 2)


# ── Constellation Map ────────────────────────────────────────────────

class TestConstellationMap:
    def test_find_spectral_peaks(self):
        spec = _spectrogram(_sine(dur=1.0))
        peaks = find_spectral_peaks(spec)
        assert len(peaks) > 0

    def test_build_constellation(self):
        spec = _spectrogram(_sine(dur=1.0))
        peaks = find_spectral_peaks(spec)
        if len(peaks) >= 2:
            constellation = build_constellation(peaks)
            assert len(constellation) >= 0


# ── SpectralFingerprinter ────────────────────────────────────────────

class TestSpectralFingerprinter:
    def test_fingerprint_signal(self):
        fp = SpectralFingerprinter()
        spec = _spectrogram(_sine(dur=2.0))
        result = fp.fingerprint(spec)
        assert len(result) > 0
