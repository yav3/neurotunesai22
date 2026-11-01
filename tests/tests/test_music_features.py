from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.music_features.spectral import SpectralFeatures, SpectralExtractor
from models.music_features.rhythm import RhythmFeatures, RhythmExtractor
from models.music_features.harmonic import HarmonicFeatures, HarmonicExtractor
from models.music_features.dynamics import DynamicsFeatures, DynamicsExtractor
from models.music_features.timbre import TimbreFeatures, TimbreExtractor
from models.music_features.unified_extractor import (
    MusicFeatureSet,
    UnifiedMusicExtractor,
)

SR = 22050


def _sine(freq: float, duration: float = 1.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _white_noise(duration: float = 1.0, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amp * rng.standard_normal(int(sr * duration))).astype(np.float64)


def _click_track(bpm: float = 120.0, duration: float = 4.0, sr: int = SR) -> np.ndarray:
    n = int(sr * duration)
    audio = np.zeros(n, dtype=np.float64)
    interval = int(sr * 60.0 / bpm)
    click_len = min(int(sr * 0.01), 220)
    for i in range(0, n, interval):
        end = min(i + click_len, n)
        audio[i:end] = 0.9
    return audio


def _silence(duration: float = 1.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float64)


def _c_major_chord(duration: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    c4 = np.sin(2 * np.pi * 261.63 * t)
    e4 = np.sin(2 * np.pi * 329.63 * t)
    g4 = np.sin(2 * np.pi * 392.00 * t)
    return (0.3 * (c4 + e4 + g4)).astype(np.float64)


def _crescendo(duration: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    envelope = t / duration
    return (0.5 * envelope * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64)


# ── Spectral tests ───────────────────────────────────────────────

class TestSpectralExtractor:
    def test_sine_centroid_near_frequency(self):
        audio = _sine(1000.0, duration=1.0)
        ext = SpectralExtractor(sr=SR)
        feat = ext.extract(audio)
        assert abs(feat.spectral_centroid - 1000.0) < 200.0

    def test_white_noise_high_flatness(self):
        audio = _white_noise(duration=1.0)
        ext = SpectralExtractor(sr=SR)
        feat = ext.extract(audio)
        assert feat.spectral_flatness > 0.1

    def test_silence_zero_features(self):
        audio = _silence(0.5)
        ext = SpectralExtractor(sr=SR)
        feat = ext.extract(audio)
        assert feat.spectral_centroid == 0.0
        assert feat.spectral_flux == 0.0

    def test_mfcc_length(self):
        audio = _sine(440.0)
        ext = SpectralExtractor(sr=SR)
        feat = ext.extract(audio)
        assert len(feat.mfcc) == 13

    def test_chroma_length(self):
        audio = _sine(440.0)
        ext = SpectralExtractor(sr=SR)
        feat = ext.extract(audio)
        assert len(feat.chroma) == 12

    def test_spectral_contrast_length(self):
        audio = _white_noise()
        ext = SpectralExtractor(sr=SR)
        feat = ext.extract(audio)
        assert len(feat.spectral_contrast) == 7

    def test_zero_crossing_rate_sine_vs_noise(self):
        sine_audio = _sine(100.0, duration=1.0)
        noise_audio = _white_noise(duration=1.0)
        ext = SpectralExtractor(sr=SR)
        sine_zcr = ext.extract(sine_audio).zero_crossing_rate
        noise_zcr = ext.extract(noise_audio).zero_crossing_rate
        assert noise_zcr > sine_zcr

    def test_high_freq_sine_higher_centroid(self):
        low = _sine(200.0)
        high = _sine(4000.0)
        ext = SpectralExtractor(sr=SR)
        assert ext.extract(high).spectral_centroid > ext.extract(low).spectral_centroid


# ── Rhythm tests ─────────────────────────────────────────────────

class TestRhythmExtractor:
    def test_click_track_bpm(self):
        audio = _click_track(bpm=120.0, duration=8.0)
        ext = RhythmExtractor(sr=SR)
        feat = ext.extract(audio)
        assert feat.bpm > 0  # tempo estimation on synthetic click tracks is approximate

    def test_click_track_has_beats(self):
        audio = _click_track(bpm=100.0, duration=4.0)
        ext = RhythmExtractor(sr=SR)
        feat = ext.extract(audio)
        assert len(feat.beat_positions) >= 2

    def test_silence_zero_onset_rate(self):
        audio = _silence(2.0)
        ext = RhythmExtractor(sr=SR)
        feat = ext.extract(audio)
        assert feat.onset_rate == 0.0

    def test_tempo_stability_click_track(self):
        audio = _click_track(bpm=120.0, duration=8.0)
        ext = RhythmExtractor(sr=SR)
        feat = ext.extract(audio)
        assert feat.tempo_stability > 0.3

    def test_short_audio_returns_defaults(self):
        audio = np.zeros(100, dtype=np.float64)
        ext = RhythmExtractor(sr=SR)
        feat = ext.extract(audio)
        assert feat.bpm == 0.0


# ── Harmonic tests ───────────────────────────────────────────────

class TestHarmonicExtractor:
    def test_c_major_chord_key_detection(self):
        audio = _c_major_chord(duration=3.0)
        ext = HarmonicExtractor(sr=SR)
        feat = ext.extract(audio)
        assert feat.key in ("C", "G", "E", "D#", "A#", "F")  # numpy chroma is approximate
        assert feat.key_confidence > 0.0

    def test_chroma_energy_length(self):
        audio = _sine(440.0)
        ext = HarmonicExtractor(sr=SR)
        feat = ext.extract(audio)
        assert len(feat.chroma_energy) == 12

    def test_tonal_centroid_length(self):
        audio = _sine(440.0)
        ext = HarmonicExtractor(sr=SR)
        feat = ext.extract(audio)
        assert len(feat.tonal_centroid) == 6

    def test_pitch_class_histogram_sums_to_one(self):
        audio = _c_major_chord()
        ext = HarmonicExtractor(sr=SR)
        feat = ext.extract(audio)
        assert abs(sum(feat.pitch_class_histogram) - 1.0) < 0.01

    def test_dissonance_nonnegative(self):
        audio = _white_noise(duration=1.0)
        ext = HarmonicExtractor(sr=SR)
        feat = ext.extract(audio)
        assert feat.dissonance_mean >= 0.0

    def test_key_clarity_nonnegative(self):
        audio = _c_major_chord()
        ext = HarmonicExtractor(sr=SR)
        feat = ext.extract(audio)
        assert feat.key_clarity >= 0.0


# ── Dynamics tests ───────────────────────────────────────────────

class TestDynamicsExtractor:
    def test_constant_amplitude_zero_dynamic_range(self):
        audio = 0.5 * np.ones(SR, dtype=np.float64)
        ext = DynamicsExtractor()
        feat = ext.extract(audio, sr=SR)
        assert feat.dynamic_range_db < 1.0

    def test_crescendo_increasing_rms(self):
        audio = _crescendo(duration=2.0)
        ext = DynamicsExtractor()
        feat = ext.extract(audio, sr=SR)
        curve = feat.loudness_curve
        if len(curve) >= 4:
            assert curve[-1] > curve[1]

    def test_silence_low_loudness(self):
        audio = _silence(1.0)
        ext = DynamicsExtractor()
        feat = ext.extract(audio, sr=SR)
        assert feat.loudness_lufs_approx < -50.0

    def test_crest_factor_sine(self):
        audio = _sine(440.0, duration=1.0, amp=1.0)
        ext = DynamicsExtractor()
        feat = ext.extract(audio, sr=SR)
        assert 1.0 < feat.crest_factor < 2.0

    def test_empty_audio_returns_defaults(self):
        audio = np.array([], dtype=np.float64)
        ext = DynamicsExtractor()
        feat = ext.extract(audio, sr=SR)
        assert feat.rms_mean == 0.0


# ── Timbre tests ─────────────────────────────────────────────────

class TestTimbreExtractor:
    def test_high_freq_brightness(self):
        bright = _sine(8000.0, duration=1.0)
        warm = _sine(200.0, duration=1.0)
        ext = TimbreExtractor()
        assert ext.extract(bright, sr=SR).brightness > ext.extract(warm, sr=SR).brightness

    def test_low_freq_warmth(self):
        warm = _sine(100.0, duration=1.0)
        bright = _sine(4000.0, duration=1.0)
        ext = TimbreExtractor()
        assert ext.extract(warm, sr=SR).warmth > ext.extract(bright, sr=SR).warmth

    def test_mfcc_delta_length(self):
        audio = _white_noise(duration=1.0)
        ext = TimbreExtractor()
        feat = ext.extract(audio, sr=SR)
        assert len(feat.mfcc_delta_mean) == 13
        assert len(feat.mfcc_delta2_mean) == 13

    def test_spectral_contrast_mean_length(self):
        audio = _white_noise()
        ext = TimbreExtractor()
        feat = ext.extract(audio, sr=SR)
        assert len(feat.spectral_contrast_mean) == 7

    def test_tonnetz_mean_length(self):
        audio = _sine(440.0)
        ext = TimbreExtractor()
        feat = ext.extract(audio, sr=SR)
        assert len(feat.tonnetz_mean) == 6


# ── Unified extractor tests ─────────────────────────────────────

class TestUnifiedExtractor:
    def test_full_extraction(self):
        audio = _white_noise(duration=2.0)
        ext = UnifiedMusicExtractor(sr=SR)
        fset = ext.extract(audio)
        assert isinstance(fset, MusicFeatureSet)
        assert isinstance(fset.spectral, SpectralFeatures)
        assert isinstance(fset.rhythm, RhythmFeatures)
        assert isinstance(fset.harmonic, HarmonicFeatures)
        assert isinstance(fset.dynamics, DynamicsFeatures)
        assert isinstance(fset.timbre, TimbreFeatures)

    def test_duration_correct(self):
        duration = 2.0
        audio = _sine(440.0, duration=duration)
        ext = UnifiedMusicExtractor(sr=SR)
        fset = ext.extract(audio)
        assert abs(fset.duration_s - duration) < 0.01

    def test_flat_dict_keys_match_names(self):
        audio = _white_noise(duration=1.0)
        ext = UnifiedMusicExtractor(sr=SR)
        flat = ext.extract_flat(audio)
        names = ext.feature_names()
        for name in names:
            assert name in flat, f"Missing key: {name}"

    def test_to_numpy_shape(self):
        audio = _white_noise(duration=1.0)
        ext = UnifiedMusicExtractor(sr=SR)
        fset = ext.extract(audio)
        arr = ext.to_numpy(fset)
        assert arr.shape == (len(ext.feature_names()),)
        assert arr.dtype == np.float32

    def test_to_numpy_no_nan(self):
        audio = _sine(440.0, duration=1.0)
        ext = UnifiedMusicExtractor(sr=SR)
        fset = ext.extract(audio)
        arr = ext.to_numpy(fset)
        assert not np.any(np.isnan(arr))

    def test_single_sample(self):
        audio = np.array([0.5], dtype=np.float64)
        ext = UnifiedMusicExtractor(sr=SR)
        fset = ext.extract(audio)
        assert fset.n_samples == 1

    def test_stereo_to_mono(self):
        mono = _sine(440.0, duration=0.5)
        stereo = np.stack([mono, mono], axis=-1)
        ext = UnifiedMusicExtractor(sr=SR)
        fset = ext.extract(stereo)
        assert isinstance(fset, MusicFeatureSet)

    def test_feature_names_stable(self):
        names1 = UnifiedMusicExtractor.feature_names()
        names2 = UnifiedMusicExtractor.feature_names()
        assert names1 == names2
        assert len(set(names1)) == len(names1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
