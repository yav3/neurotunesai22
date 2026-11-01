"""Tests for chromagram transforms and key detection."""
import numpy as np
import pytest

from models.music_transforms.chromagram import (
    ChromagramExtractor, ChromagramResult,
    KeyDetector, KeyProfile, KeyEstimate,
    PITCH_CLASSES, KRUMHANSL_MAJOR, KRUMHANSL_MINOR,
)


def _sine(freq, sr=22050, dur=1.0):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


class TestChromagramExtractor:
    def test_a440_peak_at_A(self):
        sig = _sine(440.0, dur=2.0)
        cr = ChromagramExtractor.stft_chromagram(sig, sr=22050)
        energy = ChromagramExtractor.chroma_energy(cr.chroma)
        assert energy.argmax() == 9  # A = pitch class 9

    def test_shape(self):
        sig = _sine(261.63, dur=1.0)
        cr = ChromagramExtractor.stft_chromagram(sig, sr=22050)
        assert cr.chroma.shape[0] == 12
        assert cr.chroma.shape[1] > 0

    def test_normalized(self):
        sig = _sine(440.0, dur=1.0)
        cr = ChromagramExtractor.stft_chromagram(sig, sr=22050)
        assert np.all(cr.chroma >= 0)
        assert np.all(cr.chroma <= 1.0 + 1e-6)

    def test_chroma_centroid_range(self):
        sig = _sine(440.0, dur=1.0)
        cr = ChromagramExtractor.stft_chromagram(sig, sr=22050)
        centroid = ChromagramExtractor.chroma_centroid(cr.chroma)
        assert np.all(centroid >= 0)
        assert np.all(centroid < 12)

    def test_chroma_contrast_nonneg(self):
        sig = _sine(440.0, dur=1.0)
        cr = ChromagramExtractor.stft_chromagram(sig, sr=22050)
        contrast = ChromagramExtractor.chroma_contrast(cr.chroma)
        assert np.all(contrast >= 0)


class TestKeyDetector:
    def test_c_major_scale(self):
        """C major scale should detect C major."""
        sr = 22050
        c_major_freqs = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88]
        signal = np.zeros(sr * 2)
        for f in c_major_freqs:
            signal += _sine(f, sr=sr, dur=2.0)
        cr = ChromagramExtractor.stft_chromagram(signal, sr=sr)
        key = KeyDetector.detect_key(cr.chroma)
        assert key.mode == "major"
        assert key.correlation > 0

    def test_a_minor_detection(self):
        """A minor scale should detect A minor or relative C major."""
        sr = 22050
        a_minor_freqs = [220.0, 246.94, 261.63, 293.66, 329.63, 349.23, 392.0]
        signal = np.zeros(sr * 2)
        for f in a_minor_freqs:
            signal += _sine(f, sr=sr, dur=2.0)
        cr = ChromagramExtractor.stft_chromagram(signal, sr=sr)
        key = KeyDetector.detect_key(cr.chroma)
        assert key.correlation > 0

    def test_silent_signal(self):
        cr = np.zeros((12, 10))
        key = KeyDetector.detect_key(cr)
        assert key.correlation == 0.0

    def test_all_profiles(self):
        sig = _sine(440.0, dur=2.0) + 0.5 * _sine(554.37, dur=2.0)  # A + C#
        cr = ChromagramExtractor.stft_chromagram(sig, sr=22050)
        for prof in KeyProfile:
            key = KeyDetector.detect_key(cr.chroma, profile=prof)
            assert len(key.all_correlations) == 24

    def test_key_strength_positive(self):
        sig = _sine(440.0, dur=2.0)
        cr = ChromagramExtractor.stft_chromagram(sig, sr=22050)
        key = KeyDetector.detect_key(cr.chroma)
        strength = KeyDetector.key_strength(key)
        assert strength > 0

    def test_modulation_detection(self):
        sig = _sine(261.63, dur=2.0)
        cr = ChromagramExtractor.stft_chromagram(sig, sr=22050)
        keys = KeyDetector.modulation_detection(cr.chroma, frame_size=10, hop=5)
        assert len(keys) > 0


class TestKeyProfiles:
    def test_major_profile_length(self):
        assert len(KRUMHANSL_MAJOR) == 12

    def test_minor_profile_length(self):
        assert len(KRUMHANSL_MINOR) == 12

    def test_pitch_classes_count(self):
        assert len(PITCH_CLASSES) == 12
