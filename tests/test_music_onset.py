"""Tests for music_audio.onset — onset detection functions, ODF methods, fusion, features."""
import numpy as np
import pytest

from models.music_audio.onset import (
    OnsetMethod, Onset, OnsetFeatures,
    spectral_flux, high_frequency_content, complex_domain_odf,
    phase_deviation_odf, energy_odf,
    adaptive_threshold, pick_peaks, detect_onsets,
    compute_onset_features, multi_method_onset_fusion,
    onset_to_beat_alignment,
    _stft_magnitude, _stft_complex,
)


def _impulse_signal(sr: int = 22050, duration: float = 1.0, onset_times: list[float] | None = None) -> np.ndarray:
    n = int(sr * duration)
    sig = np.random.RandomState(42).randn(n) * 0.01
    if onset_times is None:
        onset_times = [0.25, 0.5, 0.75]
    for t in onset_times:
        idx = int(t * sr)
        if idx + 512 < n:
            sig[idx:idx + 512] += np.random.RandomState(int(t * 1000)).randn(512) * 0.5
    return sig


class TestSTFT:
    def test_stft_magnitude_shape(self):
        sig = np.random.randn(4096)
        mag = _stft_magnitude(sig, 1024, 512)
        expected_frames = 1 + (4096 - 1024) // 512
        assert mag.shape == (expected_frames, 513)

    def test_stft_magnitude_nonneg(self):
        sig = np.random.randn(4096)
        mag = _stft_magnitude(sig, 1024, 512)
        assert np.all(mag >= 0)

    def test_stft_complex_shape(self):
        sig = np.random.randn(4096)
        spec = _stft_complex(sig, 1024, 512)
        expected_frames = 1 + (4096 - 1024) // 512
        assert spec.shape == (expected_frames, 513)

    def test_stft_short_signal(self):
        sig = np.random.randn(100)
        mag = _stft_magnitude(sig, 1024, 512)
        assert mag.shape[0] == 0

    def test_stft_complex_matches_magnitude(self):
        sig = np.random.randn(4096)
        mag = _stft_magnitude(sig, 1024, 512)
        spec = _stft_complex(sig, 1024, 512)
        np.testing.assert_allclose(np.abs(spec), mag, atol=1e-10)


class TestODFs:
    def test_spectral_flux_length(self):
        sig = np.random.randn(4096)
        odf = spectral_flux(sig, 1024, 512)
        expected = 1 + (4096 - 1024) // 512
        assert len(odf) == expected

    def test_spectral_flux_nonneg(self):
        sig = np.random.randn(4096)
        odf = spectral_flux(sig, 1024, 512)
        assert np.all(odf >= 0)

    def test_hfc_length(self):
        sig = np.random.randn(4096)
        odf = high_frequency_content(sig, 1024, 512)
        expected = 1 + (4096 - 1024) // 512
        assert len(odf) == expected

    def test_hfc_nonneg(self):
        sig = np.random.randn(4096)
        odf = high_frequency_content(sig, 1024, 512)
        assert np.all(odf >= 0)

    def test_complex_domain_length(self):
        sig = np.random.randn(4096)
        odf = complex_domain_odf(sig, 1024, 512)
        expected = 1 + (4096 - 1024) // 512
        assert len(odf) == expected

    def test_phase_deviation_length(self):
        sig = np.random.randn(4096)
        odf = phase_deviation_odf(sig, 1024, 512)
        expected = 1 + (4096 - 1024) // 512
        assert len(odf) == expected

    def test_energy_odf_length(self):
        sig = np.random.randn(4096)
        odf = energy_odf(sig, 1024, 512)
        expected = 1 + (4096 - 1024) // 512
        assert len(odf) == expected

    def test_energy_odf_nonneg(self):
        sig = np.random.randn(4096)
        odf = energy_odf(sig, 1024, 512)
        assert np.all(odf >= 0)

    def test_all_methods_empty_signal(self):
        sig = np.array([])
        assert len(spectral_flux(sig)) == 0
        assert len(high_frequency_content(sig)) == 0
        assert len(complex_domain_odf(sig)) == 0
        assert len(phase_deviation_odf(sig)) == 0
        assert len(energy_odf(sig)) == 0


class TestThresholdAndPeaks:
    def test_adaptive_threshold_shape(self):
        odf = np.random.rand(50)
        thresh = adaptive_threshold(odf, window=7)
        assert len(thresh) == 50

    def test_adaptive_threshold_empty(self):
        assert len(adaptive_threshold(np.array([]))) == 0

    def test_pick_peaks_basic(self):
        odf = np.array([0, 0.1, 0.5, 0.3, 0.1, 0, 0.8, 1.0, 0.6, 0.1])
        thresh = np.zeros(10)
        peaks = pick_peaks(odf, thresh, min_distance=1)
        assert 2 in peaks
        assert 7 in peaks

    def test_pick_peaks_min_distance(self):
        odf = np.array([0, 0.5, 0.3, 0.5, 0.3, 0.5, 0.1])
        thresh = np.zeros(7)
        peaks = pick_peaks(odf, thresh, min_distance=3)
        assert len(peaks) <= 2

    def test_pick_peaks_no_threshold_uses_adaptive(self):
        odf = np.zeros(20)
        odf[10] = 10.0
        peaks = pick_peaks(odf)
        assert 10 in peaks


class TestDetectOnsets:
    def test_detect_onsets_returns_list(self):
        sig = _impulse_signal()
        onsets = detect_onsets(sig, sr=22050, method=OnsetMethod.SPECTRAL_FLUX)
        assert isinstance(onsets, list)
        for o in onsets:
            assert isinstance(o, Onset)

    def test_detect_onsets_all_methods(self):
        sig = _impulse_signal()
        for method in OnsetMethod:
            onsets = detect_onsets(sig, sr=22050, method=method)
            assert isinstance(onsets, list)

    def test_detect_onsets_time_monotonic(self):
        sig = _impulse_signal()
        onsets = detect_onsets(sig, sr=22050)
        times = [o.time for o in onsets]
        assert times == sorted(times)

    def test_detect_onsets_empty_signal(self):
        assert detect_onsets(np.array([]), sr=22050) == []

    def test_detect_onsets_strength_positive(self):
        sig = _impulse_signal()
        onsets = detect_onsets(sig, sr=22050)
        for o in onsets:
            assert o.strength >= 0


class TestOnsetFeatures:
    def test_features_multiple_onsets(self):
        onsets = [Onset(frame=i * 10, time=i * 0.1, strength=1.0, method=OnsetMethod.SPECTRAL_FLUX) for i in range(5)]
        feat = compute_onset_features(onsets, total_duration=0.5)
        assert feat.mean_ioi == pytest.approx(0.1)
        assert feat.std_ioi == pytest.approx(0.0, abs=1e-10)
        assert feat.regularity == pytest.approx(1.0, abs=0.01)
        assert feat.density == pytest.approx(10.0)

    def test_features_single_onset(self):
        onsets = [Onset(frame=0, time=0.0, strength=1.0, method=OnsetMethod.SPECTRAL_FLUX)]
        feat = compute_onset_features(onsets, total_duration=1.0)
        assert feat.mean_ioi == 0.0
        assert feat.density == 1.0

    def test_features_empty(self):
        feat = compute_onset_features([], total_duration=1.0)
        assert feat.density == 0.0
        assert len(feat.inter_onset_intervals) == 0

    def test_features_irregular(self):
        onsets = [
            Onset(frame=0, time=0.0, strength=1.0, method=OnsetMethod.SPECTRAL_FLUX),
            Onset(frame=10, time=0.1, strength=1.0, method=OnsetMethod.SPECTRAL_FLUX),
            Onset(frame=50, time=0.5, strength=1.0, method=OnsetMethod.SPECTRAL_FLUX),
        ]
        feat = compute_onset_features(onsets, total_duration=1.0)
        assert feat.std_ioi > 0
        assert feat.regularity < 1.0


class TestFusion:
    def test_fusion_returns_list(self):
        sig = _impulse_signal()
        fused = multi_method_onset_fusion(sig, sr=22050)
        assert isinstance(fused, list)

    def test_fusion_with_high_vote_threshold(self):
        sig = _impulse_signal()
        fused = multi_method_onset_fusion(sig, sr=22050, vote_threshold=10)
        assert len(fused) == 0

    def test_fusion_with_low_threshold(self):
        sig = _impulse_signal()
        fused = multi_method_onset_fusion(sig, sr=22050, vote_threshold=1)
        assert isinstance(fused, list)

    def test_fusion_empty_signal(self):
        fused = multi_method_onset_fusion(np.array([]), sr=22050)
        assert fused == []


class TestBeatAlignment:
    def test_perfect_alignment(self):
        onsets = [Onset(frame=i, time=i * 0.5, strength=1.0, method=OnsetMethod.SPECTRAL_FLUX) for i in range(4)]
        beats = [0.0, 0.5, 1.0, 1.5]
        result = onset_to_beat_alignment(onsets, beats, tolerance=0.05)
        assert result["aligned_ratio"] == 1.0
        assert result["mean_offset"] == pytest.approx(0.0)

    def test_no_alignment(self):
        onsets = [Onset(frame=i, time=i * 0.5 + 0.25, strength=1.0, method=OnsetMethod.SPECTRAL_FLUX) for i in range(4)]
        beats = [0.0, 0.5, 1.0, 1.5]
        result = onset_to_beat_alignment(onsets, beats, tolerance=0.01)
        assert result["aligned_ratio"] == 0.0

    def test_empty_onsets(self):
        result = onset_to_beat_alignment([], [0.5, 1.0])
        assert result["aligned_ratio"] == 0.0

    def test_empty_beats(self):
        onsets = [Onset(frame=0, time=0.0, strength=1.0, method=OnsetMethod.SPECTRAL_FLUX)]
        result = onset_to_beat_alignment(onsets, [])
        assert result["aligned_ratio"] == 0.0
