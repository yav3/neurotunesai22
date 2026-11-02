"""Tests for music_transforms: spectral and temporal transforms."""
import pytest
import numpy as np
from models.music_transforms.spectral import (
    stft, istft, magnitude_spectrogram, power_spectrogram, phase_spectrogram,
    hz_to_mel, mel_to_hz, mel_filterbank, mel_spectrogram,
    amplitude_to_db, cqt_frequencies, pseudo_cqt, chromagram,
)
from models.music_transforms.temporal import (
    onset_strength, spectral_flux, zero_crossing_rate, rms_energy,
    autocorrelation, temporal_centroid,
)


@pytest.fixture
def sine_signal():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2 * np.pi * 440 * t), sr


@pytest.fixture
def white_noise():
    rng = np.random.default_rng(42)
    return rng.standard_normal(22050)


class TestSTFT:
    def test_shape(self, sine_signal):
        sig, sr = sine_signal
        S = stft(sig)
        assert S.shape[0] == 1025
        assert S.shape[1] > 0

    def test_complex(self, sine_signal):
        sig, sr = sine_signal
        S = stft(sig)
        assert S.dtype == complex

    def test_roundtrip(self, sine_signal):
        sig, sr = sine_signal
        S = stft(sig, n_fft=2048, hop_length=512)
        recon = istft(S, hop_length=512, n_fft=2048)
        min_len = min(len(sig), len(recon))
        corr = np.corrcoef(sig[:min_len], recon[:min_len])[0, 1]
        assert corr > 0.9

    def test_windows(self, sine_signal):
        sig, sr = sine_signal
        for win in ["hann", "hamming", "blackman"]:
            S = stft(sig, window=win)
            assert S.shape[0] == 1025


class TestSpectrogram:
    def test_magnitude(self, sine_signal):
        sig, sr = sine_signal
        S = stft(sig)
        mag = magnitude_spectrogram(S)
        assert np.all(mag >= 0)

    def test_power(self, sine_signal):
        sig, sr = sine_signal
        S = stft(sig)
        pw = power_spectrogram(S)
        assert np.all(pw >= 0)

    def test_phase(self, sine_signal):
        sig, sr = sine_signal
        S = stft(sig)
        ph = phase_spectrogram(S)
        assert np.all(np.abs(ph) <= np.pi + 1e-6)


class TestMelScale:
    def test_roundtrip(self):
        for f in [100.0, 440.0, 1000.0]:
            assert mel_to_hz(hz_to_mel(f)) == pytest.approx(f, rel=1e-6)

    def test_monotonic(self):
        freqs = [100, 500, 1000, 4000]
        mels = [float(hz_to_mel(f)) for f in freqs]
        assert all(mels[i] < mels[i + 1] for i in range(len(mels) - 1))


class TestMelFilterbank:
    def test_shape(self):
        fb = mel_filterbank(n_mels=40, n_fft=2048, sr=22050)
        assert fb.shape == (40, 1025)

    def test_nonnegative(self):
        fb = mel_filterbank()
        assert np.all(fb >= 0)

    def test_sum_positive(self):
        fb = mel_filterbank(n_mels=40)
        row_sums = fb.sum(axis=1)
        assert np.all(row_sums > 0)


class TestMelSpectrogram:
    def test_shape(self, sine_signal):
        sig, sr = sine_signal
        mel = mel_spectrogram(sig, sr=sr, n_mels=40)
        assert mel.shape[0] == 40
        assert mel.shape[1] > 0

    def test_nonnegative(self, sine_signal):
        sig, sr = sine_signal
        mel = mel_spectrogram(sig, sr=sr)
        assert np.all(mel >= 0)


class TestAmplitudeToDb:
    def test_unity(self):
        S = np.ones((10, 10))
        db = amplitude_to_db(S)
        assert np.allclose(db, 0.0)

    def test_half(self):
        S = np.full((5, 5), 0.5)
        db = amplitude_to_db(S)
        assert np.all(db < 0)


class TestCQT:
    def test_frequencies(self):
        freqs = cqt_frequencies(n_bins=84, f_min=32.7, bins_per_octave=12)
        assert len(freqs) == 84
        assert freqs[0] == pytest.approx(32.7)
        assert freqs[12] == pytest.approx(32.7 * 2, rel=0.01)

    def test_pseudo_cqt_shape(self, sine_signal):
        sig, sr = sine_signal
        cq = pseudo_cqt(sig, sr=sr, n_bins=36)
        assert cq.shape[0] == 36
        assert cq.shape[1] > 0

    def test_pseudo_cqt_nonneg(self, sine_signal):
        sig, sr = sine_signal
        cq = pseudo_cqt(sig, sr=sr)
        assert np.all(cq >= 0)


class TestChromagram:
    def test_shape(self, sine_signal):
        sig, sr = sine_signal
        chroma = chromagram(sig, sr=sr)
        assert chroma.shape[0] == 12
        assert chroma.shape[1] > 0

    def test_normalized(self, sine_signal):
        sig, sr = sine_signal
        chroma = chromagram(sig, sr=sr)
        assert np.all(chroma <= 1.0 + 1e-6)
        assert np.all(chroma >= 0)

    def test_440_peak(self, sine_signal):
        sig, sr = sine_signal
        chroma = chromagram(sig, sr=sr)
        mean_chroma = chroma.mean(axis=1)
        peak_bin = np.argmax(mean_chroma)
        assert peak_bin == 0  # A440 maps to bin 0 (12*log2(440/440) % 12 = 0)


class TestOnsetStrength:
    def test_shape(self, sine_signal):
        sig, sr = sine_signal
        onset = onset_strength(sig, sr=sr)
        assert len(onset) > 0

    def test_bounded(self, sine_signal):
        sig, sr = sine_signal
        onset = onset_strength(sig, sr=sr)
        assert np.all(onset >= 0)
        assert np.all(onset <= 1.0 + 1e-6)


class TestSpectralFlux:
    def test_basic(self, sine_signal):
        sig, sr = sine_signal
        S = stft(sig)
        mag = magnitude_spectrogram(S)
        flux = spectral_flux(mag)
        assert len(flux) == mag.shape[1] - 1
        assert np.all(flux >= 0)


class TestZCR:
    def test_sine(self, sine_signal):
        sig, sr = sine_signal
        zcr = zero_crossing_rate(sig)
        assert len(zcr) > 0
        assert np.all(zcr >= 0)
        assert np.all(zcr <= 0.5)

    def test_noise_higher(self, sine_signal, white_noise):
        sig, sr = sine_signal
        zcr_sine = zero_crossing_rate(sig).mean()
        zcr_noise = zero_crossing_rate(white_noise).mean()
        assert zcr_noise > zcr_sine


class TestRMS:
    def test_shape(self, sine_signal):
        sig, sr = sine_signal
        rms = rms_energy(sig)
        assert len(rms) > 0
        assert np.all(rms >= 0)

    def test_silence(self):
        sig = np.zeros(4096)
        rms = rms_energy(sig)
        assert np.allclose(rms, 0.0)


class TestAutocorrelation:
    def test_peak_at_zero(self, sine_signal):
        sig, sr = sine_signal
        acf = autocorrelation(sig, max_lag=1000)
        assert acf[0] == pytest.approx(1.0)

    def test_periodic(self, sine_signal):
        sig, sr = sine_signal
        acf = autocorrelation(sig, max_lag=sr // 2)
        period_samples = int(sr / 440)
        peak = acf[period_samples]
        assert peak > 0.8


class TestTemporalCentroid:
    def test_early_energy(self):
        sig = np.zeros(22050)
        sig[:1000] = 1.0
        tc = temporal_centroid(sig, sr=22050)
        assert tc < 0.1

    def test_late_energy(self):
        sig = np.zeros(22050)
        sig[-1000:] = 1.0
        tc = temporal_centroid(sig, sr=22050)
        assert tc > 0.9

    def test_silence(self):
        assert temporal_centroid(np.zeros(100)) == 0.0
