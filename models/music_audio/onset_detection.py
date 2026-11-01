"""Onset detection: spectral flux, HFC, complex-domain, SuperFlux, peak picking."""
from __future__ import annotations

import numpy as np


class OnsetDetector:
    """Frame-level onset detection functions and peak picking."""

    @staticmethod
    def spectral_flux(spectrogram: np.ndarray) -> np.ndarray:
        """Half-wave rectified spectral flux (Bello et al. 2005)."""
        if spectrogram.shape[1] < 2:
            return np.zeros(spectrogram.shape[1])
        diff = np.diff(spectrogram, axis=1)
        hw = np.maximum(diff, 0.0)
        return np.sum(hw, axis=0)

    @staticmethod
    def high_frequency_content(
        spectrogram: np.ndarray, freqs: np.ndarray | None = None
    ) -> np.ndarray:
        """Energy weighted toward high frequencies (Masri 1996)."""
        n_bins, n_frames = spectrogram.shape
        if freqs is None:
            freqs = np.arange(n_bins, dtype=float)
        weights = freqs[:n_bins] / max(freqs.max(), 1e-12)
        return (spectrogram.T @ weights).ravel()

    @staticmethod
    def complex_domain(stft_frames: np.ndarray) -> np.ndarray:
        """Complex-domain onset detection (Bello 2005).

        Uses deviation between predicted and actual complex spectrum.
        ``stft_frames`` has shape (n_bins, n_frames) and is complex-valued.
        """
        if stft_frames.shape[1] < 3:
            return np.zeros(stft_frames.shape[1])
        mag = np.abs(stft_frames)
        phase = np.angle(stft_frames)
        phase_diff = np.diff(phase, axis=1)
        predicted_phase = phase[:, 1:-1] + phase_diff[:, :-1]
        predicted = mag[:, 1:-1] * np.exp(1j * predicted_phase)
        actual = stft_frames[:, 2:]
        deviation = np.abs(actual - predicted)
        odf = np.sum(deviation, axis=0)
        return np.concatenate([np.zeros(2), odf])

    @staticmethod
    def superflux(
        spectrogram: np.ndarray, max_filter_size: int = 3
    ) -> np.ndarray:
        """Vibrato-robust onset detection (Böck & Widmer 2013).

        Applies a maximum filter along frequency before computing flux,
        suppressing spectral modulations from vibrato.
        """
        from scipy.ndimage import maximum_filter1d

        if spectrogram.shape[1] < 2:
            return np.zeros(spectrogram.shape[1])
        filtered = maximum_filter1d(spectrogram, size=max_filter_size, axis=0)
        diff = filtered[:, 1:] - spectrogram[:, :-1]
        hw = np.maximum(diff, 0.0)
        odf = np.sum(hw, axis=0)
        return np.concatenate([[0.0], odf])

    @staticmethod
    def pick_peaks(
        detection_fn: np.ndarray,
        threshold: float = 0.0,
        min_distance: int = 1,
    ) -> np.ndarray:
        """Adaptive peak picking with minimum inter-onset interval.

        Returns indices of detected onsets.
        """
        if len(detection_fn) == 0:
            return np.array([], dtype=int)
        above = detection_fn > threshold
        peaks = []
        for i in range(1, len(detection_fn) - 1):
            if (
                above[i]
                and detection_fn[i] >= detection_fn[i - 1]
                and detection_fn[i] >= detection_fn[i + 1]
            ):
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        return np.array(peaks, dtype=int)

    @staticmethod
    def onset_strength_envelope(
        signal: np.ndarray, sr: int = 22050, hop_length: int = 512
    ) -> np.ndarray:
        """Aggregate onset strength from magnitude spectrogram."""
        n_fft = 2048
        if len(signal) < n_fft:
            return np.array([0.0])
        step = hop_length
        n_frames = 1 + (len(signal) - n_fft) // step
        window = np.hanning(n_fft)
        mag = np.zeros((n_fft // 2 + 1, n_frames))
        for i in range(n_frames):
            frame = signal[i * step : i * step + n_fft] * window
            mag[:, i] = np.abs(np.fft.rfft(frame))
        return OnsetDetector.spectral_flux(mag)

    @staticmethod
    def tempo_from_onsets(onset_times: np.ndarray) -> float:
        """Estimate BPM from inter-onset intervals via histogram peak."""
        if len(onset_times) < 2:
            return 0.0
        ioi = np.diff(onset_times)
        ioi = ioi[ioi > 0.1]
        if len(ioi) == 0:
            return 0.0
        bpm_candidates = 60.0 / ioi
        bpm_candidates = bpm_candidates[(bpm_candidates > 30) & (bpm_candidates < 300)]
        if len(bpm_candidates) == 0:
            return 0.0
        hist, edges = np.histogram(bpm_candidates, bins=50)
        peak_idx = np.argmax(hist)
        return float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)
