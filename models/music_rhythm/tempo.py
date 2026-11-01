"""Tempo estimation: autocorrelation, spectral flux, onset-based methods."""
from __future__ import annotations
import numpy as np


def bpm_to_period(bpm: float, sr: float = 22050.0, hop: int = 512) -> float:
    """Convert BPM to period in frames."""
    if bpm <= 0:
        return 0.0
    seconds_per_beat = 60.0 / bpm
    return seconds_per_beat * sr / hop


def period_to_bpm(period: float, sr: float = 22050.0, hop: int = 512) -> float:
    """Convert period in frames to BPM."""
    if period <= 0:
        return 0.0
    seconds_per_beat = period * hop / sr
    return 60.0 / seconds_per_beat


def tempo_from_onsets(
    onset_times: np.ndarray,
    min_bpm: float = 30.0,
    max_bpm: float = 300.0,
) -> float:
    """Estimate tempo from onset times (seconds) via IOI histogram."""
    if len(onset_times) < 2:
        return 0.0
    ioi = np.diff(onset_times)
    ioi = ioi[ioi > 0]
    if len(ioi) == 0:
        return 0.0

    min_ioi = 60.0 / max_bpm
    max_ioi = 60.0 / min_bpm
    mask = (ioi >= min_ioi) & (ioi <= max_ioi)
    valid = ioi[mask]
    if len(valid) == 0:
        valid = ioi

    n_bins = max(10, min(100, len(valid)))
    counts, edges = np.histogram(valid, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_idx = int(np.argmax(counts))
    dominant_ioi = centers[peak_idx]
    if dominant_ioi <= 0:
        return 0.0
    return 60.0 / dominant_ioi


def tempo_from_autocorrelation(
    signal: np.ndarray,
    sr: float = 22050.0,
    min_bpm: float = 30.0,
    max_bpm: float = 300.0,
) -> tuple[float, np.ndarray]:
    """Estimate tempo via windowed autocorrelation of the onset strength envelope.

    Returns (bpm, autocorrelation_curve).
    """
    frame_len = 2048
    hop = 512
    n_frames = max(1, (len(signal) - frame_len) // hop + 1)

    # onset strength: spectral flux
    envelope = np.zeros(n_frames)
    prev_mag = None
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        window = np.hanning(frame_len)
        mag = np.abs(np.fft.rfft(frame * window))
        if prev_mag is not None:
            diff = mag - prev_mag
            envelope[i] = max(0.0, float(np.sum(diff[diff > 0])))
        prev_mag = mag

    # autocorrelation
    min_lag = max(1, int(bpm_to_period(max_bpm, sr, hop)))
    max_lag = min(len(envelope) - 1, int(bpm_to_period(min_bpm, sr, hop)))
    if max_lag <= min_lag:
        return 0.0, np.array([])

    env_centered = envelope - np.mean(envelope)
    norm = np.dot(env_centered, env_centered)
    if norm < 1e-12:
        return 0.0, np.zeros(max_lag - min_lag)

    acf = np.zeros(max_lag - min_lag)
    for k_idx, lag in enumerate(range(min_lag, max_lag)):
        acf[k_idx] = np.dot(env_centered[:len(env_centered) - lag],
                            env_centered[lag:]) / norm

    peak_idx = int(np.argmax(acf))
    best_lag = min_lag + peak_idx
    bpm = period_to_bpm(best_lag, sr, hop)
    return bpm, acf


def tempo_from_spectral_flux(
    signal: np.ndarray,
    sr: float = 22050.0,
    n_fft: int = 2048,
    hop: int = 512,
    min_bpm: float = 30.0,
    max_bpm: float = 300.0,
) -> float:
    """Estimate tempo from spectral flux peak periodicity."""
    n_frames = max(1, (len(signal) - n_fft) // hop + 1)
    flux = np.zeros(n_frames)
    prev_mag = None
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        mag = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
        if prev_mag is not None:
            diff = mag - prev_mag
            flux[i] = float(np.sum(diff[diff > 0]))
        prev_mag = mag

    if np.max(flux) < 1e-12:
        return 0.0

    # periodicity via autocorrelation of flux
    min_lag = max(1, int(bpm_to_period(max_bpm, sr, hop)))
    max_lag = min(len(flux) - 1, int(bpm_to_period(min_bpm, sr, hop)))
    if max_lag <= min_lag:
        return 0.0

    f = flux - np.mean(flux)
    norm = np.dot(f, f)
    if norm < 1e-12:
        return 0.0

    best_lag = min_lag
    best_val = -1.0
    for lag in range(min_lag, max_lag):
        val = np.dot(f[:len(f) - lag], f[lag:]) / norm
        if val > best_val:
            best_val = val
            best_lag = lag

    return period_to_bpm(best_lag, sr, hop)


class TempoEstimator:
    """Multi-method tempo estimator with octave-error resolution."""

    def __init__(
        self,
        sr: float = 22050.0,
        min_bpm: float = 30.0,
        max_bpm: float = 300.0,
    ) -> None:
        self.sr = sr
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def estimate(self, signal: np.ndarray) -> dict:
        bpm_ac, acf = tempo_from_autocorrelation(
            signal, self.sr, self.min_bpm, self.max_bpm
        )
        bpm_sf = tempo_from_spectral_flux(
            signal, self.sr, min_bpm=self.min_bpm, max_bpm=self.max_bpm
        )

        candidates = [b for b in [bpm_ac, bpm_sf] if b > 0]
        if not candidates:
            return {"bpm": 0.0, "confidence": 0.0, "method": "none"}

        # resolve octave errors: prefer candidate in [60, 180] if available
        def score(b: float) -> float:
            if 60 <= b <= 180:
                return 2.0
            if 40 <= b <= 240:
                return 1.0
            return 0.5

        best = max(candidates, key=score)
        agreement = 1.0 if len(candidates) >= 2 and abs(candidates[0] - candidates[1]) < 10 else 0.5
        return {
            "bpm": best,
            "confidence": agreement,
            "bpm_autocorrelation": bpm_ac,
            "bpm_spectral_flux": bpm_sf,
        }

    def octave_correct(self, bpm: float, target_range: tuple[float, float] = (60, 180)) -> float:
        """Move bpm into target range via doubling/halving."""
        if bpm <= 0:
            return 0.0
        lo, hi = target_range
        while bpm < lo and bpm > 0:
            bpm *= 2
        while bpm > hi:
            bpm /= 2
        return bpm
