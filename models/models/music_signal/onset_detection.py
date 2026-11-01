"""Onset detection: identify note beginnings and transients in audio signals."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class OnsetResult:
    """Result of onset detection."""
    times: np.ndarray
    strength_curve: np.ndarray
    sr: int
    hop_size: int


def spectral_flux(magnitude: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux onset strength.

    Args:
        magnitude: (n_freq, n_frames) magnitude spectrogram.

    Returns:
        (n_frames,) onset strength curve.
    """
    diff = np.diff(magnitude, axis=1)
    hw = np.maximum(diff, 0.0)
    return np.concatenate([[0.0], np.sum(hw, axis=0)])


def high_frequency_content(magnitude: np.ndarray) -> np.ndarray:
    """High-frequency content onset function.

    Weights each bin by its index (frequency), emphasising transients
    which spread energy into high bins.
    """
    n_freq, n_frames = magnitude.shape
    weights = np.arange(1, n_freq + 1, dtype=np.float64)
    hfc = np.sum(magnitude * weights[:, None], axis=0)
    return hfc


def complex_domain(stft_complex: np.ndarray) -> np.ndarray:
    """Complex-domain onset function (Duxbury et al. 2003).

    Uses deviation of actual STFT from the predicted phase+magnitude.
    """
    n_freq, n_frames = stft_complex.shape
    mag = np.abs(stft_complex)
    phase = np.angle(stft_complex)

    if n_frames < 3:
        return np.zeros(n_frames)

    predicted_phase = 2.0 * phase[:, 1:-1] - phase[:, :-2]
    predicted = mag[:, 1:-1] * np.exp(1j * predicted_phase)
    actual = stft_complex[:, 2:]
    deviation = np.abs(actual - predicted)
    odf = np.sum(deviation, axis=0)
    return np.concatenate([[0.0, 0.0], odf])


def energy_onset(signal: np.ndarray, hop_size: int = 512) -> np.ndarray:
    """Simple energy-based onset function (RMS per frame)."""
    n_frames = 1 + (len(signal) - 1) // hop_size
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + hop_size, len(signal))
        frame = signal[start:end]
        energy[i] = np.sqrt(np.mean(frame ** 2))

    diff = np.diff(energy)
    hw = np.maximum(diff, 0.0)
    return np.concatenate([[0.0], hw])


def pick_peaks(
    onset_curve: np.ndarray,
    threshold: float = 0.3,
    min_distance: int = 5,
    normalize: bool = True,
) -> np.ndarray:
    """Pick peaks from an onset strength curve.

    Args:
        onset_curve: onset strength values per frame.
        threshold: minimum normalised strength (0-1 if normalize=True).
        min_distance: minimum frames between peaks.
        normalize: whether to normalize curve to [0, 1].

    Returns:
        Array of frame indices where onsets occur.
    """
    if len(onset_curve) < 3:
        return np.array([], dtype=int)

    curve = onset_curve.copy().astype(np.float64)
    if normalize:
        mx = curve.max()
        if mx > 0:
            curve = curve / mx

    peaks = []
    for i in range(1, len(curve) - 1):
        if curve[i] > curve[i - 1] and curve[i] >= curve[i + 1]:
            if curve[i] >= threshold:
                peaks.append(i)

    if not peaks:
        return np.array([], dtype=int)

    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_distance:
            filtered.append(p)
    return np.array(filtered, dtype=int)


def frames_to_time(frames: np.ndarray, sr: int, hop_size: int) -> np.ndarray:
    """Convert frame indices to time in seconds."""
    return frames.astype(np.float64) * hop_size / sr


def onset_detect(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_size: int = 512,
    method: str = "spectral_flux",
    threshold: float = 0.3,
    min_distance_ms: float = 50.0,
) -> OnsetResult:
    """Detect onsets in an audio signal.

    Args:
        signal: mono audio signal.
        sr: sample rate.
        n_fft: FFT size.
        hop_size: hop size in samples.
        method: 'spectral_flux', 'hfc', 'energy'.
        threshold: peak-picking threshold.
        min_distance_ms: minimum time between onsets in milliseconds.
    """
    from .spectral_tools import stft as _stft

    min_frames = max(1, int(min_distance_ms * sr / (1000.0 * hop_size)))

    if method == "energy":
        strength = energy_onset(signal, hop_size)
    else:
        window = np.hanning(n_fft)
        spec = _stft(signal, n_fft, hop_size, window)
        mag = np.abs(spec)

        if method == "hfc":
            strength = high_frequency_content(mag)
        elif method == "complex":
            strength = complex_domain(spec)
        else:
            strength = spectral_flux(mag)

    peak_frames = pick_peaks(strength, threshold=threshold, min_distance=min_frames)
    times = frames_to_time(peak_frames, sr, hop_size)

    return OnsetResult(
        times=times,
        strength_curve=strength,
        sr=sr,
        hop_size=hop_size,
    )
