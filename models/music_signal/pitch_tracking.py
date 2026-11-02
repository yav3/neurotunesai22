"""Pitch tracking: fundamental frequency estimation for monophonic signals."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def hz_to_midi(frequency: float) -> float:
    """Convert frequency in Hz to MIDI note number (float)."""
    if frequency <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(frequency / 440.0)


def midi_to_hz(midi_note: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def pitch_class(midi_note: int) -> int:
    """Pitch class (0=C, 1=C#, ..., 11=B) from MIDI note."""
    return midi_note % 12


def cents_difference(f1: float, f2: float) -> float:
    """Interval in cents between two frequencies."""
    if f1 <= 0 or f2 <= 0:
        return 0.0
    return 1200.0 * np.log2(f2 / f1)


def note_name(midi_note: int) -> str:
    """Note name from MIDI number (e.g. 60 -> 'C4')."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_note // 12) - 1
    return f"{names[midi_note % 12]}{octave}"


@dataclass
class PitchResult:
    """Frame-wise pitch tracking result."""
    f0: np.ndarray
    confidence: np.ndarray
    times: np.ndarray
    sr: int


def autocorrelation_pitch(
    frame: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 4000.0,
) -> tuple[float, float]:
    """Estimate pitch of a single frame via autocorrelation.

    Returns:
        (f0_hz, confidence) where confidence in [0, 1].
    """
    n = len(frame)
    if n < 4:
        return 0.0, 0.0

    lag_min = max(1, int(sr / fmax))
    lag_max = min(n - 1, int(sr / fmin))

    if lag_min >= lag_max:
        return 0.0, 0.0

    windowed = frame * np.hanning(n)
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[n - 1:]
    if corr[0] == 0:
        return 0.0, 0.0
    corr = corr / corr[0]

    search = corr[lag_min:lag_max + 1]
    if len(search) == 0:
        return 0.0, 0.0

    peak_idx = np.argmax(search)
    peak_val = search[peak_idx]
    lag = lag_min + peak_idx

    if lag == 0:
        return 0.0, 0.0

    f0 = sr / lag
    confidence = float(np.clip(peak_val, 0.0, 1.0))
    return f0, confidence


def _cumulative_mean_normalized_difference(d: np.ndarray) -> np.ndarray:
    """Cumulative mean normalised difference function for YIN."""
    n = len(d)
    out = np.ones(n)
    out[0] = 1.0
    running_sum = 0.0
    for tau in range(1, n):
        running_sum += d[tau]
        if running_sum == 0:
            out[tau] = 1.0
        else:
            out[tau] = d[tau] * tau / running_sum
    return out


def yin_frame(
    frame: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    threshold: float = 0.1,
) -> tuple[float, float]:
    """YIN pitch estimation on a single frame.

    Returns:
        (f0_hz, confidence). f0=0 if unvoiced.
    """
    n = len(frame)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(n // 2, int(sr / fmin))

    if tau_min >= tau_max:
        return 0.0, 0.0

    d = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        diff = frame[:n - tau] - frame[tau:n]
        d[tau] = np.sum(diff ** 2)

    cmnd = _cumulative_mean_normalized_difference(d)

    best_tau = 0
    for tau in range(tau_min, tau_max + 1):
        if cmnd[tau] < threshold:
            while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            best_tau = tau
            break

    if best_tau == 0:
        idx = np.argmin(cmnd[tau_min:tau_max + 1]) + tau_min
        if cmnd[idx] < 0.5:
            best_tau = idx
        else:
            return 0.0, 0.0

    f0 = sr / best_tau
    confidence = float(1.0 - np.clip(cmnd[best_tau], 0.0, 1.0))
    return f0, confidence


def track_pitch(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    method: str = "yin",
    fmin: float = 50.0,
    fmax: float = 2000.0,
    threshold: float = 0.1,
) -> PitchResult:
    """Track pitch over time in a monophonic signal.

    Args:
        signal: mono audio.
        sr: sample rate.
        frame_size: analysis frame size.
        hop_size: hop size.
        method: 'yin' or 'autocorrelation'.
        fmin, fmax: frequency range.
        threshold: YIN threshold.
    """
    n_frames = 1 + max(0, (len(signal) - frame_size) // hop_size)
    f0 = np.zeros(n_frames)
    confidence = np.zeros(n_frames)
    times = np.arange(n_frames) * hop_size / sr

    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        if end > len(signal):
            frame = np.pad(signal[start:], (0, end - len(signal)))
        else:
            frame = signal[start:end]

        if method == "autocorrelation":
            f0[i], confidence[i] = autocorrelation_pitch(frame, sr, fmin, fmax)
        else:
            f0[i], confidence[i] = yin_frame(frame, sr, fmin, fmax, threshold)

    return PitchResult(f0=f0, confidence=confidence, times=times, sr=sr)


def harmonic_product_spectrum(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    n_harmonics: int = 5,
) -> float:
    """Estimate f0 via harmonic product spectrum.

    Downsample and multiply the magnitude spectrum at harmonic intervals.
    """
    n = len(magnitude)
    hps = magnitude[:n].copy()
    for h in range(2, n_harmonics + 1):
        decimated = magnitude[::h]
        length = min(len(hps), len(decimated))
        hps[:length] *= decimated[:length]

    hps[0] = 0.0
    peak_idx = np.argmax(hps)
    if peak_idx < len(freqs):
        return float(freqs[peak_idx])
    return 0.0


def median_filter_pitch(f0: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter to smooth pitch contour and remove octave errors."""
    n = len(f0)
    result = f0.copy()
    half = kernel_size // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = f0[start:end]
        voiced = window[window > 0]
        if len(voiced) > 0:
            result[i] = np.median(voiced)
        else:
            result[i] = 0.0
    return result
