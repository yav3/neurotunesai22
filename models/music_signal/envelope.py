"""Signal envelope analysis: Hilbert transform, analytic signal, AM/FM decomposition.

Extracts the instantaneous amplitude (envelope) and instantaneous frequency
of audio signals via the analytic signal representation.  Useful for tremolo
detection, amplitude modulation analysis, and transient shaping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert as scipy_hilbert


def analytic_signal(x: np.ndarray) -> np.ndarray:
    """Compute the analytic signal via the Hilbert transform.

    Returns a complex array z = x + j·H{x} where H is the Hilbert transform.
    """
    if len(x) == 0:
        return np.array([], dtype=complex)
    return scipy_hilbert(x)


def instantaneous_amplitude(x: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude (envelope) of a real signal."""
    z = analytic_signal(x)
    return np.abs(z)


def instantaneous_phase(x: np.ndarray) -> np.ndarray:
    """Instantaneous phase (unwrapped) in radians."""
    z = analytic_signal(x)
    return np.unwrap(np.angle(z))


def instantaneous_frequency(x: np.ndarray, sr: float) -> np.ndarray:
    """Instantaneous frequency in Hz via the phase derivative.

    Returns an array of length len(x) - 1.
    """
    if len(x) < 2:
        return np.array([])
    phase = instantaneous_phase(x)
    return np.diff(phase) / (2.0 * np.pi) * sr


def envelope_follower(
    x: np.ndarray,
    attack_samples: int = 10,
    release_samples: int = 100,
) -> np.ndarray:
    """Simple attack/release envelope follower (peak detector).

    ``attack_samples`` and ``release_samples`` control how quickly the
    envelope rises and falls respectively.
    """
    if len(x) == 0:
        return np.array([])
    rectified = np.abs(x)
    env = np.zeros(len(x))
    alpha_a = 1.0 / max(attack_samples, 1)
    alpha_r = 1.0 / max(release_samples, 1)
    env[0] = rectified[0]
    for i in range(1, len(x)):
        if rectified[i] > env[i - 1]:
            env[i] = env[i - 1] + alpha_a * (rectified[i] - env[i - 1])
        else:
            env[i] = env[i - 1] + alpha_r * (rectified[i] - env[i - 1])
    return env


def rms_envelope(x: np.ndarray, window: int = 1024, hop: int = 512) -> np.ndarray:
    """RMS envelope computed over overlapping windows."""
    if len(x) < window:
        if len(x) == 0:
            return np.array([])
        return np.array([float(np.sqrt(np.mean(x ** 2)))])
    n_frames = 1 + (len(x) - window) // hop
    env = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + window]
        env[i] = float(np.sqrt(np.mean(frame ** 2)))
    return env


@dataclass(frozen=True)
class AMFMDecomposition:
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray


def am_fm_decompose(x: np.ndarray, sr: float) -> AMFMDecomposition:
    """Decompose a signal into AM (amplitude modulation) and FM (frequency modulation) components."""
    amp = instantaneous_amplitude(x)
    phase = instantaneous_phase(x)
    freq = instantaneous_frequency(x, sr)
    return AMFMDecomposition(amplitude=amp, phase=phase, frequency=freq)


def tremolo_rate(x: np.ndarray, sr: float) -> float:
    """Estimate tremolo rate (Hz) from the envelope's dominant low-frequency oscillation.

    Computes the FFT of the instantaneous amplitude and finds the peak
    frequency in the 3-12 Hz range (typical tremolo).
    """
    env = instantaneous_amplitude(x)
    if len(env) < 64:
        return 0.0
    env = env - np.mean(env)
    spectrum = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / sr)
    mask = (freqs >= 3.0) & (freqs <= 12.0)
    if not np.any(mask):
        return 0.0
    masked_spectrum = spectrum[mask]
    masked_freqs = freqs[mask]
    return float(masked_freqs[np.argmax(masked_spectrum)])


def tremolo_depth(x: np.ndarray) -> float:
    """Tremolo depth as the coefficient of variation of the envelope."""
    env = instantaneous_amplitude(x)
    if len(env) == 0:
        return 0.0
    mean_env = float(np.mean(env))
    if mean_env < 1e-12:
        return 0.0
    return float(np.std(env) / mean_env)


def envelope_centroid(x: np.ndarray, sr: float) -> float:
    """Temporal centroid of the amplitude envelope (in seconds).

    Analogous to spectral centroid but in the time domain — indicates
    where the "weight" of the sound's energy lies temporally.
    """
    env = instantaneous_amplitude(x)
    if len(env) == 0:
        return 0.0
    total = float(np.sum(env))
    if total < 1e-12:
        return 0.0
    t = np.arange(len(env)) / sr
    return float(np.sum(t * env) / total)


def attack_time(x: np.ndarray, sr: float, threshold: float = 0.9) -> float:
    """Time (seconds) for the envelope to reach ``threshold`` × peak."""
    env = instantaneous_amplitude(x)
    if len(env) == 0:
        return 0.0
    peak = float(np.max(env))
    if peak < 1e-12:
        return 0.0
    target = threshold * peak
    indices = np.where(env >= target)[0]
    if len(indices) == 0:
        return float(len(env)) / sr
    return float(indices[0]) / sr


def decay_time(x: np.ndarray, sr: float, threshold: float = 0.1) -> float:
    """Time (seconds) from envelope peak to dropping below ``threshold`` × peak."""
    env = instantaneous_amplitude(x)
    if len(env) == 0:
        return 0.0
    peak_idx = int(np.argmax(env))
    peak = float(env[peak_idx])
    if peak < 1e-12:
        return 0.0
    target = threshold * peak
    post_peak = env[peak_idx:]
    below = np.where(post_peak <= target)[0]
    if len(below) == 0:
        return float(len(post_peak)) / sr
    return float(below[0]) / sr
