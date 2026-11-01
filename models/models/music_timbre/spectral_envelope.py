"""Spectral envelope estimation — LPC, cepstral smoothing, true envelope."""
from __future__ import annotations

import math
import numpy as np


def autocorrelation(signal: np.ndarray, order: int) -> np.ndarray:
    """Biased autocorrelation for LPC analysis."""
    n = len(signal)
    r = np.zeros(order + 1)
    for k in range(order + 1):
        r[k] = np.sum(signal[:n - k] * signal[k:])
    r /= n
    return r


def levinson_durbin(r: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Levinson-Durbin recursion for LPC coefficients.

    Returns (lpc_coefficients, prediction_error_power).
    """
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    if e <= 0:
        return a[:order], 0.0
    for i in range(1, order + 1):
        lambda_val = 0.0
        for j in range(1, i):
            lambda_val += a[j] * r[i - j]
        lambda_val = -(r[i] + lambda_val) / e

        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + lambda_val * a[i - j]
        a_new[i] = lambda_val
        a = a_new
        e *= (1.0 - lambda_val * lambda_val)
        if e <= 0:
            break
    return a[1:order + 1], e


def lpc_envelope(signal: np.ndarray, order: int = 16, n_fft: int = 512) -> np.ndarray:
    """Compute spectral envelope via LPC all-pole model.

    Returns the magnitude envelope at n_fft//2+1 frequency bins.
    """
    r = autocorrelation(signal, order)
    coeffs, gain = levinson_durbin(r, order)
    if gain <= 0:
        gain = 1e-10
    freqs = np.linspace(0, np.pi, n_fft // 2 + 1)
    envelope = np.zeros(len(freqs))
    for k, w in enumerate(freqs):
        denom = 1.0
        for i, c in enumerate(coeffs):
            denom += c * np.exp(-1j * (i + 1) * w)
        envelope[k] = math.sqrt(gain) / max(abs(denom), 1e-30)
    return envelope


def cepstral_smoothing(log_spectrum: np.ndarray, n_keep: int = 30) -> np.ndarray:
    """Smooth a log-magnitude spectrum by truncating its cepstrum."""
    cepstrum = np.fft.irfft(log_spectrum)
    cepstrum[n_keep:-n_keep] = 0.0
    return np.fft.rfft(cepstrum).real


def true_envelope(log_spectrum: np.ndarray, iterations: int = 20,
                  order: int = 40) -> np.ndarray:
    """Iterative true envelope estimation.

    Repeatedly lifts the cepstral-smoothed envelope to never fall below
    the original spectrum.
    """
    env = log_spectrum.copy()
    for _ in range(iterations):
        smoothed = cepstral_smoothing(env, n_keep=order)
        env = np.maximum(log_spectrum, smoothed)
    return env


def formant_frequencies(envelope_db: np.ndarray, sr: float = 44100.0) -> list[float]:
    """Extract formant frequencies from spectral envelope peaks."""
    n = len(envelope_db)
    freqs_hz = np.linspace(0, sr / 2, n)
    peaks: list[float] = []
    for i in range(1, n - 1):
        if envelope_db[i] > envelope_db[i - 1] and envelope_db[i] > envelope_db[i + 1]:
            if freqs_hz[i] > 80.0:
                peaks.append(float(freqs_hz[i]))
    return peaks


def spectral_tilt(log_spectrum: np.ndarray, sr: float = 44100.0) -> float:
    """Spectral tilt (dB/octave) via linear regression on log-frequency axis."""
    n = len(log_spectrum)
    freqs = np.linspace(1, sr / 2, n)
    log_freqs = np.log2(freqs)
    valid = np.isfinite(log_spectrum) & np.isfinite(log_freqs)
    if valid.sum() < 2:
        return 0.0
    x = log_freqs[valid]
    y = log_spectrum[valid]
    slope = float(np.polyfit(x, y, 1)[0])
    return slope
