"""Harmonic Product Spectrum and multi-method pitch estimation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchEstimate:
    frequency_hz: float
    confidence: float
    method: str


def harmonic_product_spectrum(
    signal: np.ndarray,
    sr: int,
    n_harmonics: int = 5,
    n_fft: int = 4096,
    fmin: float = 50.0,
    fmax: float = 2000.0,
) -> PitchEstimate:
    windowed = signal[:n_fft] * np.hanning(min(len(signal), n_fft))
    if len(windowed) < n_fft:
        windowed = np.pad(windowed, (0, n_fft - len(windowed)))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    imin = max(1, int(fmin * n_fft / sr))
    imax = min(len(spectrum) - 1, int(fmax * n_fft / sr))

    hps = spectrum[imin:imax + 1].copy()
    for h in range(2, n_harmonics + 1):
        decimated = spectrum[imin * h::h]
        length = min(len(hps), len(decimated))
        hps[:length] *= decimated[:length]

    if len(hps) == 0 or hps.max() < 1e-30:
        return PitchEstimate(0.0, 0.0, "hps")

    peak_idx = np.argmax(hps)
    freq = freqs[peak_idx + imin]
    confidence = float(hps[peak_idx] / (hps.sum() + 1e-30))
    return PitchEstimate(freq, min(confidence, 1.0), "hps")


def autocorrelation_pitch(
    signal: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 2000.0,
) -> PitchEstimate:
    n = len(signal)
    if n < 2:
        return PitchEstimate(0.0, 0.0, "autocorrelation")

    lag_min = max(1, int(sr / fmax))
    lag_max = min(n - 1, int(sr / fmin))

    x = signal - signal.mean()
    norm = np.sqrt(np.sum(x ** 2))
    if norm < 1e-15:
        return PitchEstimate(0.0, 0.0, "autocorrelation")

    autocorr = np.correlate(x, x, mode="full")
    autocorr = autocorr[n - 1:]
    autocorr /= autocorr[0] + 1e-30

    search = autocorr[lag_min:lag_max + 1]
    if len(search) == 0:
        return PitchEstimate(0.0, 0.0, "autocorrelation")

    peak_idx = np.argmax(search)
    lag = peak_idx + lag_min
    freq = sr / lag if lag > 0 else 0.0
    confidence = float(search[peak_idx])
    return PitchEstimate(freq, max(0.0, min(confidence, 1.0)), "autocorrelation")


def cepstral_pitch(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    fmin: float = 50.0,
    fmax: float = 2000.0,
) -> PitchEstimate:
    windowed = signal[:n_fft] * np.hanning(min(len(signal), n_fft))
    if len(windowed) < n_fft:
        windowed = np.pad(windowed, (0, n_fft - len(windowed)))

    spectrum = np.abs(np.fft.rfft(windowed))
    log_spectrum = np.log(spectrum + 1e-30)
    cepstrum = np.abs(np.fft.irfft(log_spectrum))

    quefrency_min = max(1, int(sr / fmax))
    quefrency_max = min(len(cepstrum) - 1, int(sr / fmin))

    search = cepstrum[quefrency_min:quefrency_max + 1]
    if len(search) == 0:
        return PitchEstimate(0.0, 0.0, "cepstral")

    peak_idx = np.argmax(search)
    quefrency = peak_idx + quefrency_min
    freq = sr / quefrency if quefrency > 0 else 0.0
    confidence = float(search[peak_idx] / (np.max(cepstrum[:quefrency_min]) + 1e-30))
    return PitchEstimate(freq, min(max(0.0, confidence), 1.0), "cepstral")


def multi_pitch_estimate(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    max_pitches: int = 3,
    fmin: float = 80.0,
    fmax: float = 1500.0,
    residual_threshold: float = 0.1,
) -> list[PitchEstimate]:
    windowed = signal[:n_fft] * np.hanning(min(len(signal), n_fft))
    if len(windowed) < n_fft:
        windowed = np.pad(windowed, (0, n_fft - len(windowed)))

    spectrum = np.abs(np.fft.rfft(windowed)).copy()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    original_energy = np.sum(spectrum ** 2)
    if original_energy < 1e-30:
        return []

    pitches: list[PitchEstimate] = []
    for _ in range(max_pitches):
        imin = max(1, int(fmin * n_fft / sr))
        imax = min(len(spectrum) - 1, int(fmax * n_fft / sr))
        hps = spectrum[imin:imax + 1].copy()
        for h in range(2, 4):
            decimated = spectrum[imin * h::h]
            length = min(len(hps), len(decimated))
            hps[:length] *= decimated[:length]

        if hps.max() < 1e-30:
            break

        peak_idx = np.argmax(hps)
        f0 = freqs[peak_idx + imin]
        conf = float(hps[peak_idx] / (hps.sum() + 1e-30))
        pitches.append(PitchEstimate(f0, min(conf, 1.0), "multi_hps"))

        for h in range(1, 6):
            center = int(f0 * h * n_fft / sr)
            width = max(3, int(0.02 * center))
            lo = max(0, center - width)
            hi = min(len(spectrum), center + width + 1)
            spectrum[lo:hi] *= 0.1

        residual_energy = np.sum(spectrum ** 2)
        if residual_energy / original_energy < residual_threshold:
            break

    return pitches


def consensus_pitch(
    signal: np.ndarray,
    sr: int,
    tolerance_cents: float = 50.0,
) -> PitchEstimate:
    estimates = [
        harmonic_product_spectrum(signal, sr),
        autocorrelation_pitch(signal, sr),
        cepstral_pitch(signal, sr),
    ]
    valid = [e for e in estimates if e.frequency_hz > 0]
    if not valid:
        return PitchEstimate(0.0, 0.0, "consensus")

    freqs = np.array([e.frequency_hz for e in valid])
    confs = np.array([e.confidence for e in valid])
    weighted_freq = float(np.average(freqs, weights=confs))

    agreement = 0
    for f in freqs:
        if f > 0 and weighted_freq > 0:
            cents = abs(1200.0 * np.log2(f / weighted_freq))
            if cents < tolerance_cents:
                agreement += 1

    consensus_conf = float(agreement / len(valid)) * float(confs.max())
    return PitchEstimate(weighted_freq, min(consensus_conf, 1.0), "consensus")
