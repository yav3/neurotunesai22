"""Spectral envelope extraction and manipulation.

True envelope estimation via:
- Cepstral smoothing (liftered cepstrum)
- Linear prediction (LP/LPC)
- Discrete all-pole modeling
- Spectral peak interpolation
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SpectralEnvelope:
    """Estimated spectral envelope."""
    frequencies: np.ndarray
    magnitude_db: np.ndarray
    formants: np.ndarray  # formant frequencies in Hz
    bandwidths: np.ndarray  # formant bandwidths in Hz


def cepstral_envelope(magnitude_spectrum: np.ndarray, n_cepstral: int = 30,
                      sr: float = 22050) -> SpectralEnvelope:
    """Spectral envelope via cepstral smoothing (liftering).

    Low-quefrency cepstral coefficients capture the smooth envelope;
    high-quefrency coefficients carry the fine harmonic structure.
    Keeping only the first n_cepstral coefficients yields the envelope.
    """
    n = len(magnitude_spectrum)
    log_mag = np.log(np.maximum(magnitude_spectrum, 1e-10))
    cepstrum = np.fft.irfft(log_mag)
    lifter = np.zeros_like(cepstrum)
    lifter[:n_cepstral] = 1.0
    if len(lifter) > n_cepstral:
        lifter[n_cepstral] = 0.5  # smooth transition
    smoothed = np.fft.rfft(cepstrum * lifter).real
    smoothed_db = smoothed * 20 / np.log(10)
    freqs = np.linspace(0, sr / 2, len(smoothed_db))
    formants, bws = _find_formants(smoothed_db, freqs)
    return SpectralEnvelope(
        frequencies=freqs,
        magnitude_db=smoothed_db,
        formants=formants,
        bandwidths=bws,
    )


def lpc_envelope(signal_frame: np.ndarray, order: int = 16,
                 sr: float = 22050,
                 n_fft: int = 1024) -> SpectralEnvelope:
    """Spectral envelope via Linear Predictive Coding (Levinson-Durbin).

    Fits an all-pole model H(z) = 1/A(z) to the signal frame,
    then evaluates |H(e^{jω})| on the frequency axis.
    """
    frame = signal_frame * np.hanning(len(signal_frame))
    r = np.correlate(frame, frame, mode='full')
    mid = len(r) // 2
    acf = r[mid:mid + order + 1]
    a = _levinson_durbin(acf, order)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    omega = 2 * np.pi * freqs / sr
    H = np.zeros(len(omega), dtype=complex)
    for k, w in enumerate(omega):
        denom = 0.0 + 0j
        for i, ai in enumerate(a):
            denom += ai * np.exp(-1j * w * i)
        H[k] = 1.0 / (denom + 1e-10)
    mag_db = 20 * np.log10(np.abs(H) + 1e-10)
    formants, bws = _find_formants(mag_db, freqs)
    return SpectralEnvelope(
        frequencies=freqs,
        magnitude_db=mag_db,
        formants=formants,
        bandwidths=bws,
    )


def _levinson_durbin(acf: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin recursion for LP coefficients."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = acf[0]
    for i in range(1, order + 1):
        lam = 0.0
        for j in range(1, i):
            lam += a[j] * acf[i - j]
        lam = (acf[i] - lam) / (e + 1e-30)
        new_a = np.zeros(order + 1)
        new_a[0] = 1.0
        for j in range(1, i):
            new_a[j] = a[j] - lam * a[i - j]
        new_a[i] = -lam
        a = new_a
        e *= (1 - lam * lam)
        if e <= 0:
            break
    return a


def _find_formants(mag_db: np.ndarray, freqs: np.ndarray,
                   min_prominence: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Find formant peaks in a spectral envelope."""
    formant_freqs = []
    bandwidths = []
    for i in range(1, len(mag_db) - 1):
        if mag_db[i] > mag_db[i-1] and mag_db[i] > mag_db[i+1]:
            left_valley = mag_db[i]
            for j in range(i-1, -1, -1):
                if mag_db[j] < left_valley:
                    left_valley = mag_db[j]
                if mag_db[j] > mag_db[i]:
                    break
            right_valley = mag_db[i]
            for j in range(i+1, len(mag_db)):
                if mag_db[j] < right_valley:
                    right_valley = mag_db[j]
                if mag_db[j] > mag_db[i]:
                    break
            prominence = mag_db[i] - max(left_valley, right_valley)
            if prominence > min_prominence and freqs[i] > 80:
                formant_freqs.append(freqs[i])
                half_power = mag_db[i] - 3.0
                lo = freqs[max(0, i-1)]
                hi = freqs[min(len(freqs)-1, i+1)]
                for j in range(i, -1, -1):
                    if mag_db[j] < half_power:
                        lo = freqs[j]
                        break
                for j in range(i, len(mag_db)):
                    if mag_db[j] < half_power:
                        hi = freqs[j]
                        break
                bandwidths.append(hi - lo)
    return np.array(formant_freqs), np.array(bandwidths)


def spectral_tilt(magnitude_spectrum: np.ndarray, sr: float = 22050) -> float:
    """Spectral tilt: slope of linear regression on log-magnitude vs log-frequency.

    Negative tilt = energy concentrated at low frequencies (typical speech/music).
    """
    n = len(magnitude_spectrum)
    freqs = np.linspace(1, sr / 2, n)  # start at 1 to avoid log(0)
    log_freq = np.log(freqs)
    log_mag = np.log(np.maximum(magnitude_spectrum, 1e-10))
    A = np.vstack([log_freq, np.ones(n)]).T
    slope, _ = np.linalg.lstsq(A, log_mag, rcond=None)[0][:2]
    return float(slope)


def spectral_flatness_per_band(magnitude_spectrum: np.ndarray,
                               n_bands: int = 8) -> np.ndarray:
    """Spectral flatness (Wiener entropy) per frequency band.

    SFM = geometric_mean / arithmetic_mean.
    1.0 = white noise, 0.0 = pure tone.
    """
    n = len(magnitude_spectrum)
    band_size = max(1, n // n_bands)
    flatness = np.zeros(n_bands)
    for b in range(n_bands):
        lo = b * band_size
        hi = min((b + 1) * band_size, n)
        band = magnitude_spectrum[lo:hi]
        if len(band) == 0 or np.all(band < 1e-10):
            flatness[b] = 0.0
            continue
        band = np.maximum(band, 1e-10)
        geo_mean = np.exp(np.mean(np.log(band)))
        arith_mean = np.mean(band)
        flatness[b] = geo_mean / (arith_mean + 1e-10)
    return flatness


def envelope_interpolation(env1: SpectralEnvelope, env2: SpectralEnvelope,
                           alpha: float) -> np.ndarray:
    """Interpolate between two spectral envelopes (morphing).

    alpha = 0 → env1, alpha = 1 → env2. Linear in dB domain.
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    n = min(len(env1.magnitude_db), len(env2.magnitude_db))
    return (1 - alpha) * env1.magnitude_db[:n] + alpha * env2.magnitude_db[:n]


def formant_shift(envelope: SpectralEnvelope,
                  shift_factor: float) -> SpectralEnvelope:
    """Shift formant frequencies by a multiplicative factor.

    Useful for voice transformation (e.g., shift_factor > 1 = child voice).
    Implemented by resampling the envelope in the frequency domain.
    """
    new_freqs = envelope.frequencies / shift_factor
    new_mag = np.interp(envelope.frequencies, new_freqs, envelope.magnitude_db,
                        left=envelope.magnitude_db[0],
                        right=envelope.magnitude_db[-1])
    new_formants = envelope.formants * shift_factor
    return SpectralEnvelope(
        frequencies=envelope.frequencies,
        magnitude_db=new_mag,
        formants=new_formants,
        bandwidths=envelope.bandwidths * shift_factor,
    )
