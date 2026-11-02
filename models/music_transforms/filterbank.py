"""Filterbank construction: mel, Bark, ERB, gammatone, octave bands."""
from __future__ import annotations

import numpy as np


def hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_filters: int = 40, n_fft: int = 2048,
                   sr: float = 44100.0, f_min: float = 0.0,
                   f_max: float | None = None) -> np.ndarray:
    if f_max is None:
        f_max = sr / 2.0
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_filters, n_bins))
    for i in range(n_filters):
        lo, center, hi = bins[i], bins[i + 1], bins[i + 2]
        for j in range(lo, min(center, n_bins)):
            if center > lo:
                fb[i, j] = (j - lo) / (center - lo)
        for j in range(center, min(hi, n_bins)):
            if hi > center:
                fb[i, j] = (hi - j) / (hi - center)
    return fb


def hz_to_bark(f: float) -> float:
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def bark_to_hz(z: float) -> float:
    lo, hi = 0.0, 25000.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if hz_to_bark(mid) < z:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bark_filterbank(n_filters: int = 24, n_fft: int = 2048,
                    sr: float = 44100.0) -> np.ndarray:
    f_max = sr / 2.0
    bark_max = hz_to_bark(f_max)
    bark_points = np.linspace(0, bark_max, n_filters + 2)
    hz_points = np.array([bark_to_hz(b) for b in bark_points])
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_filters, n_bins))
    for i in range(n_filters):
        lo, center, hi = bins[i], bins[i + 1], bins[i + 2]
        for j in range(lo, min(center, n_bins)):
            if center > lo:
                fb[i, j] = (j - lo) / (center - lo)
        for j in range(center, min(hi, n_bins)):
            if hi > center:
                fb[i, j] = (hi - j) / (hi - center)
    return fb


def hz_to_erb(f: float) -> float:
    return 21.4 * np.log10(1.0 + 0.00437 * f)


def erb_to_hz(e: float) -> float:
    return (10.0 ** (e / 21.4) - 1.0) / 0.00437


def erb_filterbank(n_filters: int = 32, n_fft: int = 2048,
                   sr: float = 44100.0, f_min: float = 50.0,
                   f_max: float | None = None) -> np.ndarray:
    if f_max is None:
        f_max = sr / 2.0
    erb_min = hz_to_erb(f_min)
    erb_max = hz_to_erb(f_max)
    erb_points = np.linspace(erb_min, erb_max, n_filters + 2)
    hz_points = np.array([erb_to_hz(e) for e in erb_points])
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_filters, n_bins))
    for i in range(n_filters):
        lo, center, hi = bins[i], bins[i + 1], bins[i + 2]
        for j in range(lo, min(center, n_bins)):
            if center > lo:
                fb[i, j] = (j - lo) / (center - lo)
        for j in range(center, min(hi, n_bins)):
            if hi > center:
                fb[i, j] = (hi - j) / (hi - center)
    return fb


def gammatone_impulse_response(t: np.ndarray, fc: float, n: int = 4,
                                b: float | None = None) -> np.ndarray:
    if b is None:
        erb_w = 24.7 * (4.37 * fc / 1000 + 1)
        b = 2 * np.pi * 1.019 * erb_w
    return t ** (n - 1) * np.exp(-b * t) * np.cos(2 * np.pi * fc * t) * (t >= 0)


def gammatone_filterbank(n_filters: int = 64, sr: float = 44100.0,
                         f_min: float = 50.0, f_max: float = 8000.0,
                         filter_length: int = 1024) -> np.ndarray:
    erb_min = hz_to_erb(f_min)
    erb_max = hz_to_erb(f_max)
    center_freqs = np.array([erb_to_hz(e) for e in np.linspace(erb_min, erb_max, n_filters)])
    t = np.arange(filter_length) / sr
    fb = np.zeros((n_filters, filter_length))
    for i, fc in enumerate(center_freqs):
        ir = gammatone_impulse_response(t, fc)
        norm = np.sqrt(np.sum(ir ** 2) + 1e-10)
        fb[i] = ir / norm
    return fb


def octave_band_edges(n_bands: int = 10, f_ref: float = 1000.0) -> np.ndarray:
    bands = np.zeros((n_bands, 2))
    center_band = n_bands // 2
    for i in range(n_bands):
        offset = i - center_band
        fc = f_ref * 2.0 ** offset
        bands[i, 0] = fc / np.sqrt(2)
        bands[i, 1] = fc * np.sqrt(2)
    return bands


def third_octave_band_edges(n_bands: int = 30, f_ref: float = 1000.0) -> np.ndarray:
    bands = np.zeros((n_bands, 2))
    center_band = n_bands // 2
    for i in range(n_bands):
        offset = i - center_band
        fc = f_ref * 2.0 ** (offset / 3.0)
        bands[i, 0] = fc / 2.0 ** (1.0 / 6.0)
        bands[i, 1] = fc * 2.0 ** (1.0 / 6.0)
    return bands


def windowed_sinc_lowpass(cutoff: float, sr: float, n_taps: int = 101) -> np.ndarray:
    fc = cutoff / sr
    half = n_taps // 2
    t = np.arange(n_taps) - half
    h = np.zeros(n_taps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 2 * fc
        else:
            h[i] = np.sin(2 * np.pi * fc * ti) / (np.pi * ti)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n_taps) / (n_taps - 1))
    h *= window
    h /= np.sum(h)
    return h


def apply_filter(signal: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    n = len(coeffs)
    pad = np.pad(signal, (n // 2, n // 2), mode='reflect')
    out = np.convolve(pad, coeffs, mode='same')
    return out[n // 2:n // 2 + len(signal)]
