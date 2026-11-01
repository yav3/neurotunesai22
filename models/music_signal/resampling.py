"""Sample rate conversion for musical audio."""
from __future__ import annotations

import math

import numpy as np


def sample_rate_ratio(sr_in: int, sr_out: int) -> tuple[int, int]:
    g = math.gcd(sr_in, sr_out)
    return sr_out // g, sr_in // g


def _sinc_kernel(length: int, cutoff: float) -> np.ndarray:
    n = np.arange(length) - (length - 1) / 2
    h = np.where(n == 0, 2 * cutoff, np.sin(2 * np.pi * cutoff * n) / (np.pi * n))
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(length) / (length - 1))
    return h * w


def resample_sinc(
    signal: np.ndarray, sr_in: int, sr_out: int, quality: str = "medium",
) -> np.ndarray:
    if sr_in == sr_out:
        return signal.copy()
    quality_map = {"low": 32, "medium": 64, "high": 128}
    filt_len = quality_map.get(quality, 64)
    up, down = sample_rate_ratio(sr_in, sr_out)
    n_out = int(math.ceil(len(signal) * up / down))
    cutoff = min(0.5 / up, 0.5 / down) * 0.95
    kernel = _sinc_kernel(filt_len * up, cutoff) * up
    upsampled = np.zeros(len(signal) * up)
    upsampled[::up] = signal
    padded = np.pad(upsampled, filt_len * up // 2)
    filtered = np.convolve(padded, kernel, mode="same")[: len(upsampled)]
    return filtered[::down][:n_out]


def decimate(signal: np.ndarray, factor: int, filter_order: int = 8) -> np.ndarray:
    if factor <= 1:
        return signal.copy()
    cutoff = 0.8 / factor
    filt_len = filter_order * factor * 2 + 1
    kernel = _sinc_kernel(filt_len, cutoff)
    kernel /= np.sum(kernel)
    filtered = np.convolve(signal, kernel, mode="same")
    return filtered[::factor]


def interpolate_linear(signal: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or len(signal) < 2:
        return signal.copy()
    n_out = (len(signal) - 1) * factor + 1
    out = np.zeros(n_out)
    for i in range(len(signal) - 1):
        for j in range(factor):
            t = j / factor
            out[i * factor + j] = signal[i] * (1 - t) + signal[i + 1] * t
    out[-1] = signal[-1]
    return out


def interpolate_cubic(signal: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or len(signal) < 4:
        return interpolate_linear(signal, factor)
    n_out = (len(signal) - 1) * factor + 1
    out = np.zeros(n_out)
    for i in range(len(signal) - 1):
        p0 = signal[max(0, i - 1)]
        p1 = signal[i]
        p2 = signal[min(len(signal) - 1, i + 1)]
        p3 = signal[min(len(signal) - 1, i + 2)]
        for j in range(factor):
            t = j / factor
            t2 = t * t
            t3 = t2 * t
            out[i * factor + j] = 0.5 * (
                (2 * p1) + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
    out[-1] = signal[-1]
    return out


def fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    int_delay = int(math.floor(delay_samples))
    frac = delay_samples - int_delay
    if frac < 1e-10:
        return np.roll(signal, int_delay)
    h = np.array([
        -frac * (frac - 1) * (frac - 2) / 6,
        (frac + 1) * (frac - 1) * (frac - 2) / 2,
        -(frac + 1) * frac * (frac - 2) / 2,
        (frac + 1) * frac * (frac - 1) / 6,
    ])
    delayed = np.convolve(signal, h, mode="full")[:len(signal)]
    return np.roll(delayed, int_delay)
