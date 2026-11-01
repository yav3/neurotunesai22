"""Window functions for spectral analysis of musical signals."""
from __future__ import annotations

import math
from enum import Enum

import numpy as np


class WindowType(Enum):
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    KAISER = "kaiser"
    FLATTOP = "flattop"
    BARTLETT = "bartlett"
    RECTANGULAR = "rectangular"


def window_function(wtype: WindowType, n: int, beta: float = 8.6) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.float64)
    if wtype == WindowType.HANN:
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1))) if n > 1 else np.ones(1)
    if wtype == WindowType.HAMMING:
        return (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))) if n > 1 else np.ones(1)
    if wtype == WindowType.BLACKMAN:
        k = np.arange(n)
        return (0.42 - 0.5 * np.cos(2 * np.pi * k / (n - 1))
                + 0.08 * np.cos(4 * np.pi * k / (n - 1))) if n > 1 else np.ones(1)
    if wtype == WindowType.KAISER:
        return np.kaiser(n, beta)
    if wtype == WindowType.FLATTOP:
        k = np.arange(n)
        d = n - 1 if n > 1 else 1
        a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
        return sum((-1) ** i * a[i] * np.cos(2 * np.pi * i * k / d) for i in range(5))
    if wtype == WindowType.BARTLETT:
        return 1 - np.abs(2 * np.arange(n) / (n - 1) - 1) if n > 1 else np.ones(1)
    return np.ones(n)


def overlap_add_window(wtype: WindowType, n: int, hop: int) -> np.ndarray:
    w = window_function(wtype, n)
    if hop <= 0 or n <= 0:
        return w
    n_frames = math.ceil(n / hop) + 1
    out = np.zeros(n + (n_frames - 1) * hop)
    for i in range(n_frames):
        start = i * hop
        end = start + n
        if end <= len(out):
            out[start:end] += w
    return out[:n + (n_frames - 1) * hop]


def equivalent_noise_bandwidth(w: np.ndarray) -> float:
    if len(w) == 0 or np.sum(w) == 0:
        return 0.0
    return len(w) * np.sum(w ** 2) / np.sum(w) ** 2


def scalloping_loss(w: np.ndarray) -> float:
    if len(w) == 0:
        return 0.0
    n = len(w)
    W = np.fft.fft(w, 4 * n)
    mag = np.abs(W)
    peak = mag[0]
    if peak == 0:
        return 0.0
    half_bin = mag[2]
    ratio = half_bin / peak
    return 20.0 * np.log10(max(ratio, 1e-20))


def coherent_gain(w: np.ndarray) -> float:
    if len(w) == 0:
        return 0.0
    return float(np.sum(w) / len(w))


def cola_check(wtype: WindowType, n: int, hop: int, tol: float = 1e-6) -> bool:
    ola = overlap_add_window(wtype, n, hop)
    trim = n
    mid = ola[n // 2: n // 2 + trim] if len(ola) >= n // 2 + trim else ola
    if len(mid) == 0:
        return False
    return float(np.max(np.abs(mid - mid[0]))) < tol


def optimal_hop(wtype: WindowType, n: int) -> int:
    if wtype in (WindowType.HANN, WindowType.HAMMING):
        return n // 4
    if wtype == WindowType.BLACKMAN:
        return n // 4
    if wtype == WindowType.RECTANGULAR:
        return n // 2
    return n // 4


def apply_window(signal: np.ndarray, wtype: WindowType, center: int, n: int) -> np.ndarray:
    w = window_function(wtype, n)
    start = center - n // 2
    end = start + n
    pad_left = max(0, -start)
    pad_right = max(0, end - len(signal))
    seg = signal[max(0, start):min(len(signal), end)]
    if pad_left > 0 or pad_right > 0:
        seg = np.pad(seg, (pad_left, pad_right))
    return seg * w
