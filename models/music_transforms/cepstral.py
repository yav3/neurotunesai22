"""Cepstral analysis: MFCCs, LPCCs, cepstral liftering, delta features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dct_ii(x: np.ndarray) -> np.ndarray:
    N = len(x)
    n = np.arange(N)
    result = np.empty(N)
    for k in range(N):
        result[k] = np.sum(x * np.cos(np.pi * k * (2 * n + 1) / (2 * N)))
    return result


def mel_frequency(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def inv_mel_frequency(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank_matrix(
    n_filters: int,
    n_fft: int,
    sr: float = 22050.0,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    mel_min = mel_frequency(fmin)
    mel_max = mel_frequency(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = np.array([inv_mel_frequency(m) for m in mel_points])
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        for j in range(int(bins[i]), int(bins[i + 1])):
            if bins[i + 1] > bins[i]:
                fb[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
        for j in range(int(bins[i + 1]), int(bins[i + 2])):
            if bins[i + 2] > bins[i + 1]:
                fb[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])
    return fb


def compute_mfcc(
    signal: np.ndarray,
    sr: float = 22050.0,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 40,
) -> np.ndarray:
    n = len(signal)
    n_frames = max(1, 1 + (n - n_fft) // hop)
    window = np.hanning(n_fft)
    fb = mel_filterbank_matrix(n_mels, n_fft, sr)

    mfcc = np.zeros((n_frames, n_mfcc))
    for t in range(n_frames):
        start = t * hop
        frame = np.zeros(n_fft)
        end = min(start + n_fft, n)
        frame[: end - start] = signal[start:end]
        frame *= window
        spec = np.abs(np.fft.rfft(frame)) ** 2
        mel_spec = fb @ spec
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel = np.log(mel_spec)
        cepstrum = dct_ii(log_mel)
        mfcc[t, :] = cepstrum[:n_mfcc]
    return mfcc


def lifter(cepstral: np.ndarray, L: int = 22) -> np.ndarray:
    n_coeff = cepstral.shape[-1]
    n = np.arange(n_coeff)
    lift = 1.0 + (L / 2.0) * np.sin(np.pi * n / L)
    return cepstral * lift


def delta_features(features: np.ndarray, width: int = 2) -> np.ndarray:
    n_frames = features.shape[0]
    if n_frames < 2:
        return np.zeros_like(features)
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(i * i for i in range(1, width + 1))
    if denom == 0:
        return np.zeros_like(features)
    delta = np.zeros_like(features)
    for t in range(n_frames):
        for i in range(1, width + 1):
            delta[t] += i * (padded[t + width + i] - padded[t + width - i])
        delta[t] /= denom
    return delta


def delta_delta_features(features: np.ndarray, width: int = 2) -> np.ndarray:
    d = delta_features(features, width)
    return delta_features(d, width)


@dataclass(frozen=True)
class MFCCResult:
    mfcc: np.ndarray
    delta: np.ndarray
    delta_delta: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.mfcc.shape[0]

    @property
    def full_features(self) -> np.ndarray:
        return np.concatenate([self.mfcc, self.delta, self.delta_delta], axis=1)


def compute_mfcc_full(
    signal: np.ndarray,
    sr: float = 22050.0,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 40,
    delta_width: int = 2,
) -> MFCCResult:
    mfcc = compute_mfcc(signal, sr, n_mfcc, n_fft, hop, n_mels)
    d = delta_features(mfcc, delta_width)
    dd = delta_delta_features(mfcc, delta_width)
    return MFCCResult(mfcc=mfcc, delta=d, delta_delta=dd)


def lpc_coefficients(signal: np.ndarray, order: int = 12) -> np.ndarray:
    n = len(signal)
    if n <= order:
        return np.zeros(order + 1)
    r = np.correlate(signal, signal, mode="full")
    r = r[n - 1 :]
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            if abs(i - j) < len(r):
                R[i, j] = r[abs(i - j)]
    rhs = r[1 : order + 1]
    try:
        a = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        a = np.zeros(order)
    return np.concatenate([[1.0], -a])


def lpc_to_cepstrum(lpc: np.ndarray, n_cepstral: int = 13) -> np.ndarray:
    p = len(lpc) - 1
    c = np.zeros(n_cepstral)
    if p == 0:
        return c
    c[0] = np.log(max(abs(lpc[0]), 1e-10))
    for n in range(1, n_cepstral):
        s = 0.0
        for k in range(1, min(n, p) + 1):
            if k < len(lpc):
                s += k * (-lpc[k]) * (c[n - k] if n - k < len(c) else 0.0)
        c[n] = s / n + (-lpc[n] if n < len(lpc) else 0.0)
    return c
