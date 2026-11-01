"""Mel-Frequency Cepstral Coefficients (MFCC) computation."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MFCCResult:
    """MFCC extraction result."""
    coefficients: np.ndarray
    n_mfcc: int
    sr: int
    hop_size: int
    n_fft: int


def dct_ii(x: np.ndarray) -> np.ndarray:
    """Type-II Discrete Cosine Transform (orthonormal)."""
    n = len(x)
    result = np.zeros(n)
    for k in range(n):
        s = 0.0
        for i in range(n):
            s += x[i] * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        result[k] = s
    result *= np.sqrt(2.0 / n)
    result[0] *= 1.0 / np.sqrt(2.0)
    return result


def compute_mfcc(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_size: int = 512,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> MFCCResult:
    """Compute MFCCs from a time-domain signal.

    Pipeline: signal -> windowed frames -> FFT -> power spectrum
    -> mel filterbank -> log -> DCT -> first n_mfcc coefficients.
    """
    from .spectral_tools import stft as _stft, mel_filterbank as _mel_fb, hz_to_mel

    if fmax is None:
        fmax = sr / 2.0

    spec = _stft(signal, n_fft, hop_size)
    power = np.abs(spec) ** 2

    mel_fb = _mel_fb(n_mels, n_fft, sr, fmin, fmax)

    n_freq = power.shape[0]
    fb_rows = mel_fb.shape[1] if mel_fb.ndim == 2 else len(mel_fb)
    if mel_fb.ndim == 2:
        fb = mel_fb
    else:
        fb = mel_fb.reshape(1, -1)

    min_dim = min(fb.shape[1], n_freq)
    mel_spec = fb[:, :min_dim] @ power[:min_dim, :]

    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    n_frames = log_mel.shape[1]
    mfcc = np.zeros((n_mfcc, n_frames))
    for t in range(n_frames):
        dct_coeffs = dct_ii(log_mel[:, t])
        mfcc[:, t] = dct_coeffs[:n_mfcc]

    return MFCCResult(
        coefficients=mfcc,
        n_mfcc=n_mfcc,
        sr=sr,
        hop_size=hop_size,
        n_fft=n_fft,
    )


def mfcc_delta(mfcc: np.ndarray, width: int = 9) -> np.ndarray:
    """Compute delta (first derivative) of MFCC features.

    Uses linear regression over a window of `width` frames.
    """
    n_coeffs, n_frames = mfcc.shape
    half = width // 2
    padded = np.pad(mfcc, ((0, 0), (half, half)), mode="edge")

    delta = np.zeros_like(mfcc)
    denom = 2.0 * sum(n ** 2 for n in range(1, half + 1))
    if denom == 0:
        return delta

    for t in range(n_frames):
        for n in range(1, half + 1):
            delta[:, t] += n * (padded[:, t + half + n] - padded[:, t + half - n])
        delta[:, t] /= denom

    return delta


def mfcc_delta_delta(mfcc: np.ndarray, width: int = 9) -> np.ndarray:
    """Compute delta-delta (second derivative / acceleration) of MFCC features."""
    d1 = mfcc_delta(mfcc, width)
    return mfcc_delta(d1, width)


def lifter(mfcc: np.ndarray, factor: int = 22) -> np.ndarray:
    """Apply sinusoidal liftering to MFCCs for ASR-style normalisation."""
    n_coeffs = mfcc.shape[0]
    lift = 1.0 + (factor / 2.0) * np.sin(np.pi * np.arange(n_coeffs) / factor)
    return mfcc * lift[:, None]
