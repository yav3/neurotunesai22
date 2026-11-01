"""Spectral analysis: STFT, mel filterbank, CQT, chromagram, spectral descriptors."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def stft(signal: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
         window: str = "hann") -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if window == "hann":
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))
    elif window == "hamming":
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    else:
        w = np.ones(n_fft)
    pad_len = n_fft // 2
    padded = np.pad(signal, (pad_len, pad_len), mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    frames = np.stack([padded[i * hop_length:i * hop_length + n_fft] * w
                       for i in range(n_frames)])
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def istft(S: np.ndarray, hop_length: int = 512, window: str = "hann") -> np.ndarray:
    n_fft = 2 * (S.shape[0] - 1)
    n_frames = S.shape[1]
    if window == "hann":
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))
    elif window == "hamming":
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    else:
        w = np.ones(n_fft)
    out_len = n_fft + hop_length * (n_frames - 1)
    output = np.zeros(out_len)
    window_sum = np.zeros(out_len)
    for i in range(n_frames):
        frame = np.fft.irfft(S[:, i], n=n_fft) * w
        start = i * hop_length
        output[start:start + n_fft] += frame
        window_sum[start:start + n_fft] += w ** 2
    nonzero = window_sum > 1e-10
    output[nonzero] /= window_sum[nonzero]
    return output


def mel_filterbank(n_filters: int, n_fft: int, sample_rate: int,
                   f_min: float = 0.0, f_max: float | None = None) -> np.ndarray:
    if f_max is None:
        f_max = sample_rate / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        for j in range(bins[i], bins[i + 1]):
            if bins[i + 1] > bins[i]:
                fb[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
        for j in range(bins[i + 1], bins[i + 2]):
            if bins[i + 2] > bins[i + 1]:
                fb[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])
    return fb


def mel_spectrogram(signal: np.ndarray, sample_rate: int, n_fft: int = 2048,
                    hop_length: int = 512, n_mels: int = 128) -> np.ndarray:
    S = stft(signal, n_fft=n_fft, hop_length=hop_length)
    power = np.abs(S) ** 2
    fb = mel_filterbank(n_mels, n_fft, sample_rate)
    return fb @ power


def cqt_frequencies(n_bins: int, f_min: float = 32.7,
                    bins_per_octave: int = 12) -> np.ndarray:
    return f_min * 2.0 ** (np.arange(n_bins) / bins_per_octave)


def constant_q_transform(signal: np.ndarray, sample_rate: int,
                         n_bins: int = 84, f_min: float = 32.7,
                         bins_per_octave: int = 12) -> np.ndarray:
    freqs = cqt_frequencies(n_bins, f_min, bins_per_octave)
    hop = 512
    n_frames = max(1, len(signal) // hop)
    cqt = np.zeros((n_bins, n_frames), dtype=complex)
    for k, fk in enumerate(freqs):
        nk = max(4, int(sample_rate / fk * bins_per_octave))
        nk = min(nk, len(signal))
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(nk) / nk)) / nk
        for t in range(n_frames):
            start = t * hop
            end = start + nk
            if end > len(signal):
                break
            seg = signal[start:end] * w
            cqt[k, t] = np.sum(seg * np.exp(-2j * np.pi * fk * np.arange(nk) / sample_rate))
    return cqt


def chromagram(signal: np.ndarray, sample_rate: int,
               n_chroma: int = 12, n_fft: int = 2048,
               hop_length: int = 512) -> np.ndarray:
    S = np.abs(stft(signal, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    chroma_map = np.zeros((n_chroma, len(freqs)))
    for i, f in enumerate(freqs):
        if f > 0:
            pitch_class = int(round(12.0 * np.log2(f / 440.0))) % 12
            chroma_map[pitch_class, i] += 1.0
    chroma = chroma_map @ S
    norms = np.maximum(chroma.max(axis=0, keepdims=True), 1e-10)
    return chroma / norms


@dataclass
class SpectralDescriptors:
    centroid: float
    spread: float
    flatness: float
    rolloff: float
    flux: float


def spectral_centroid(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    total = np.sum(magnitude)
    if total < 1e-30:
        return 0.0
    return float(np.sum(freqs * magnitude) / total)


def spectral_spread(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    centroid = spectral_centroid(magnitude, freqs)
    total = np.sum(magnitude)
    if total < 1e-30:
        return 0.0
    return float(np.sqrt(np.sum(magnitude * (freqs - centroid) ** 2) / total))


def spectral_flatness(magnitude: np.ndarray) -> float:
    magnitude = np.maximum(magnitude, 1e-30)
    geo_mean = np.exp(np.mean(np.log(magnitude)))
    arith_mean = np.mean(magnitude)
    if arith_mean < 1e-30:
        return 0.0
    return float(geo_mean / arith_mean)


def spectral_rolloff(magnitude: np.ndarray, freqs: np.ndarray,
                     threshold: float = 0.85) -> float:
    total = np.sum(magnitude)
    if total < 1e-30:
        return 0.0
    cumsum = np.cumsum(magnitude)
    idx = np.searchsorted(cumsum, threshold * total)
    return float(freqs[min(idx, len(freqs) - 1)])


def spectral_flux(current: np.ndarray, previous: np.ndarray) -> float:
    diff = np.maximum(current - previous, 0.0)
    return float(np.sum(diff ** 2))


def compute_spectral_descriptors(magnitude: np.ndarray, freqs: np.ndarray,
                                 prev_magnitude: np.ndarray | None = None) -> SpectralDescriptors:
    flux = spectral_flux(magnitude, prev_magnitude) if prev_magnitude is not None else 0.0
    return SpectralDescriptors(
        centroid=spectral_centroid(magnitude, freqs),
        spread=spectral_spread(magnitude, freqs),
        flatness=spectral_flatness(magnitude),
        rolloff=spectral_rolloff(magnitude, freqs),
        flux=flux,
    )
