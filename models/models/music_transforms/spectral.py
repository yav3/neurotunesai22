"""Spectral transforms: STFT, CQT, mel-spectrogram, chromagram."""
from __future__ import annotations

import numpy as np
from typing import Literal


def stft(
    signal: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
    window: Literal["hann", "hamming", "blackman"] = "hann",
) -> np.ndarray:
    if window == "hann":
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
    elif window == "hamming":
        w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    elif window == "blackman":
        w = (0.42 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
             + 0.08 * np.cos(4 * np.pi * np.arange(n_fft) / n_fft))
    else:
        w = np.ones(n_fft)

    pad_len = n_fft // 2
    padded = np.pad(signal, (pad_len, pad_len), mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    frames = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
    for i in range(n_frames):
        start = i * hop_length
        segment = padded[start:start + n_fft] * w
        frames[:, i] = np.fft.rfft(segment)
    return frames


def istft(
    stft_matrix: np.ndarray, hop_length: int = 512, n_fft: int = 2048,
    window: str = "hann",
) -> np.ndarray:
    n_freq, n_frames = stft_matrix.shape
    expected_len = n_fft + hop_length * (n_frames - 1)
    output = np.zeros(expected_len)
    window_sum = np.zeros(expected_len)

    if window == "hann":
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
    else:
        w = np.ones(n_fft)

    for i in range(n_frames):
        start = i * hop_length
        frame = np.fft.irfft(stft_matrix[:, i], n=n_fft)
        output[start:start + n_fft] += frame * w
        window_sum[start:start + n_fft] += w ** 2

    nonzero = window_sum > 1e-10
    output[nonzero] /= window_sum[nonzero]
    pad_len = n_fft // 2
    return output[pad_len:-pad_len] if pad_len > 0 else output


def magnitude_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    return np.abs(stft_matrix)


def power_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    return np.abs(stft_matrix) ** 2


def phase_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    return np.angle(stft_matrix)


def hz_to_mel(f: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int = 128, n_fft: int = 2048, sr: int = 22050,
    f_min: float = 0.0, f_max: float | None = None,
) -> np.ndarray:
    f_max = f_max or sr / 2.0
    mel_min = float(hz_to_mel(f_min))
    mel_max = float(hz_to_mel(f_max))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_freq = n_fft // 2 + 1
    filters = np.zeros((n_mels, n_freq))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filters[i, j] = (right - j) / (right - center)
    return filters


def mel_spectrogram(
    signal: np.ndarray, sr: int = 22050, n_fft: int = 2048,
    hop_length: int = 512, n_mels: int = 128,
) -> np.ndarray:
    S = stft(signal, n_fft=n_fft, hop_length=hop_length)
    power = power_spectrogram(S)
    fb = mel_filterbank(n_mels=n_mels, n_fft=n_fft, sr=sr)
    return fb @ power


def amplitude_to_db(S: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    magnitude = np.maximum(np.abs(S), amin)
    return 20.0 * np.log10(magnitude / ref)


def cqt_frequencies(n_bins: int = 84, f_min: float = 32.7, bins_per_octave: int = 12) -> np.ndarray:
    return f_min * 2.0 ** (np.arange(n_bins) / bins_per_octave)


def pseudo_cqt(
    signal: np.ndarray, sr: int = 22050, n_bins: int = 84,
    f_min: float = 32.7, bins_per_octave: int = 12,
    hop_length: int = 512, n_fft: int = 4096,
) -> np.ndarray:
    S = stft(signal, n_fft=n_fft, hop_length=hop_length)
    mag = magnitude_spectrogram(S)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    cq_freqs = cqt_frequencies(n_bins, f_min, bins_per_octave)

    result = np.zeros((n_bins, mag.shape[1]))
    for i, fc in enumerate(cq_freqs):
        bw = fc * (2.0 ** (1.0 / bins_per_octave) - 1.0)
        weights = np.exp(-0.5 * ((freqs - fc) / (bw / 2)) ** 2)
        weights /= weights.sum() + 1e-12
        result[i] = weights @ mag
    return result


def chromagram(
    signal: np.ndarray, sr: int = 22050, n_fft: int = 4096,
    hop_length: int = 512, n_chroma: int = 12,
) -> np.ndarray:
    S = stft(signal, n_fft=n_fft, hop_length=hop_length)
    mag = magnitude_spectrogram(S)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    chroma = np.zeros((n_chroma, mag.shape[1]))
    for i in range(1, len(freqs)):
        if freqs[i] < 20.0:
            continue
        pitch_class = int(round(12 * np.log2(freqs[i] / 440.0))) % 12
        chroma[pitch_class] += mag[i]
    norm = chroma.max(axis=0, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    return chroma / norm
