"""Data augmentation for music ML: pitch shift, time stretch, noise injection, mixing.

Provides audio-domain and symbolic-domain augmentation strategies for training
robust music models. Each transform is invertible or parameterized for
controlled augmentation pipelines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class AugmentationType(Enum):
    PITCH_SHIFT = "pitch_shift"
    TIME_STRETCH = "time_stretch"
    NOISE_INJECTION = "noise_injection"
    GAIN_CHANGE = "gain_change"
    CHANNEL_SWAP = "channel_swap"
    FREQUENCY_MASK = "frequency_mask"
    TIME_MASK = "time_mask"
    POLARITY_INVERT = "polarity_invert"
    RANDOM_EQ = "random_eq"
    MIX_BLEND = "mix_blend"


@dataclass(frozen=True)
class AugmentationConfig:
    aug_type: AugmentationType
    probability: float = 0.5
    min_param: float = 0.0
    max_param: float = 1.0


@dataclass(frozen=True)
class AugmentationResult:
    original_shape: tuple
    augmented_shape: tuple
    applied: list[AugmentationType]
    params: dict


def pitch_shift_stft(
    signal: np.ndarray,
    semitones: float,
    sr: float = 44100.0,
    hop: int = 512,
) -> np.ndarray:
    ratio = 2.0 ** (semitones / 12.0)
    n = len(signal)
    indices = np.arange(0, n, ratio)
    indices = indices[indices < n - 1].astype(int)
    if len(indices) == 0:
        return signal.copy()
    shifted = np.interp(np.arange(n), indices, signal[indices])
    return shifted


def time_stretch_ola(
    signal: np.ndarray,
    rate: float,
    frame_size: int = 2048,
    hop: int = 512,
) -> np.ndarray:
    if rate <= 0:
        return signal.copy()
    n = len(signal)
    out_len = int(n / rate)
    output = np.zeros(out_len)
    window = np.hanning(frame_size)
    analysis_hop = hop
    synthesis_hop = int(hop / rate)
    for i in range(0, out_len - frame_size, synthesis_hop):
        src_pos = int(i * rate)
        if src_pos + frame_size > n:
            break
        frame = signal[src_pos:src_pos + frame_size] * window
        end = min(i + frame_size, out_len)
        output[i:end] += frame[:end - i]
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak
        output *= np.max(np.abs(signal))
    return output


def add_noise(
    signal: np.ndarray,
    snr_db: float = 20.0,
    noise_type: str = "white",
) -> np.ndarray:
    rng = np.random.RandomState(42)
    sig_power = float(np.mean(signal ** 2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    if noise_type == "pink":
        white = rng.randn(len(signal))
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(len(signal))
        freqs[0] = 1e-6
        fft /= np.sqrt(freqs)
        noise = np.fft.irfft(fft, len(signal))
    else:
        noise = rng.randn(len(signal))
    noise *= math.sqrt(noise_power) / max(float(np.std(noise)), 1e-12)
    return signal + noise


def gain_change(signal: np.ndarray, db: float) -> np.ndarray:
    return signal * (10.0 ** (db / 20.0))


def polarity_invert(signal: np.ndarray) -> np.ndarray:
    return -signal


def frequency_mask(
    spectrogram: np.ndarray,
    n_bands: int = 1,
    max_width: int = 20,
) -> np.ndarray:
    result = spectrogram.copy()
    rng = np.random.RandomState(0)
    n_freq = result.shape[0]
    for _ in range(n_bands):
        width = rng.randint(1, min(max_width, n_freq) + 1)
        start = rng.randint(0, max(n_freq - width, 1))
        result[start:start + width, :] = 0.0
    return result


def time_mask(
    spectrogram: np.ndarray,
    n_masks: int = 1,
    max_width: int = 30,
) -> np.ndarray:
    result = spectrogram.copy()
    rng = np.random.RandomState(1)
    n_time = result.shape[1]
    for _ in range(n_masks):
        width = rng.randint(1, min(max_width, n_time) + 1)
        start = rng.randint(0, max(n_time - width, 1))
        result[:, start:start + width] = 0.0
    return result


def mix_blend(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    ratio: float = 0.5,
) -> np.ndarray:
    min_len = min(len(signal_a), len(signal_b))
    return signal_a[:min_len] * (1 - ratio) + signal_b[:min_len] * ratio


def apply_augmentation_pipeline(
    signal: np.ndarray,
    configs: list[AugmentationConfig],
    sr: float = 44100.0,
) -> AugmentationResult:
    rng = np.random.RandomState(42)
    result = signal.copy()
    applied: list[AugmentationType] = []
    params: dict = {}
    for cfg in configs:
        if rng.random() > cfg.probability:
            continue
        param = cfg.min_param + rng.random() * (cfg.max_param - cfg.min_param)
        if cfg.aug_type == AugmentationType.PITCH_SHIFT:
            result = pitch_shift_stft(result, param, sr)
        elif cfg.aug_type == AugmentationType.TIME_STRETCH:
            result = time_stretch_ola(result, 0.8 + param * 0.4)
        elif cfg.aug_type == AugmentationType.NOISE_INJECTION:
            result = add_noise(result, 10.0 + param * 30.0)
        elif cfg.aug_type == AugmentationType.GAIN_CHANGE:
            result = gain_change(result, -6.0 + param * 12.0)
        elif cfg.aug_type == AugmentationType.POLARITY_INVERT:
            result = polarity_invert(result)
        elif cfg.aug_type == AugmentationType.FREQUENCY_MASK:
            spec = np.abs(np.fft.rfft(result))
            spec = frequency_mask(spec.reshape(-1, 1), max_width=int(param * 50 + 1)).flatten()
            result = np.fft.irfft(spec, len(result))
        applied.append(cfg.aug_type)
        params[cfg.aug_type.value] = param
    return AugmentationResult(
        original_shape=signal.shape,
        augmented_shape=result.shape,
        applied=applied,
        params=params,
    )
