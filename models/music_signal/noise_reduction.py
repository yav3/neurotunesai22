"""Noise reduction and audio denoising algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "NoiseProfile", "DenoiseConfig",
    "estimate_noise_profile", "spectral_subtraction",
    "wiener_filter", "adaptive_noise_gate",
    "spectral_noise_gate", "compute_snr",
    "multi_band_noise_gate", "noise_floor_estimate",
    "harmonic_percussive_separation", "median_filter_denoise",
]


@dataclass
class NoiseProfile:
    mean_spectrum: np.ndarray
    std_spectrum: np.ndarray
    floor_db: float


@dataclass
class DenoiseConfig:
    n_fft: int = 2048
    hop_length: int = 512
    oversubtraction: float = 1.0
    spectral_floor: float = 0.01
    wiener_beta: float = 0.01


def _stft(audio: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop
    frames = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
    for t in range(n_frames):
        start = t * hop
        frames[:, t] = np.fft.rfft(audio[start:start + n_fft] * window)
    return frames


def _istft(stft: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    window = np.hanning(n_fft)
    n_frames = stft.shape[1]
    output = np.zeros(length)
    window_sum = np.zeros(length)
    for t in range(n_frames):
        start = t * hop
        end = min(start + n_fft, length)
        frame = np.fft.irfft(stft[:, t], n_fft)
        output[start:end] += frame[:end - start] * window[:end - start]
        window_sum[start:end] += window[:end - start] ** 2
    mask = window_sum > 1e-9
    output[mask] /= window_sum[mask]
    return output


def estimate_noise_profile(
    audio: np.ndarray, sr: int = 22050, n_fft: int = 2048, hop: int = 512,
    noise_frames: int = 10,
) -> NoiseProfile:
    stft = _stft(audio, n_fft, hop)
    mag = np.abs(stft)
    n_use = min(noise_frames, mag.shape[1])
    noise_section = mag[:, :n_use]
    mean_spec = noise_section.mean(axis=1)
    std_spec = noise_section.std(axis=1)
    floor_db = float(20 * np.log10(mean_spec.mean() + 1e-10))
    return NoiseProfile(mean_spec, std_spec, floor_db)


def spectral_subtraction(
    audio: np.ndarray, noise_profile: NoiseProfile,
    config: Optional[DenoiseConfig] = None,
) -> np.ndarray:
    cfg = config or DenoiseConfig()
    stft = _stft(audio, cfg.n_fft, cfg.hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)
    subtracted = mag ** 2 - cfg.oversubtraction * noise_profile.mean_spectrum[:, None] ** 2
    floor = cfg.spectral_floor * mag ** 2
    subtracted = np.maximum(subtracted, floor)
    clean_mag = np.sqrt(subtracted)
    clean_stft = clean_mag * np.exp(1j * phase)
    return _istft(clean_stft, cfg.n_fft, cfg.hop_length, len(audio))


def wiener_filter(
    audio: np.ndarray, noise_profile: NoiseProfile,
    config: Optional[DenoiseConfig] = None,
) -> np.ndarray:
    cfg = config or DenoiseConfig()
    stft = _stft(audio, cfg.n_fft, cfg.hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)
    noise_power = noise_profile.mean_spectrum[:, None] ** 2
    signal_power = np.maximum(mag ** 2 - noise_power, 0)
    gain = signal_power / (signal_power + noise_power + cfg.wiener_beta)
    clean_stft = gain * mag * np.exp(1j * phase)
    return _istft(clean_stft, cfg.n_fft, cfg.hop_length, len(audio))


def adaptive_noise_gate(
    audio: np.ndarray, threshold_db: float = -40.0,
    attack_ms: float = 1.0, release_ms: float = 50.0, sr: int = 22050,
) -> np.ndarray:
    threshold = 10 ** (threshold_db / 20.0)
    attack_samples = max(1, int(attack_ms * sr / 1000))
    release_samples = max(1, int(release_ms * sr / 1000))
    envelope = np.abs(audio)
    smoothed = np.zeros_like(envelope)
    smoothed[0] = envelope[0]
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i - 1]:
            coeff = 1.0 - np.exp(-1.0 / attack_samples)
        else:
            coeff = 1.0 - np.exp(-1.0 / release_samples)
        smoothed[i] = smoothed[i - 1] + coeff * (envelope[i] - smoothed[i - 1])
    gain = np.where(smoothed > threshold, 1.0, smoothed / (threshold + 1e-10))
    return audio * gain


def spectral_noise_gate(
    audio: np.ndarray, threshold_db: float = -60.0,
    n_fft: int = 2048, hop: int = 512,
) -> np.ndarray:
    stft = _stft(audio, n_fft, hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    threshold_lin = 10 ** (threshold_db / 20.0)
    gain = np.where(mag > threshold_lin, 1.0, mag / (threshold_lin + 1e-10))
    clean_stft = gain * mag * np.exp(1j * phase)
    return _istft(clean_stft, n_fft, hop, len(audio))


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-15:
        return float('inf')
    return float(10 * np.log10(sig_power / noise_power))


def multi_band_noise_gate(
    audio: np.ndarray, sr: int = 22050,
    band_edges: Optional[np.ndarray] = None,
    thresholds_db: Optional[np.ndarray] = None,
    n_fft: int = 2048, hop: int = 512,
) -> np.ndarray:
    if band_edges is None:
        band_edges = np.array([0, 200, 1000, 4000, sr // 2])
    if thresholds_db is None:
        thresholds_db = np.array([-50, -45, -40, -35])
    stft = _stft(audio, n_fft, hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    gain = np.ones_like(mag)
    for b in range(len(band_edges) - 1):
        band_mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        thresh_lin = 10 ** (thresholds_db[b] / 20.0)
        for k in np.where(band_mask)[0]:
            low = mag[k] < thresh_lin
            gain[k, low] = mag[k, low] / (thresh_lin + 1e-10)
    clean_stft = gain * mag * np.exp(1j * phase)
    return _istft(clean_stft, n_fft, hop, len(audio))


def noise_floor_estimate(audio: np.ndarray, percentile: float = 10.0) -> float:
    rms_frames = []
    frame_size = 1024
    for i in range(0, len(audio) - frame_size, frame_size):
        rms_frames.append(np.sqrt(np.mean(audio[i:i + frame_size] ** 2)))
    if not rms_frames:
        return 0.0
    return float(np.percentile(rms_frames, percentile))


def harmonic_percussive_separation(
    audio: np.ndarray, n_fft: int = 2048, hop: int = 512,
    kernel_harmonic: int = 31, kernel_percussive: int = 31,
) -> Tuple[np.ndarray, np.ndarray]:
    stft = _stft(audio, n_fft, hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    h_mag = np.zeros_like(mag)
    half_h = kernel_harmonic // 2
    for k in range(mag.shape[0]):
        for t in range(mag.shape[1]):
            start = max(0, t - half_h)
            end = min(mag.shape[1], t + half_h + 1)
            h_mag[k, t] = np.median(mag[k, start:end])
    p_mag = np.zeros_like(mag)
    half_p = kernel_percussive // 2
    for t in range(mag.shape[1]):
        for k in range(mag.shape[0]):
            start = max(0, k - half_p)
            end = min(mag.shape[0], k + half_p + 1)
            p_mag[k, t] = np.median(mag[start:end, t])
    mask_h = h_mag ** 2 / (h_mag ** 2 + p_mag ** 2 + 1e-10)
    mask_p = 1.0 - mask_h
    harmonic = _istft(mask_h * mag * np.exp(1j * phase), n_fft, hop, len(audio))
    percussive = _istft(mask_p * mag * np.exp(1j * phase), n_fft, hop, len(audio))
    return harmonic, percussive


def median_filter_denoise(
    audio: np.ndarray, n_fft: int = 2048, hop: int = 512, kernel: int = 5,
) -> np.ndarray:
    stft = _stft(audio, n_fft, hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    filtered = np.zeros_like(mag)
    half = kernel // 2
    for t in range(mag.shape[1]):
        start = max(0, t - half)
        end = min(mag.shape[1], t + half + 1)
        filtered[:, t] = np.median(mag[:, start:end], axis=1)
    clean_stft = filtered * np.exp(1j * phase)
    return _istft(clean_stft, n_fft, hop, len(audio))
