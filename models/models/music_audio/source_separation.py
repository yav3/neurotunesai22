"""Audio source separation primitives: masking, Wiener filtering, HPSS, REPET, Griffin-Lim, BSS metrics."""
from __future__ import annotations
import numpy as np
from scipy.signal import medfilt2d


def ideal_binary_mask(mix_stft: np.ndarray, source_stft: np.ndarray) -> np.ndarray:
    return (np.abs(source_stft) > np.abs(mix_stft) / 2).astype(float)


def ideal_ratio_mask(mix_stft: np.ndarray, source_stft: np.ndarray) -> np.ndarray:
    mix_mag = np.abs(mix_stft)
    src_mag = np.abs(source_stft)
    return np.where(mix_mag > 1e-10, src_mag / mix_mag, 0.0)


def wiener_filter(mix_stft: np.ndarray, source_estimates: list[np.ndarray]) -> list[np.ndarray]:
    powers = [np.abs(s) ** 2 for s in source_estimates]
    total_power = sum(powers)
    total_power = np.where(total_power > 1e-10, total_power, 1e-10)
    return [mix_stft * (p / total_power) for p in powers]


def harmonic_percussive_separation(
    spectrogram: np.ndarray, kernel_h: int = 31, kernel_p: int = 31
) -> tuple[np.ndarray, np.ndarray]:
    mag = np.abs(spectrogram)
    kh = kernel_h if kernel_h % 2 == 1 else kernel_h + 1
    kp = kernel_p if kernel_p % 2 == 1 else kernel_p + 1
    h_median = medfilt2d(mag, kernel_size=(1, kh))
    p_median = medfilt2d(mag, kernel_size=(kp, 1))
    total = h_median + p_median
    total = np.where(total > 1e-10, total, 1e-10)
    h_mask = h_median / total
    p_mask = p_median / total
    phase = np.exp(1j * np.angle(spectrogram)) if np.iscomplexobj(spectrogram) else 1.0
    return mag * h_mask * phase, mag * p_mask * phase


def repet_sim(spectrogram: np.ndarray, min_period: int = 4, max_period: int = 0) -> np.ndarray:
    mag = np.abs(spectrogram)
    n_frames = mag.shape[1]
    if max_period <= 0:
        max_period = n_frames // 2
    beat_spectrum = np.zeros(n_frames)
    for lag in range(n_frames):
        shifted = np.roll(mag, -lag, axis=1)
        overlap = n_frames - lag
        if overlap <= 0:
            break
        beat_spectrum[lag] = np.sum(mag[:, :overlap] * shifted[:, :overlap]) / overlap

    search = beat_spectrum[min_period:max_period + 1]
    if len(search) == 0:
        return mag
    period = int(np.argmax(search)) + min_period

    repeating = np.zeros_like(mag)
    for f in range(mag.shape[0]):
        for t in range(n_frames):
            segments = []
            k = t
            while k < n_frames:
                segments.append(mag[f, k])
                k += period
            k = t - period
            while k >= 0:
                segments.append(mag[f, k])
                k -= period
            repeating[f, t] = np.median(segments)
    return np.minimum(repeating, mag)


def griffin_lim(magnitude: np.ndarray, n_iter: int = 32, hop_length: int = 512) -> np.ndarray:
    n_fft = 2 * (magnitude.shape[0] - 1)
    phase = np.exp(2j * np.pi * np.random.default_rng(42).random(magnitude.shape))
    for _ in range(n_iter):
        stft_est = magnitude * phase
        n_frames = stft_est.shape[1]
        signal_len = n_fft + (n_frames - 1) * hop_length
        signal = np.zeros(signal_len)
        window = np.hanning(n_fft)
        norm = np.zeros(signal_len)
        for t in range(n_frames):
            start = t * hop_length
            frame = np.fft.irfft(stft_est[:, t], n=n_fft)
            signal[start:start + n_fft] += frame * window
            norm[start:start + n_fft] += window ** 2
        norm = np.where(norm > 1e-10, norm, 1e-10)
        signal /= norm
        for t in range(n_frames):
            start = t * hop_length
            frame = signal[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            phase[:, t] = np.exp(1j * np.angle(spectrum))
    return signal


def _project(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    ref = reference.ravel()
    est = estimate.ravel()
    dot = np.dot(ref, est)
    norm_sq = np.dot(ref, ref)
    if norm_sq < 1e-20:
        return np.zeros_like(ref)
    return (dot / norm_sq) * ref


def SDR(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.ravel().astype(float)
    est = estimate.ravel().astype(float)
    s_target = _project(ref, est)
    e_noise = est - s_target
    num = np.dot(s_target, s_target)
    den = np.dot(e_noise, e_noise)
    if den < 1e-20:
        return np.inf if num > 1e-20 else 0.0
    if num < 1e-20:
        return 0.0
    return float(10 * np.log10(num / den))


def SIR(reference: np.ndarray, estimate: np.ndarray, interference: np.ndarray) -> float:
    ref = reference.ravel().astype(float)
    est = estimate.ravel().astype(float)
    intf = interference.ravel().astype(float)
    s_target = _project(ref, est)
    s_interf = _project(intf, est - s_target)
    num = np.dot(s_target, s_target)
    den = np.dot(s_interf, s_interf)
    if den < 1e-20:
        return np.inf if num > 1e-20 else 0.0
    return float(10 * np.log10(num / den))


def SAR(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.ravel().astype(float)
    est = estimate.ravel().astype(float)
    s_target = _project(ref, est)
    e_artif = est - s_target
    den = np.dot(e_artif, e_artif)
    num = np.dot(s_target, s_target) + den
    if den < 1e-20:
        return np.inf if num > 1e-20 else 0.0
    return float(10 * np.log10(num / den))
