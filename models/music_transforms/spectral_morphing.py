"""Spectral morphing: cross-synthesis, spectral interpolation,
timbre blending, frequency warping, harmonic matching.

Based on Serra & Smith (1990) spectral modeling synthesis and
Rodet & Depalle (1992) spectral envelope estimation. Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class SpectralFrame:
    frequencies: np.ndarray
    magnitudes: np.ndarray
    phases: np.ndarray


@dataclass
class MorphParams:
    alpha: float = 0.5
    frequency_warp: float = 1.0
    preserve_formants: bool = False
    smoothing_order: int = 20


@dataclass
class MorphTrajectory:
    alphas: np.ndarray
    times: np.ndarray


def extract_spectral_envelope(magnitudes: np.ndarray,
                               order: int = 30) -> np.ndarray:
    n = len(magnitudes)
    log_mag = np.log(np.abs(magnitudes) + 1e-10)
    cepstrum = np.fft.irfft(log_mag, n=2 * (n - 1))
    liftered = np.zeros_like(cepstrum)
    liftered[:order] = cepstrum[:order]
    liftered[0] *= 0.5
    envelope = np.fft.rfft(liftered, n=2 * (n - 1))
    return np.exp(np.real(envelope[:n]))


def spectral_interpolation(mag_a: np.ndarray, mag_b: np.ndarray,
                           alpha: float = 0.5,
                           log_domain: bool = True) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)
    if log_domain:
        log_a = np.log(np.abs(mag_a) + 1e-10)
        log_b = np.log(np.abs(mag_b) + 1e-10)
        return np.exp((1.0 - alpha) * log_a + alpha * log_b)
    return (1.0 - alpha) * mag_a + alpha * mag_b


def phase_vocoder_morph(phase_a: np.ndarray, phase_b: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
    unwrapped_a = np.unwrap(phase_a)
    unwrapped_b = np.unwrap(phase_b)
    morphed = (1.0 - alpha) * unwrapped_a + alpha * unwrapped_b
    return morphed % (2 * np.pi)


def cross_synthesis(excitation: np.ndarray, envelope_source: np.ndarray,
                    fft_size: int = 2048, hop_size: int = 512,
                    envelope_order: int = 30) -> np.ndarray:
    n_frames = (len(excitation) - fft_size) // hop_size + 1
    output = np.zeros(len(excitation))
    window = np.hanning(fft_size)
    for i in range(n_frames):
        start = i * hop_size
        exc_frame = excitation[start:start + fft_size] * window
        env_frame = envelope_source[start:start + fft_size] * window
        exc_fft = np.fft.rfft(exc_frame)
        env_fft = np.fft.rfft(env_frame)
        target_env = extract_spectral_envelope(np.abs(env_fft), envelope_order)
        source_env = extract_spectral_envelope(np.abs(exc_fft), envelope_order)
        gain = target_env / (source_env + 1e-10)
        morphed_fft = exc_fft * gain
        frame_out = np.fft.irfft(morphed_fft, n=fft_size)
        output[start:start + fft_size] += frame_out * window
    return output


def timbre_distance(centroid_a: float, centroid_b: float,
                    flatness_a: float, flatness_b: float,
                    w_centroid: float = 0.7,
                    w_flatness: float = 0.3) -> float:
    c_diff = abs(np.log2((centroid_a + 1.0) / (centroid_b + 1.0)))
    f_diff = abs(flatness_a - flatness_b)
    return float(w_centroid * c_diff + w_flatness * f_diff)


def morph_trajectory(n_frames: int,
                     curve: str = "linear") -> MorphTrajectory:
    t = np.linspace(0, 1, n_frames)
    if curve == "sigmoid":
        alphas = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))
    elif curve == "exponential":
        alphas = t ** 2
    elif curve == "logarithmic":
        alphas = np.sqrt(t)
    elif curve == "s_curve":
        alphas = 0.5 * (1 - np.cos(np.pi * t))
    else:
        alphas = t
    return MorphTrajectory(alphas=alphas, times=t)


def harmonic_matching(freqs_a: np.ndarray, freqs_b: np.ndarray,
                      max_ratio: float = 1.2) -> List[Tuple[int, int, float]]:
    pairs = []
    used_b = set()
    for i, fa in enumerate(freqs_a):
        best_j = -1
        best_ratio = float("inf")
        for j, fb in enumerate(freqs_b):
            if j in used_b:
                continue
            ratio = max(fa, fb) / (min(fa, fb) + 1e-10)
            if ratio < best_ratio and ratio <= max_ratio:
                best_ratio = ratio
                best_j = j
        if best_j >= 0:
            pairs.append((i, best_j, best_ratio))
            used_b.add(best_j)
    return pairs


def frequency_warping(frequencies: np.ndarray,
                      warp_factor: float = 1.0,
                      anchor: float = 440.0) -> np.ndarray:
    ratios = frequencies / (anchor + 1e-10)
    warped_ratios = np.sign(ratios) * np.abs(ratios) ** warp_factor
    return warped_ratios * anchor


def spectral_blend(frame_a: SpectralFrame, frame_b: SpectralFrame,
                   params: MorphParams) -> SpectralFrame:
    alpha = np.clip(params.alpha, 0.0, 1.0)
    n = min(len(frame_a.magnitudes), len(frame_b.magnitudes))
    mag_a = frame_a.magnitudes[:n]
    mag_b = frame_b.magnitudes[:n]
    if params.preserve_formants:
        env_a = extract_spectral_envelope(mag_a, params.smoothing_order)
        env_b = extract_spectral_envelope(mag_b, params.smoothing_order)
        fine_a = mag_a / (env_a + 1e-10)
        fine_b = mag_b / (env_b + 1e-10)
        blended_env = spectral_interpolation(env_a, env_b, alpha)
        blended_fine = spectral_interpolation(fine_a, fine_b, alpha)
        blended_mag = blended_env * blended_fine
    else:
        blended_mag = spectral_interpolation(mag_a, mag_b, alpha)
    freq_a = frame_a.frequencies[:n]
    freq_b = frame_b.frequencies[:n]
    if params.frequency_warp != 1.0:
        freq_b = frequency_warping(freq_b, params.frequency_warp)
    blended_freq = (1.0 - alpha) * freq_a + alpha * freq_b
    blended_phase = phase_vocoder_morph(
        frame_a.phases[:n], frame_b.phases[:n], alpha)
    return SpectralFrame(blended_freq, blended_mag, blended_phase)


def morph_sequence(frames_a: List[SpectralFrame],
                   frames_b: List[SpectralFrame],
                   trajectory: MorphTrajectory,
                   params: MorphParams) -> List[SpectralFrame]:
    n = min(len(frames_a), len(frames_b), len(trajectory.alphas))
    result = []
    for i in range(n):
        p = MorphParams(
            alpha=float(trajectory.alphas[i]),
            frequency_warp=params.frequency_warp,
            preserve_formants=params.preserve_formants,
            smoothing_order=params.smoothing_order,
        )
        result.append(spectral_blend(frames_a[i], frames_b[i], p))
    return result
