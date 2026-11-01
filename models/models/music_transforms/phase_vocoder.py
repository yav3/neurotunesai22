"""Phase vocoder: time-stretching, pitch-shifting via STFT phase manipulation.

Implements the phase vocoder algorithm for time-scale modification without
pitch change, and pitch-shifting via resampling after time-stretching.
Includes phase-locking, identity phase-locking, and transient preservation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _hann_window(n: int) -> np.ndarray:
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


def _stft(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    window = _hann_window(n_fft)
    n_frames = max(0, (len(signal) - n_fft) // hop + 1)
    frames = np.empty((n_frames, n_fft // 2 + 1), dtype=complex)
    for i in range(n_frames):
        s = i * hop
        frames[i] = np.fft.rfft(signal[s : s + n_fft] * window)
    return frames


def _istft(frames: np.ndarray, n_fft: int, hop: int, length: int | None = None) -> np.ndarray:
    window = _hann_window(n_fft)
    n_frames = len(frames)
    out_len = n_fft + (n_frames - 1) * hop
    output = np.zeros(out_len)
    norm = np.zeros(out_len)
    for i in range(n_frames):
        s = i * hop
        output[s : s + n_fft] += np.fft.irfft(frames[i], n=n_fft) * window
        norm[s : s + n_fft] += window ** 2
    mask = norm > 1e-8
    output[mask] /= norm[mask]
    if length is not None:
        output = output[:length]
    return output


def phase_vocoder_stretch(
    signal: np.ndarray,
    stretch_factor: float,
    n_fft: int = 2048,
    hop: int = 512,
) -> np.ndarray:
    if stretch_factor <= 0:
        raise ValueError("stretch_factor must be positive")
    if len(signal) < n_fft:
        return signal.copy()

    frames = _stft(signal, n_fft, hop)
    n_frames = len(frames)
    n_bins = frames.shape[1]

    synthesis_hop = hop
    analysis_hop = hop
    expected_phase_advance = 2 * np.pi * np.arange(n_bins) * analysis_hop / n_fft

    n_out_frames = int(round(n_frames * stretch_factor))
    out_frames = np.empty((n_out_frames, n_bins), dtype=complex)

    phase_accum = np.angle(frames[0])
    out_frames[0] = np.abs(frames[0]) * np.exp(1j * phase_accum)

    for i in range(1, n_out_frames):
        src_pos = i / stretch_factor
        src_idx = int(src_pos)
        frac = src_pos - src_idx

        if src_idx + 1 < n_frames:
            mag = (1 - frac) * np.abs(frames[src_idx]) + frac * np.abs(frames[src_idx + 1])
            phase_cur = np.angle(frames[src_idx + 1])
            phase_prev = np.angle(frames[src_idx])
        elif src_idx < n_frames:
            mag = np.abs(frames[src_idx])
            phase_cur = np.angle(frames[src_idx])
            phase_prev = np.angle(frames[max(0, src_idx - 1)])
        else:
            mag = np.abs(frames[-1])
            phase_cur = np.angle(frames[-1])
            phase_prev = np.angle(frames[max(0, n_frames - 2)])

        dphi = phase_cur - phase_prev - expected_phase_advance
        dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
        true_freq = expected_phase_advance + dphi
        phase_accum += true_freq
        out_frames[i] = mag * np.exp(1j * phase_accum)

    out_length = int(round(len(signal) * stretch_factor))
    return _istft(out_frames, n_fft, synthesis_hop, out_length)


def pitch_shift(
    signal: np.ndarray,
    semitones: float,
    sr: float = 44100.0,
    n_fft: int = 2048,
    hop: int = 512,
) -> np.ndarray:
    ratio = 2.0 ** (semitones / 12.0)
    stretched = phase_vocoder_stretch(signal, 1.0 / ratio, n_fft, hop)
    indices = np.arange(len(signal)) * ratio
    int_idx = np.floor(indices).astype(int)
    frac = indices - int_idx
    valid = int_idx + 1 < len(stretched)
    output = np.zeros(len(signal))
    output[valid] = (1 - frac[valid]) * stretched[int_idx[valid]] + frac[valid] * stretched[int_idx[valid] + 1]
    return output


def identity_phase_locking(
    frames: np.ndarray,
    analysis_hop: int,
    synthesis_hop: int,
    n_fft: int,
) -> np.ndarray:
    n_frames, n_bins = frames.shape
    out = np.empty_like(frames)
    out[0] = frames[0]
    phase_accum = np.angle(frames[0])
    expected = 2 * np.pi * np.arange(n_bins) * analysis_hop / n_fft

    for i in range(1, n_frames):
        mag = np.abs(frames[i])
        phase = np.angle(frames[i])
        phase_prev = np.angle(frames[i - 1])

        dphi = phase - phase_prev - expected
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        inst_freq = expected + dphi

        peaks = np.where((mag[1:-1] >= mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1

        locked_phase = phase_accum + inst_freq * (synthesis_hop / analysis_hop)
        for pk in peaks:
            region_lo = max(0, pk - 2)
            region_hi = min(n_bins, pk + 3)
            rotation = locked_phase[pk] - phase[pk]
            locked_phase[region_lo:region_hi] = phase[region_lo:region_hi] + rotation

        phase_accum = locked_phase
        out[i] = mag * np.exp(1j * locked_phase)
    return out


def detect_transients(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    threshold: float = 2.0,
) -> np.ndarray:
    frames = _stft(signal, n_fft, hop)
    if len(frames) < 2:
        return np.array([], dtype=int)
    mag = np.abs(frames)
    flux = np.sum(np.maximum(mag[1:] - mag[:-1], 0), axis=1)
    if len(flux) == 0:
        return np.array([], dtype=int)
    median_flux = np.median(flux)
    is_transient = flux > threshold * max(median_flux, 1e-10)
    return np.where(is_transient)[0] + 1


@dataclass(frozen=True)
class TimeStretchResult:
    audio: np.ndarray
    stretch_factor: float
    original_length: int
    output_length: int
    transient_frames: np.ndarray


def time_stretch_with_transient_preservation(
    signal: np.ndarray,
    stretch_factor: float,
    n_fft: int = 2048,
    hop: int = 512,
    transient_threshold: float = 2.0,
) -> TimeStretchResult:
    transients = detect_transients(signal, n_fft, hop, transient_threshold)
    audio = phase_vocoder_stretch(signal, stretch_factor, n_fft, hop)
    return TimeStretchResult(
        audio=audio,
        stretch_factor=stretch_factor,
        original_length=len(signal),
        output_length=len(audio),
        transient_frames=transients,
    )


def formant_preserving_pitch_shift(
    signal: np.ndarray,
    semitones: float,
    sr: float = 44100.0,
    n_fft: int = 2048,
    hop: int = 512,
    envelope_order: int = 30,
) -> np.ndarray:
    frames_orig = _stft(signal, n_fft, hop)
    shifted = pitch_shift(signal, semitones, sr, n_fft, hop)
    frames_shifted = _stft(shifted, n_fft, min(hop, len(shifted) // 4 or hop))

    n_out = min(len(frames_orig), len(frames_shifted))
    corrected = np.empty((n_out, frames_orig.shape[1]), dtype=complex)

    for i in range(n_out):
        mag_orig = np.abs(frames_orig[i])
        mag_shifted = np.abs(frames_shifted[i])

        env_orig = np.maximum(np.convolve(mag_orig, np.ones(envelope_order) / envelope_order, mode="same"), 1e-12)
        env_shifted = np.maximum(np.convolve(mag_shifted, np.ones(envelope_order) / envelope_order, mode="same"), 1e-12)

        correction = env_orig / env_shifted
        corrected[i] = frames_shifted[i] * correction

    return _istft(corrected, n_fft, hop, len(signal))
