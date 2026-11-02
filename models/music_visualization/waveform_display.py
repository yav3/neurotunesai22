"""Waveform rendering and audio visualization data preparation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WaveformData:
    peaks_positive: np.ndarray
    peaks_negative: np.ndarray
    rms: np.ndarray
    sample_rate: int
    duration_s: float
    n_display_points: int


def compute_waveform_display(
    signal: np.ndarray, sr: int, n_points: int = 1000,
) -> WaveformData:
    n = len(signal)
    duration = n / sr
    block = max(1, n // n_points)
    actual_points = n // block
    peaks_pos = np.zeros(actual_points)
    peaks_neg = np.zeros(actual_points)
    rms_arr = np.zeros(actual_points)
    for i in range(actual_points):
        chunk = signal[i * block : (i + 1) * block]
        peaks_pos[i] = np.max(chunk)
        peaks_neg[i] = np.min(chunk)
        rms_arr[i] = np.sqrt(np.mean(chunk ** 2))
    return WaveformData(
        peaks_positive=peaks_pos, peaks_negative=peaks_neg,
        rms=rms_arr, sample_rate=sr, duration_s=duration,
        n_display_points=actual_points,
    )


def compute_spectrogram_display(
    signal: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512, n_mels: int = 128,
) -> dict:
    from numpy.fft import rfft
    n = len(signal)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    n_frames = max(1, (n - n_fft) // hop + 1)
    n_bins = n_fft // 2 + 1
    mag = np.zeros((n_bins, n_frames))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        mag[:, i] = np.abs(rfft(frame * window))
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-10))
    freqs = np.linspace(0, sr / 2, n_bins)
    times = np.arange(n_frames) * hop / sr
    return {"frequencies": freqs, "times": times, "magnitudes_db": mag_db}


def compute_chromagram_display(
    signal: np.ndarray, sr: int, hop: int = 512,
) -> dict:
    n_fft = 4096
    from numpy.fft import rfft
    n = len(signal)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    n_frames = max(1, (n - n_fft) // hop + 1)
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    chroma = np.zeros((12, n_frames))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spectrum = np.abs(rfft(frame * window)) ** 2
        for b in range(1, n_bins):
            if freqs[b] > 20:
                midi = 69 + 12 * np.log2(freqs[b] / 440.0)
                pc = int(round(midi)) % 12
                chroma[pc, i] += spectrum[b]
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    times = np.arange(n_frames) * hop / sr
    return {"times": times, "pitch_classes": pitch_classes, "energies": chroma}


def loudness_curve(
    signal: np.ndarray, sr: int, frame_ms: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_size = int(sr * frame_ms / 1000.0)
    n_frames = max(1, len(signal) // frame_size)
    times = np.zeros(n_frames)
    levels = np.zeros(n_frames)
    for i in range(n_frames):
        chunk = signal[i * frame_size : (i + 1) * frame_size]
        ms = np.mean(chunk ** 2)
        levels[i] = -0.691 + 10.0 * np.log10(max(ms, 1e-10))
        times[i] = (i + 0.5) * frame_ms / 1000.0
    return times, levels


def stereo_correlation(
    left: np.ndarray, right: np.ndarray, frame_size: int = 4096,
) -> np.ndarray:
    n = min(len(left), len(right))
    n_frames = max(1, n // frame_size)
    corr = np.zeros(n_frames)
    for i in range(n_frames):
        l_chunk = left[i * frame_size : (i + 1) * frame_size]
        r_chunk = right[i * frame_size : (i + 1) * frame_size]
        l_z = l_chunk - np.mean(l_chunk)
        r_z = r_chunk - np.mean(r_chunk)
        denom = np.sqrt(np.sum(l_z ** 2) * np.sum(r_z ** 2))
        corr[i] = np.sum(l_z * r_z) / denom if denom > 1e-10 else 0.0
    return corr


def phase_correlation(left: np.ndarray, right: np.ndarray) -> float:
    n = min(len(left), len(right))
    l = left[:n]
    r = right[:n]
    l_z = l - np.mean(l)
    r_z = r - np.mean(r)
    denom = np.sqrt(np.sum(l_z ** 2) * np.sum(r_z ** 2))
    return float(np.sum(l_z * r_z) / denom) if denom > 1e-10 else 0.0


def frequency_spectrum_snapshot(
    signal: np.ndarray, sr: int, n_fft: int = 4096, center: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    start = max(0, center - n_fft // 2)
    frame = signal[start : start + n_fft]
    if len(frame) < n_fft:
        frame = np.pad(frame, (0, n_fft - len(frame)))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    spectrum = np.abs(np.fft.rfft(frame * window))
    mag_db = 20.0 * np.log10(np.maximum(spectrum, 1e-10))
    freqs = np.linspace(0, sr / 2, len(spectrum))
    return freqs, mag_db


def color_map_note(midi_note: int) -> tuple[float, float, float]:
    pc = midi_note % 12
    hue = pc / 12.0
    saturation = 0.75
    lightness = 0.55
    return (hue, saturation, lightness)
