"""Haptic music rendering: audio-to-vibrotactile translation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HapticChannel:
    frequency_range: tuple[float, float]
    intensity: float
    body_position: str


HAPTIC_LAYOUTS: dict[str, tuple[float, float]] = {
    "wrist": (20, 300),
    "chest": (20, 200),
    "back": (20, 250),
    "fingertip": (50, 500),
    "foot": (10, 150),
}


def audio_to_haptic(
    audio: np.ndarray,
    sr: int,
    n_channels: int = 4,
) -> list[np.ndarray]:
    """Decompose audio into frequency bands for tactile transducers."""
    n_fft = 2048
    hop = n_fft // 2
    n_frames = (len(audio) - n_fft) // hop + 1
    if n_frames <= 0:
        return [np.zeros(0) for _ in range(n_channels)]

    band_edges = np.logspace(np.log10(20), np.log10(min(500, sr // 2)), n_channels + 1)
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    channels = []

    for ch in range(n_channels):
        envelope = np.zeros(n_frames)
        lo, hi = band_edges[ch], band_edges[ch + 1]
        band_mask = (freq_bins >= lo) & (freq_bins < hi)
        for i in range(n_frames):
            frame = audio[i * hop: i * hop + n_fft] * np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(frame))
            envelope[i] = np.sqrt(np.mean(spectrum[band_mask] ** 2)) if band_mask.any() else 0.0
        channels.append(envelope)
    return channels


def rhythm_haptic(
    onsets: list[float],
    intensities: list[float],
    duration: float,
    sr: int = 250,
) -> np.ndarray:
    """Generate pulse train for rhythm haptics."""
    n_samples = int(duration * sr)
    signal = np.zeros(n_samples)
    pulse_width = max(1, int(0.02 * sr))  # 20ms pulse
    for t, intensity in zip(onsets, intensities):
        idx = int(t * sr)
        if 0 <= idx < n_samples:
            end = min(idx + pulse_width, n_samples)
            signal[idx:end] = min(1.0, abs(intensity))
    return signal


def bass_haptic(audio: np.ndarray, sr: int, cutoff: float = 200) -> np.ndarray:
    """Extract low-frequency content for subwoofer/vibrotactile."""
    n_fft = 2048
    hop = n_fft // 2
    n_frames = (len(audio) - n_fft) // hop + 1
    if n_frames <= 0:
        return np.zeros(0)
    freq_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    envelope = np.zeros(n_frames)
    mask = freq_bins <= cutoff
    for i in range(n_frames):
        frame = audio[i * hop: i * hop + n_fft] * np.hanning(n_fft)
        spectrum = np.abs(np.fft.rfft(frame))
        envelope[i] = np.sqrt(np.mean(spectrum[mask] ** 2))
    return envelope


def melody_contour_haptic(
    pitches: list[int],
    durations: list[float],
    sr: int = 250,
) -> np.ndarray:
    """Map melody pitch to vibration frequency for contour perception."""
    if not pitches:
        return np.zeros(0)
    total_dur = sum(durations)
    n_samples = int(total_dur * sr)
    signal = np.zeros(n_samples)
    idx = 0
    for pitch, dur in zip(pitches, durations):
        vib_freq = 20 + (pitch - 36) * 3.0  # map MIDI to ~20-300 Hz
        vib_freq = max(20, min(300, vib_freq))
        n = int(dur * sr)
        t = np.arange(n) / sr
        signal[idx: idx + n] = 0.5 * np.sin(2 * np.pi * vib_freq * t)
        idx += n
        if idx >= n_samples:
            break
    return signal[:n_samples]
