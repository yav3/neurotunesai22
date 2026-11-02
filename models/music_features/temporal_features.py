"""Time-domain and dynamic audio features.

Implements RMS energy, zero-crossing rate, amplitude envelope descriptors,
crest factor, dynamic range, loudness range, and modulation spectrum.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def rms_energy(signal: np.ndarray, frame_size: int = 2048,
               hop_size: int = 512) -> np.ndarray:
    """Frame-wise RMS energy."""
    n = len(signal)
    n_frames = max(1, 1 + (n - frame_size) // hop_size)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + frame_size, n)
        frame = signal[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    return rms


def zero_crossing_rate(signal: np.ndarray, frame_size: int = 2048,
                        hop_size: int = 512) -> np.ndarray:
    """Frame-wise zero-crossing rate."""
    n = len(signal)
    n_frames = max(1, 1 + (n - frame_size) // hop_size)
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + frame_size, n)
        frame = signal[start:end]
        if len(frame) < 2:
            continue
        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2.0 * len(frame))
    return zcr


def amplitude_envelope(signal: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude via Hilbert transform."""
    analytic = hilbert(signal)
    return np.abs(analytic)


def attack_time(envelope: np.ndarray, sr: float = 22050.0,
                threshold_lo: float = 0.1, threshold_hi: float = 0.9) -> float:
    """Time (seconds) from *threshold_lo* to *threshold_hi* of peak amplitude."""
    peak = np.max(envelope)
    if peak < 1e-12:
        return 0.0
    env_norm = envelope / peak
    lo_idx = np.argmax(env_norm >= threshold_lo)
    hi_idx = np.argmax(env_norm >= threshold_hi)
    if hi_idx <= lo_idx:
        return 0.0
    return float((hi_idx - lo_idx) / sr)


def decay_time(envelope: np.ndarray, sr: float = 22050.0,
               threshold: float = 0.1) -> float:
    """Time (seconds) from peak to *threshold* of peak amplitude."""
    peak_idx = np.argmax(envelope)
    peak = envelope[peak_idx]
    if peak < 1e-12:
        return 0.0
    env_after = envelope[peak_idx:]
    below = np.where(env_after / peak < threshold)[0]
    if len(below) == 0:
        return float(len(env_after) / sr)
    return float(below[0] / sr)


def sustain_level(envelope: np.ndarray, sustain_start: float = 0.3,
                  sustain_end: float = 0.7) -> float:
    """Mean amplitude in the middle portion of the envelope, normalized to peak."""
    peak = np.max(envelope)
    if peak < 1e-12:
        return 0.0
    n = len(envelope)
    s = int(sustain_start * n)
    e = int(sustain_end * n)
    if e <= s:
        return 0.0
    return float(np.mean(envelope[s:e]) / peak)


def temporal_centroid(signal: np.ndarray, sr: float = 22050.0) -> float:
    """Center of mass of the energy distribution (seconds)."""
    energy = signal ** 2
    total = np.sum(energy)
    if total < 1e-12:
        return 0.0
    times = np.arange(len(signal)) / sr
    return float(np.sum(times * energy) / total)


def effective_duration(signal: np.ndarray, sr: float = 22050.0,
                       threshold_db: float = -40.0) -> float:
    """Duration (seconds) where signal exceeds threshold relative to peak."""
    peak = np.max(np.abs(signal))
    if peak < 1e-12:
        return 0.0
    threshold_lin = peak * 10 ** (threshold_db / 20.0)
    above = np.abs(signal) >= threshold_lin
    if not np.any(above):
        return 0.0
    first = np.argmax(above)
    last = len(above) - 1 - np.argmax(above[::-1])
    return float((last - first) / sr)


def crest_factor(signal: np.ndarray) -> float:
    """Peak-to-RMS ratio (linear). Higher = more transient/peaky."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-12:
        return 0.0
    return float(np.max(np.abs(signal)) / rms)


def dynamic_range(signal: np.ndarray, frame_size: int = 2048,
                  hop_size: int = 512, percentile_lo: float = 10,
                  percentile_hi: float = 95) -> float:
    """Difference between loud and quiet frame levels (dB)."""
    rms = rms_energy(signal, frame_size, hop_size)
    rms_db = 20 * np.log10(rms + 1e-10)
    lo = np.percentile(rms_db, percentile_lo)
    hi = np.percentile(rms_db, percentile_hi)
    return float(hi - lo)


def loudness_range(signal: np.ndarray, sr: float = 22050.0,
                   gate_threshold_db: float = -70.0) -> float:
    """EBU R128-inspired loudness range (LRA) in dB.

    Simplified: uses frame-based RMS with absolute gating.
    """
    frame_size = int(0.4 * sr)
    hop_size = frame_size
    rms = rms_energy(signal, frame_size, hop_size)
    rms_db = 20 * np.log10(rms + 1e-10)
    gated = rms_db[rms_db > gate_threshold_db]
    if len(gated) < 2:
        return 0.0
    return float(np.percentile(gated, 95) - np.percentile(gated, 10))


def modulation_spectrum(signal: np.ndarray, sr: float = 22050.0,
                        frame_size: int = 2048, hop_size: int = 512,
                        n_mod_freq: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude modulation spectrum — rate of energy fluctuation.

    Returns (mod_freqs_hz, mod_spectrum).
    """
    rms = rms_energy(signal, frame_size, hop_size)
    if len(rms) < 4:
        return np.zeros(n_mod_freq), np.zeros(n_mod_freq)

    rms_centered = rms - np.mean(rms)
    n_fft = max(n_mod_freq * 2, len(rms_centered))
    spectrum = np.abs(np.fft.rfft(rms_centered, n=n_fft))
    frame_rate = sr / hop_size
    mod_freqs = np.fft.rfftfreq(n_fft, d=1.0 / frame_rate)

    if len(spectrum) > n_mod_freq:
        spectrum = spectrum[:n_mod_freq]
        mod_freqs = mod_freqs[:n_mod_freq]
    elif len(spectrum) < n_mod_freq:
        spectrum = np.pad(spectrum, (0, n_mod_freq - len(spectrum)))
        mod_freqs = np.pad(mod_freqs, (0, n_mod_freq - len(mod_freqs)))

    return mod_freqs, spectrum


def temporal_feature_vector(signal: np.ndarray, sr: float = 22050.0) -> np.ndarray:
    """Concatenated temporal descriptor vector."""
    env = amplitude_envelope(signal)
    return np.array([
        np.sqrt(np.mean(signal ** 2)),
        np.mean(zero_crossing_rate(signal)),
        attack_time(env, sr),
        decay_time(env, sr),
        sustain_level(env),
        temporal_centroid(signal, sr),
        effective_duration(signal, sr),
        crest_factor(signal),
        dynamic_range(signal),
        loudness_range(signal, sr),
    ])
