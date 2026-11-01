"""Spectral modeling synthesis (SMS): sinusoidal + residual decomposition.

Sinusoidal peak tracking across STFT frames, residual extraction,
and spectral morphing between two sounds.  Pure numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


def stft_frames(
    signal: np.ndarray, frame_size: int = 2048, hop: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (magnitudes, phases) arrays of shape (n_frames, n_bins)."""
    win = _hann(frame_size)
    n = len(signal)
    n_frames = max(1, 1 + (n - frame_size) // hop)
    n_bins = frame_size // 2 + 1
    mags = np.zeros((n_frames, n_bins), dtype=np.float64)
    phases = np.zeros((n_frames, n_bins), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        end = start + frame_size
        chunk = signal[start:end]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        spectrum = np.fft.rfft(chunk * win)
        mags[i] = np.abs(spectrum)
        phases[i] = np.angle(spectrum)
    return mags, phases


@dataclass
class SinusoidalPeak:
    """A detected sinusoidal peak in one STFT frame."""
    bin_index: int
    frequency_hz: float
    amplitude: float
    phase: float


def detect_peaks(
    mag_frame: np.ndarray,
    sr: int = 44100,
    frame_size: int = 2048,
    threshold_db: float = -60.0,
    max_peaks: int = 40,
) -> list[SinusoidalPeak]:
    """Find spectral peaks in a single magnitude frame."""
    n_bins = len(mag_frame)
    if n_bins < 3:
        return []
    threshold_lin = 10.0 ** (threshold_db / 20.0) * np.max(mag_frame + 1e-30)
    peaks: list[SinusoidalPeak] = []
    for i in range(1, n_bins - 1):
        if mag_frame[i] > mag_frame[i - 1] and mag_frame[i] > mag_frame[i + 1]:
            if mag_frame[i] >= threshold_lin:
                freq = i * sr / frame_size
                peaks.append(SinusoidalPeak(
                    bin_index=i, frequency_hz=freq,
                    amplitude=float(mag_frame[i]), phase=0.0,
                ))
    peaks.sort(key=lambda p: -p.amplitude)
    return peaks[:max_peaks]


@dataclass
class SinusoidalTrack:
    """A tracked sinusoidal partial across frames."""
    frequencies: list[float] = field(default_factory=list)
    amplitudes: list[float] = field(default_factory=list)
    phases: list[float] = field(default_factory=list)
    start_frame: int = 0

    @property
    def n_frames(self) -> int:
        return len(self.frequencies)

    @property
    def mean_frequency(self) -> float:
        return float(np.mean(self.frequencies)) if self.frequencies else 0.0


def track_sinusoids(
    mags: np.ndarray,
    phases: np.ndarray,
    sr: int = 44100,
    frame_size: int = 2048,
    max_peaks: int = 40,
    freq_tolerance_hz: float = 50.0,
    min_track_frames: int = 3,
) -> list[SinusoidalTrack]:
    """Track sinusoidal peaks across STFT frames via nearest-frequency matching."""
    n_frames = mags.shape[0]
    active: list[SinusoidalTrack] = []
    finished: list[SinusoidalTrack] = []

    for fi in range(n_frames):
        peaks = detect_peaks(mags[fi], sr, frame_size, max_peaks=max_peaks)
        used_peaks: set[int] = set()
        continued: list[SinusoidalTrack] = []

        for track in active:
            last_freq = track.frequencies[-1]
            best_idx = -1
            best_dist = freq_tolerance_hz
            for pi, pk in enumerate(peaks):
                if pi in used_peaks:
                    continue
                d = abs(pk.frequency_hz - last_freq)
                if d < best_dist:
                    best_dist = d
                    best_idx = pi
            if best_idx >= 0:
                pk = peaks[best_idx]
                track.frequencies.append(pk.frequency_hz)
                track.amplitudes.append(pk.amplitude)
                track.phases.append(float(phases[fi, pk.bin_index]))
                used_peaks.add(best_idx)
                continued.append(track)
            else:
                finished.append(track)

        for pi, pk in enumerate(peaks):
            if pi not in used_peaks:
                t = SinusoidalTrack(
                    frequencies=[pk.frequency_hz],
                    amplitudes=[pk.amplitude],
                    phases=[float(phases[fi, pk.bin_index])],
                    start_frame=fi,
                )
                continued.append(t)
        active = continued

    finished.extend(active)
    return [t for t in finished if t.n_frames >= min_track_frames]


def resynthesize_sinusoids(
    tracks: list[SinusoidalTrack],
    n_samples: int,
    sr: int = 44100,
    hop: int = 512,
) -> np.ndarray:
    """Overlap-add resynthesis from sinusoidal tracks."""
    out = np.zeros(n_samples, dtype=np.float64)
    for track in tracks:
        for fi in range(track.n_frames):
            frame_idx = track.start_frame + fi
            start = frame_idx * hop
            end = min(start + hop, n_samples)
            if start >= n_samples:
                break
            length = end - start
            t = np.arange(length, dtype=np.float64) / sr
            freq = track.frequencies[fi]
            amp = track.amplitudes[fi]
            phase = track.phases[fi]
            out[start:end] += amp * np.sin(2.0 * np.pi * freq * t + phase)
    mx = np.max(np.abs(out))
    if mx > 0:
        out /= mx
    return out


def extract_residual(
    original: np.ndarray,
    sinusoidal: np.ndarray,
) -> np.ndarray:
    """Residual = original - sinusoidal reconstruction (length-matched)."""
    n = min(len(original), len(sinusoidal))
    return original[:n] - sinusoidal[:n]


def spectral_morph(
    mags_a: np.ndarray,
    mags_b: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Interpolate magnitude spectra: (1 - alpha)*A + alpha*B."""
    alpha = np.clip(alpha, 0.0, 1.0)
    na, nb = mags_a.shape[0], mags_b.shape[0]
    n_frames = min(na, nb)
    n_bins = min(mags_a.shape[1], mags_b.shape[1])
    return (1.0 - alpha) * mags_a[:n_frames, :n_bins] + alpha * mags_b[:n_frames, :n_bins]


def sms_analyze(
    signal: np.ndarray,
    sr: int = 44100,
    frame_size: int = 2048,
    hop: int = 512,
    max_peaks: int = 40,
) -> tuple[list[SinusoidalTrack], np.ndarray]:
    """Full SMS analysis: returns (sinusoidal tracks, residual signal)."""
    mags, phases = stft_frames(signal, frame_size, hop)
    tracks = track_sinusoids(mags, phases, sr, frame_size, max_peaks=max_peaks)
    sin_resynth = resynthesize_sinusoids(tracks, len(signal), sr, hop)
    residual = extract_residual(signal, sin_resynth)
    return tracks, residual
