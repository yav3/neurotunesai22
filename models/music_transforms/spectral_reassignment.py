"""Spectral reassignment and synchrosqueezing: improved time-frequency resolution.

Provides reassigned spectrograms, instantaneous frequency estimation,
group delay, synchrosqueezed transforms, and identity phase-locked vocoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sig
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Derivative windows for reassignment
# ---------------------------------------------------------------------------

def _time_derivative_window(window: np.ndarray) -> np.ndarray:
    """Compute the time-ramped window: t · w(t)."""
    n = len(window)
    t = np.arange(n) - (n - 1) / 2.0
    return t * window


def _frequency_derivative_window(window: np.ndarray) -> np.ndarray:
    """Compute the frequency-derivative window: dw/dt (central differences)."""
    dw = np.zeros_like(window)
    dw[1:-1] = (window[2:] - window[:-2]) / 2.0
    dw[0] = window[1] - window[0]
    dw[-1] = window[-1] - window[-2]
    return dw


# ---------------------------------------------------------------------------
# STFT helpers
# ---------------------------------------------------------------------------

def _stft(x: np.ndarray, window: np.ndarray, hop: int, n_fft: int) -> np.ndarray:
    """Short-time Fourier transform. Returns (n_freq, n_frames) complex."""
    n = len(x)
    win_len = len(window)
    n_frames = 1 + (n - win_len) // hop
    n_freq = n_fft // 2 + 1
    S = np.zeros((n_freq, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        frame = x[start : start + win_len] * window
        S[:, i] = np.fft.rfft(frame, n=n_fft)
    return S


# ---------------------------------------------------------------------------
# Reassigned spectrogram
# ---------------------------------------------------------------------------

@dataclass
class ReassignedSpectrogram:
    magnitude: np.ndarray       # (n_freq, n_frames)
    reassigned_times: np.ndarray  # (n_freq, n_frames)
    reassigned_freqs: np.ndarray  # (n_freq, n_frames) in Hz
    original_times: np.ndarray
    original_freqs: np.ndarray


def reassigned_spectrogram(x: np.ndarray, sr: float,
                           n_fft: int = 2048, hop: int = 512,
                           window_type: str = "hann") -> ReassignedSpectrogram:
    """Compute the reassigned spectrogram via Auger-Flandrin method."""
    window = sig.get_window(window_type, n_fft)
    tw = _time_derivative_window(window)
    dw = _frequency_derivative_window(window)

    S = _stft(x, window, hop, n_fft)
    S_tw = _stft(x, tw, hop, n_fft)
    S_dw = _stft(x, dw, hop, n_fft)

    n_freq, n_frames = S.shape
    mag = np.abs(S)
    eps = 1e-10

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = np.arange(n_frames) * hop / sr

    # Instantaneous frequency via frequency-derivative window
    inst_freq = np.zeros_like(mag)
    for t in range(n_frames):
        for f in range(n_freq):
            if mag[f, t] > eps:
                inst_freq[f, t] = freqs[f] - np.imag(S_dw[f, t] / (S[f, t] + eps)) * sr / (2 * np.pi)
            else:
                inst_freq[f, t] = freqs[f]

    # Reassigned time via time-derivative window
    r_times = np.zeros_like(mag)
    for t in range(n_frames):
        for f in range(n_freq):
            if mag[f, t] > eps:
                r_times[f, t] = times[t] + np.real(S_tw[f, t] / (S[f, t] + eps)) / sr
            else:
                r_times[f, t] = times[t]

    return ReassignedSpectrogram(
        magnitude=mag,
        reassigned_times=r_times,
        reassigned_freqs=np.clip(inst_freq, 0, sr / 2),
        original_times=times,
        original_freqs=freqs,
    )


# ---------------------------------------------------------------------------
# Instantaneous frequency
# ---------------------------------------------------------------------------

def instantaneous_frequency(S: np.ndarray, hop: int, sr: float) -> np.ndarray:
    """Estimate IF from STFT phase differences between consecutive frames.

    S: complex STFT (n_freq, n_frames).
    Returns IF in Hz (n_freq, n_frames-1).
    """
    phase = np.angle(S)
    dp = np.diff(phase, axis=1)
    # Unwrap
    dp = dp - 2 * np.pi * np.round(dp / (2 * np.pi))

    expected = 2 * np.pi * np.arange(S.shape[0])[:, None] * hop / (2 * (S.shape[0] - 1))
    inst_freq = (dp + expected) * sr / (2 * np.pi * hop)
    return np.clip(inst_freq, 0, sr / 2)


# ---------------------------------------------------------------------------
# Group delay
# ---------------------------------------------------------------------------

def group_delay(S: np.ndarray, S_dw: np.ndarray) -> np.ndarray:
    """Compute group delay from STFT and its frequency-derivative STFT.

    Returns group delay in frames (n_freq, n_frames).
    """
    mag = np.abs(S)
    eps = 1e-10
    gd = -np.real(S_dw * np.conj(S)) / (mag ** 2 + eps)
    return gd


# ---------------------------------------------------------------------------
# Synchrosqueezed transform
# ---------------------------------------------------------------------------

@dataclass
class SynchrosqueezedResult:
    squeezed: np.ndarray  # (n_freq, n_frames) magnitude
    freqs: np.ndarray


def synchrosqueeze(S: np.ndarray, inst_freq: np.ndarray,
                   sr: float, n_fft: int) -> SynchrosqueezedResult:
    """Synchrosqueeze an STFT by reassigning energy to IF-determined bins."""
    n_freq, n_frames = S.shape
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    mag = np.abs(S)

    squeezed = np.zeros_like(mag)
    for t in range(min(n_frames, inst_freq.shape[1])):
        for f in range(n_freq):
            target_bin = int(round(inst_freq[f, t] / (freq_res + 1e-10)))
            target_bin = max(0, min(target_bin, n_freq - 1))
            squeezed[target_bin, t] += mag[f, t]

    return SynchrosqueezedResult(squeezed=squeezed, freqs=freqs)


# ---------------------------------------------------------------------------
# Spectral peak tracking via Hungarian algorithm
# ---------------------------------------------------------------------------

@dataclass
class PartialTrack:
    start_frame: int
    freq_trajectory: list[float]
    amp_trajectory: list[float]


def track_spectral_peaks(peak_freqs: list[np.ndarray],
                         peak_amps: list[np.ndarray],
                         max_freq_jump: float = 50.0) -> list[PartialTrack]:
    """Link spectral peaks across frames using the Hungarian algorithm.

    peak_freqs: list of arrays, one per frame, each containing peak frequencies.
    peak_amps: list of arrays, one per frame, each containing peak amplitudes.
    """
    n_frames = len(peak_freqs)
    if n_frames == 0:
        return []

    active: list[PartialTrack] = []
    for i, freq in enumerate(peak_freqs[0]):
        active.append(PartialTrack(0, [float(freq)], [float(peak_amps[0][i])]))

    for t in range(1, n_frames):
        new_freqs = peak_freqs[t]
        new_amps = peak_amps[t]
        n_active = len(active)
        n_new = len(new_freqs)

        if n_active == 0 or n_new == 0:
            for i in range(n_new):
                active.append(PartialTrack(t, [float(new_freqs[i])], [float(new_amps[i])]))
            continue

        # Cost matrix: frequency distance
        cost = np.full((n_active, n_new), max_freq_jump * 10)
        for i in range(n_active):
            last_f = active[i].freq_trajectory[-1]
            for j in range(n_new):
                d = abs(last_f - new_freqs[j])
                if d <= max_freq_jump:
                    cost[i, j] = d

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_new = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= max_freq_jump:
                active[r].freq_trajectory.append(float(new_freqs[c]))
                active[r].amp_trajectory.append(float(new_amps[c]))
                matched_new.add(c)

        # Birth new tracks for unmatched peaks
        for j in range(n_new):
            if j not in matched_new:
                active.append(PartialTrack(t, [float(new_freqs[j])], [float(new_amps[j])]))

    return active


# ---------------------------------------------------------------------------
# Identity phase-locked vocoder for time-stretching
# ---------------------------------------------------------------------------

def identity_phase_lock_vocoder(S: np.ndarray, stretch_factor: float,
                                hop: int) -> np.ndarray:
    """Phase-locked phase vocoder for time-stretching.

    Propagates phase from spectral peaks to surrounding bins,
    preserving vertical phase coherence (identity locking).
    """
    n_freq, n_frames = S.shape
    new_n_frames = int(n_frames * stretch_factor)

    mag = np.abs(S)
    phase = np.angle(S)

    # Phase advance per hop
    omega = 2 * np.pi * np.arange(n_freq) * hop / (2 * (n_freq - 1))

    new_phase = np.zeros((n_freq, new_n_frames))
    new_mag = np.zeros((n_freq, new_n_frames))

    for t in range(new_n_frames):
        src = t / stretch_factor
        t0 = int(src)
        frac = src - t0
        t0 = min(t0, n_frames - 1)
        t1 = min(t0 + 1, n_frames - 1)

        # Interpolate magnitude
        new_mag[:, t] = (1 - frac) * mag[:, t0] + frac * mag[:, t1]

        if t == 0:
            new_phase[:, t] = phase[:, 0]
        else:
            # Phase propagation with peak locking
            dphi = phase[:, min(t1, n_frames - 1)] - phase[:, t0]
            dphi -= omega
            dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
            inst_freq = omega + dphi

            # Find peaks in current magnitude frame
            peaks = sig.argrelmax(new_mag[:, t], order=2)[0]

            new_phase[:, t] = new_phase[:, t - 1] + inst_freq * stretch_factor

            # Lock non-peak bins to nearest peak's phase
            if len(peaks) > 0:
                for f in range(n_freq):
                    nearest_peak = peaks[np.argmin(np.abs(peaks - f))]
                    phase_diff = new_phase[nearest_peak, t] - new_phase[f, t]
                    region_diff = phase[nearest_peak, t0] - phase[f, t0]
                    new_phase[f, t] = new_phase[nearest_peak, t] - region_diff

    return new_mag * np.exp(1j * new_phase)
