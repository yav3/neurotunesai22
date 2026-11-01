"""Neural entrainment — PLV, ITC, FFR, ASSR, beat alignment.

Exact implementations using scipy.signal Hilbert transform and numpy FFT.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def phase_locking_value(signal1: np.ndarray, signal2: np.ndarray) -> dict:
    analytic1 = hilbert(np.asarray(signal1, dtype=float))
    analytic2 = hilbert(np.asarray(signal2, dtype=float))
    phase_diff = np.angle(analytic1) - np.angle(analytic2)
    plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
    mean_direction = float(np.angle(np.mean(np.exp(1j * phase_diff))))
    return {"plv": plv, "mean_phase_difference": mean_direction, "significant": plv > 0.3}


def inter_trial_coherence(trials: np.ndarray, fs: float) -> dict:
    arr = np.atleast_2d(np.asarray(trials, dtype=float))
    n_trials, n_samples = arr.shape
    if n_trials == 0 or n_samples <= 1:
        return {"itc": [], "frequencies": [], "peak_itc": 0.0, "peak_frequency": 0.0}
    fft_trials = np.fft.rfft(arr, n=n_samples, axis=1)
    phases = np.angle(fft_trials)
    itc = np.abs(np.mean(np.exp(1j * phases), axis=0))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    itc_ac = itc[1:]
    freqs_ac = freqs[1:]
    return {
        "itc": itc.tolist(),
        "frequencies": freqs.tolist(),
        "peak_itc": float(np.max(itc_ac)) if itc_ac.size else 0.0,
        "peak_frequency": float(freqs_ac[np.argmax(itc_ac)]) if itc_ac.size else 0.0,
    }


def frequency_following_response(eeg: np.ndarray, stimulus_freq: float,
                                 fs: float, bandwidth: float = 2.0) -> dict:
    eeg = np.asarray(eeg, dtype=float)
    n = len(eeg)
    if n == 0 or fs <= 0:
        return {"snr": 0.0, "power_at_freq": 0.0, "noise_floor": 0.0,
                "harmonic_snrs": [], "significant": False}
    fft_mag = np.abs(np.fft.rfft(eeg))
    freq_resolution = fs / n
    target_idx = int(round(stimulus_freq / freq_resolution))
    if target_idx >= len(fft_mag):
        return {"snr": 0.0, "power_at_freq": 0.0, "noise_floor": 0.0,
                "harmonic_snrs": [], "significant": False}
    signal_power = float(fft_mag[target_idx] ** 2)
    bw_bins = max(1, int(bandwidth / freq_resolution))
    noise_indices = list(range(max(1, target_idx - 5 * bw_bins), target_idx - bw_bins))
    noise_indices += list(range(target_idx + bw_bins + 1,
                                min(len(fft_mag), target_idx + 5 * bw_bins + 1)))
    noise_power = float(np.mean(fft_mag[noise_indices] ** 2)) if noise_indices else 1e-12
    snr = float(10 * np.log10(signal_power / (noise_power + 1e-12)))
    harmonic_snrs = []
    for h in range(2, 5):
        h_idx = int(round(h * stimulus_freq / freq_resolution))
        if h_idx < len(fft_mag):
            h_power = float(fft_mag[h_idx] ** 2)
            harmonic_snrs.append(float(10 * np.log10(h_power / (noise_power + 1e-12))))
    return {
        "snr": snr,
        "power_at_freq": signal_power,
        "noise_floor": noise_power,
        "harmonic_snrs": harmonic_snrs,
        "significant": snr > 6.0,
    }


def assr(eeg: np.ndarray, modulation_freq: float, fs: float) -> dict:
    return frequency_following_response(eeg, modulation_freq, fs, bandwidth=1.0)


def beat_alignment(neural_signal: np.ndarray, beat_times: np.ndarray,
                   fs: float) -> dict:
    analytic = hilbert(np.asarray(neural_signal, dtype=float))
    phases_at_beats = []
    for bt in np.asarray(beat_times, dtype=float):
        idx = int(bt * fs)
        if 0 <= idx < len(analytic):
            phases_at_beats.append(np.angle(analytic[idx]))
    if len(phases_at_beats) < 3:
        return {"alignment": 0.0, "mean_phase": 0.0, "rayleigh_p": 1.0,
                "significant": False, "n_beats": 0}
    phases = np.array(phases_at_beats)
    n = len(phases)
    r = float(np.abs(np.mean(np.exp(1j * phases))))
    mean_phase = float(np.angle(np.mean(np.exp(1j * phases))))
    z = n * r ** 2
    p = float(np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n)))
    p = max(0.0, min(p, 1.0))
    return {
        "alignment": r,
        "mean_phase": mean_phase,
        "rayleigh_z": float(z),
        "rayleigh_p": p,
        "significant": p < 0.05,
        "n_beats": n,
    }
