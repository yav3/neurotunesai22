"""Adaptive and non-uniform filterbank design for music signal analysis."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class GammatoneParams:
    center_freq: float
    bandwidth: float
    order: int = 4
    phase: float = 0.0


def erb_space(f_low: float, f_high: float, n_filters: int) -> np.ndarray:
    """Center frequencies spaced on the ERB-rate scale."""
    ear_q = 9.26449
    min_bw = 24.7
    low_erb = ear_q * np.log(1.0 + f_low / (ear_q * min_bw))
    high_erb = ear_q * np.log(1.0 + f_high / (ear_q * min_bw))
    erb_points = np.linspace(low_erb, high_erb, n_filters)
    return (np.exp(erb_points / ear_q) - 1.0) * ear_q * min_bw


def erb_bandwidth(f_center: float) -> float:
    """Equivalent rectangular bandwidth at center frequency (Glasberg & Moore 1990)."""
    return 24.7 * (4.37e-3 * f_center + 1.0)


def gammatone_ir(params: GammatoneParams, sample_rate: int,
                 duration: float = 0.05) -> np.ndarray:
    """Gammatone filter impulse response."""
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    b = 2.0 * np.pi * 1.019 * params.bandwidth
    ir = (t ** (params.order - 1)) * np.exp(-b * t) * np.cos(
        2.0 * np.pi * params.center_freq * t + params.phase
    )
    ir /= np.max(np.abs(ir)) + 1e-12
    return ir


def apply_gammatone_bank(signal: np.ndarray, center_freqs: np.ndarray,
                         sample_rate: int, order: int = 4) -> np.ndarray:
    """Apply gammatone filterbank; returns (n_filters, n_samples)."""
    n_filters = len(center_freqs)
    n_samples = len(signal)
    output = np.zeros((n_filters, n_samples))
    sig_fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    for i, cf in enumerate(center_freqs):
        bw = erb_bandwidth(cf)
        params = GammatoneParams(cf, bw, order)
        ir = gammatone_ir(params, sample_rate, duration=min(0.05, n_samples / sample_rate))
        ir_padded = np.zeros(n_samples)
        ir_padded[:len(ir)] = ir
        ir_fft = np.fft.rfft(ir_padded)
        output[i] = np.fft.irfft(sig_fft * ir_fft, n=n_samples)
    return output


def bark_critical_band(f_center: float) -> float:
    """Critical bandwidth in Hz at a given center frequency (Zwicker)."""
    return 25.0 + 75.0 * (1.0 + 1.4e-6 * f_center ** 2) ** 0.69


def bark_space(f_low: float, f_high: float, n_filters: int) -> np.ndarray:
    """Center frequencies spaced on the Bark scale."""
    def hz_to_bark(f: float) -> float:
        return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

    def bark_to_hz(z: float) -> float:
        return 600.0 * np.sinh(z / 6.0)

    low_bark = hz_to_bark(f_low)
    high_bark = hz_to_bark(f_high)
    bark_points = np.linspace(low_bark, high_bark, n_filters)
    return np.array([bark_to_hz(z) for z in bark_points])


def constant_q_bank(f_min: float, f_max: float, bins_per_octave: int,
                    sample_rate: int) -> List[np.ndarray]:
    """Constant-Q filter kernels with Q = bins_per_octave / (2^(1/bpo) - 1)."""
    q_factor = bins_per_octave / (2.0 ** (1.0 / bins_per_octave) - 1.0)
    n_bins = int(np.ceil(bins_per_octave * np.log2(f_max / f_min)))
    kernels = []
    for k in range(n_bins):
        fk = f_min * (2.0 ** (k / bins_per_octave))
        n_k = int(np.ceil(q_factor * sample_rate / fk))
        if n_k < 1:
            n_k = 1
        t = np.arange(n_k) / sample_rate
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_k) / n_k))
        kernel = window * np.exp(2j * np.pi * fk * t) / n_k
        kernels.append(kernel)
    return kernels


def wavelet_packet_bank(signal: np.ndarray, max_level: int = 4
                        ) -> Dict[Tuple[int, int], np.ndarray]:
    """Wavelet packet decomposition using Haar wavelets."""
    lo = np.array([1.0, 1.0]) / np.sqrt(2.0)
    hi = np.array([1.0, -1.0]) / np.sqrt(2.0)
    packets: Dict[Tuple[int, int], np.ndarray] = {(0, 0): signal.copy()}
    for level in range(1, max_level + 1):
        for node in range(2 ** (level - 1)):
            parent = packets.get((level - 1, node))
            if parent is None:
                continue
            n = len(parent)
            if n < 2:
                packets[(level, 2 * node)] = parent
                packets[(level, 2 * node + 1)] = np.zeros(1)
                continue
            approx = np.convolve(parent, lo, mode='full')[1::2]
            detail = np.convolve(parent, hi, mode='full')[1::2]
            packets[(level, 2 * node)] = approx
            packets[(level, 2 * node + 1)] = detail
    return packets


def adaptive_resolution(signal: np.ndarray, sample_rate: int,
                        energy_threshold: float = 0.1,
                        min_window: int = 64, max_window: int = 4096
                        ) -> List[Dict[str, np.ndarray]]:
    """Variable-resolution time-frequency: finer where energy is concentrated."""
    total_energy = np.sum(signal ** 2)
    if total_energy < 1e-12:
        return []
    frames = []
    pos = 0
    n = len(signal)
    while pos < n:
        window_size = min_window
        while window_size < max_window and pos + window_size < n:
            chunk = signal[pos:pos + window_size]
            chunk_energy = np.sum(chunk ** 2) / total_energy
            if chunk_energy > energy_threshold:
                break
            window_size *= 2
        window_size = min(window_size, n - pos)
        chunk = signal[pos:pos + window_size]
        spectrum = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
        frames.append({
            "time": np.array([pos / sample_rate]),
            "frequencies": freqs,
            "magnitudes": spectrum,
            "window_size": np.array([window_size]),
        })
        pos += window_size
    return frames


def subband_envelope(filtered_signals: np.ndarray) -> np.ndarray:
    """Hilbert envelope per subband; input shape (n_bands, n_samples)."""
    analytic = np.zeros_like(filtered_signals, dtype=complex)
    for i in range(filtered_signals.shape[0]):
        n = filtered_signals.shape[1]
        fft_sig = np.fft.fft(filtered_signals[i])
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0
        analytic[i] = np.fft.ifft(fft_sig * h)
    return np.abs(analytic)


def cochleagram(signal: np.ndarray, sample_rate: int,
                n_filters: int = 64) -> np.ndarray:
    """Auditory-model spectrogram via gammatone filterbank + envelope."""
    cfs = erb_space(50.0, sample_rate / 2.0 - 1.0, n_filters)
    filtered = apply_gammatone_bank(signal, cfs, sample_rate)
    envelopes = subband_envelope(filtered)
    hop = max(1, sample_rate // 100)
    n_frames = len(signal) // hop
    cochlea = np.zeros((n_filters, n_frames))
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(signal))
        cochlea[:, i] = np.mean(envelopes[:, start:end], axis=1)
    return cochlea


def modulation_spectrum(envelopes: np.ndarray, sample_rate: int,
                        hop_rate: int = 100) -> np.ndarray:
    """Per-band modulation spectrum from subband envelopes."""
    n_bands, n_samples = envelopes.shape
    mod_sr = hop_rate
    n_fft = max(64, n_samples)
    mod_freqs = np.fft.rfftfreq(n_fft, 1.0 / mod_sr)
    result = np.zeros((n_bands, len(mod_freqs)))
    for i in range(n_bands):
        padded = np.zeros(n_fft)
        padded[:n_samples] = envelopes[i]
        result[i] = np.abs(np.fft.rfft(padded))
    return result


def mel_gammatone_bank(signal: np.ndarray, sample_rate: int,
                       n_filters: int = 40, f_min: float = 64.0
                       ) -> np.ndarray:
    """Mel-spaced gammatone filterbank for robust pitch/timbre analysis."""
    f_max = sample_rate / 2.0 - 1.0
    mel_low = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_high = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_filters)
    cfs = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    return apply_gammatone_bank(signal, cfs, sample_rate)


def filter_energy_profile(filtered_signals: np.ndarray) -> np.ndarray:
    """RMS energy per band; input (n_bands, n_samples)."""
    return np.sqrt(np.mean(filtered_signals ** 2, axis=1))
