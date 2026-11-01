"""Audio watermarking: spread-spectrum embedding and QIM extraction.

Implements imperceptible audio watermark embedding for ownership
verification and content tracking. Two primary methods:
- Spread-spectrum (SS): multiply-accumulate with PN sequence in frequency domain
- Quantization Index Modulation (QIM): dither-modulated quantizer per subband
Both operate in the STFT domain for temporal localization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.fft import fft, ifft


@dataclass
class WatermarkConfig:
    key_seed: int = 42
    embed_strength: float = 0.01
    segment_length: int = 1024
    hop_length: int = 512
    freq_min_bin: int = 10
    freq_max_bin: int = 200


@dataclass
class WatermarkResult:
    watermarked_signal: np.ndarray
    snr_db: float
    payload_bits: int
    method: str


def _generate_pn_sequence(length: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=length).astype(np.float64)


def spread_spectrum_embed(signal: np.ndarray, payload: Sequence[int],
                           config: WatermarkConfig | None = None) -> WatermarkResult:
    cfg = config or WatermarkConfig()
    sig = np.array(signal, dtype=np.float64)
    n_bins = cfg.freq_max_bin - cfg.freq_min_bin
    n_segments = (len(sig) - cfg.segment_length) // cfg.hop_length + 1
    bits_per_segment = 1
    n_bits = min(len(payload), n_segments)
    watermarked = sig.copy()
    for i in range(n_bits):
        start = i * cfg.hop_length
        end = start + cfg.segment_length
        if end > len(sig):
            break
        segment = sig[start:end]
        spectrum = fft(segment)
        pn = _generate_pn_sequence(n_bins, cfg.key_seed + i)
        bit = 1.0 if payload[i] else -1.0
        spectrum[cfg.freq_min_bin:cfg.freq_max_bin] += cfg.embed_strength * bit * pn * np.abs(spectrum[cfg.freq_min_bin:cfg.freq_max_bin])
        modified = np.real(ifft(spectrum))
        watermarked[start:end] = modified
    noise_power = np.mean((watermarked - sig) ** 2)
    signal_power = np.mean(sig ** 2) + 1e-15
    snr = 10 * np.log10(signal_power / (noise_power + 1e-15))
    return WatermarkResult(
        watermarked_signal=watermarked,
        snr_db=float(snr),
        payload_bits=n_bits,
        method="spread_spectrum",
    )


def spread_spectrum_extract(signal: np.ndarray, n_bits: int,
                             config: WatermarkConfig | None = None) -> list[int]:
    cfg = config or WatermarkConfig()
    sig = np.array(signal, dtype=np.float64)
    n_bins = cfg.freq_max_bin - cfg.freq_min_bin
    bits = []
    for i in range(n_bits):
        start = i * cfg.hop_length
        end = start + cfg.segment_length
        if end > len(sig):
            break
        segment = sig[start:end]
        spectrum = fft(segment)
        pn = _generate_pn_sequence(n_bins, cfg.key_seed + i)
        correlation = np.real(np.sum(spectrum[cfg.freq_min_bin:cfg.freq_max_bin] * pn))
        bits.append(1 if correlation > 0 else 0)
    return bits


def qim_embed(signal: np.ndarray, payload: Sequence[int],
              step_size: float = 0.005,
              config: WatermarkConfig | None = None) -> WatermarkResult:
    cfg = config or WatermarkConfig()
    sig = np.array(signal, dtype=np.float64)
    n_segments = (len(sig) - cfg.segment_length) // cfg.hop_length + 1
    n_bits = min(len(payload), n_segments)
    watermarked = sig.copy()
    delta = step_size
    for i in range(n_bits):
        start = i * cfg.hop_length
        end = start + cfg.segment_length
        if end > len(sig):
            break
        segment = sig[start:end]
        spectrum = fft(segment)
        mag = np.abs(spectrum[cfg.freq_min_bin:cfg.freq_max_bin])
        phase = np.angle(spectrum[cfg.freq_min_bin:cfg.freq_max_bin])
        mean_mag = np.mean(mag)
        bit = payload[i]
        if bit:
            quantized = delta * np.round(mean_mag / delta) + delta / 4
        else:
            quantized = delta * np.round(mean_mag / delta) - delta / 4
        scale = quantized / (mean_mag + 1e-15)
        spectrum[cfg.freq_min_bin:cfg.freq_max_bin] = scale * mag * np.exp(1j * phase)
        modified = np.real(ifft(spectrum))
        watermarked[start:end] = modified
    noise_power = np.mean((watermarked - sig) ** 2)
    signal_power = np.mean(sig ** 2) + 1e-15
    snr = 10 * np.log10(signal_power / (noise_power + 1e-15))
    return WatermarkResult(
        watermarked_signal=watermarked,
        snr_db=float(snr),
        payload_bits=n_bits,
        method="qim",
    )


def qim_extract(signal: np.ndarray, n_bits: int,
                step_size: float = 0.005,
                config: WatermarkConfig | None = None) -> list[int]:
    cfg = config or WatermarkConfig()
    sig = np.array(signal, dtype=np.float64)
    delta = step_size
    bits = []
    for i in range(n_bits):
        start = i * cfg.hop_length
        end = start + cfg.segment_length
        if end > len(sig):
            break
        segment = sig[start:end]
        spectrum = fft(segment)
        mag = np.abs(spectrum[cfg.freq_min_bin:cfg.freq_max_bin])
        mean_mag = np.mean(mag)
        nearest = delta * np.round(mean_mag / delta)
        residual = mean_mag - nearest
        bits.append(1 if residual > 0 else 0)
    return bits


def watermark_robustness(original_bits: Sequence[int],
                          extracted_bits: Sequence[int]) -> float:
    if not original_bits or not extracted_bits:
        return 0.0
    n = min(len(original_bits), len(extracted_bits))
    matches = sum(1 for a, b in zip(original_bits[:n], extracted_bits[:n]) if a == b)
    return matches / n


def payload_capacity(signal_length: int, config: WatermarkConfig | None = None) -> int:
    cfg = config or WatermarkConfig()
    return max(0, (signal_length - cfg.segment_length) // cfg.hop_length + 1)
