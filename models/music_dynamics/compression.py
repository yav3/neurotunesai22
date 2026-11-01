"""Audio dynamics processing: compressor, expander, limiter, side-chain models.

Implements transfer functions and gain reduction models for dynamics processors
used in music production and mastering, including soft/hard knee compression,
multi-band splitting, and envelope followers with attack/release behavior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CompressorParams:
    threshold_db: float = -20.0
    ratio: float = 4.0
    knee_db: float = 6.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    makeup_gain_db: float = 0.0


@dataclass(frozen=True)
class GainReduction:
    input_db: np.ndarray
    output_db: np.ndarray
    gain_reduction_db: np.ndarray
    peak_reduction_db: float
    mean_reduction_db: float


@dataclass(frozen=True)
class MultibandConfig:
    crossover_freqs: list[float]
    per_band_params: list[CompressorParams]


def static_characteristic(input_db: float, params: CompressorParams) -> float:
    """Compressor static curve: hard or soft knee."""
    t = params.threshold_db
    r = params.ratio
    w = params.knee_db
    if w < 0.01:
        if input_db <= t:
            return input_db
        return t + (input_db - t) / r
    half_w = w / 2.0
    if input_db < t - half_w:
        return input_db
    if input_db > t + half_w:
        return t + (input_db - t) / r
    excess = input_db - t + half_w
    return input_db + (1.0 / r - 1.0) * excess ** 2 / (2.0 * w)


def expander_characteristic(
    input_db: float,
    threshold_db: float = -40.0,
    ratio: float = 2.0,
) -> float:
    """Expander/gate static curve: below threshold, reduce by ratio."""
    if input_db >= threshold_db:
        return input_db
    return threshold_db + (input_db - threshold_db) * ratio


def limiter_characteristic(input_db: float, ceiling_db: float = -1.0) -> float:
    """Brickwall limiter: infinite ratio above ceiling."""
    return min(input_db, ceiling_db)


def envelope_follower(
    signal: np.ndarray,
    sr: float,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    mode: str = "peak",
) -> np.ndarray:
    """Side-chain envelope follower with attack/release smoothing.

    mode: "peak" (rectified input) or "rms" (squared + sqrt).
    """
    attack_coeff = math.exp(-1.0 / (sr * attack_ms / 1000.0)) if attack_ms > 0 else 0.0
    release_coeff = math.exp(-1.0 / (sr * release_ms / 1000.0)) if release_ms > 0 else 0.0
    if mode == "rms":
        detector = signal ** 2
    else:
        detector = np.abs(signal)
    envelope = np.zeros(len(detector))
    env = 0.0
    for i in range(len(detector)):
        if detector[i] > env:
            env = attack_coeff * env + (1.0 - attack_coeff) * detector[i]
        else:
            env = release_coeff * env + (1.0 - release_coeff) * detector[i]
        envelope[i] = env
    if mode == "rms":
        envelope = np.sqrt(np.maximum(envelope, 0.0))
    return envelope


def compute_gain_reduction(
    input_db: np.ndarray,
    params: CompressorParams,
) -> GainReduction:
    """Apply static characteristic to get gain reduction curve."""
    output_db = np.array([static_characteristic(x, params) for x in input_db])
    gr = output_db - input_db + params.makeup_gain_db
    return GainReduction(
        input_db=input_db,
        output_db=output_db + params.makeup_gain_db,
        gain_reduction_db=gr,
        peak_reduction_db=float(np.min(gr)),
        mean_reduction_db=float(np.mean(gr)),
    )


def compress_signal(
    signal: np.ndarray,
    sr: float,
    params: CompressorParams,
    frame_size: int = 1024,
) -> np.ndarray:
    """Apply compression to an audio signal using frame-by-frame processing."""
    output = np.copy(signal)
    hop = frame_size
    n_frames = max(1, len(signal) // hop)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(signal))
        frame = signal[start:end]
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        input_db = 20.0 * math.log10(max(rms, 1e-12))
        output_db = static_characteristic(input_db, params)
        gain_db = output_db - input_db + params.makeup_gain_db
        gain_linear = 10.0 ** (gain_db / 20.0)
        output[start:end] = frame * gain_linear
    return output


def multiband_split(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    crossover_freqs: list[float],
) -> list[np.ndarray]:
    """Split a spectrum into frequency bands at crossover frequencies."""
    edges = [0.0] + sorted(crossover_freqs) + [float(freqs[-1]) + 1.0]
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        band = np.zeros_like(spectrum)
        band[mask] = spectrum[mask]
        bands.append(band)
    return bands


def multiband_compress(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    config: MultibandConfig,
) -> np.ndarray:
    """Apply per-band compression to a magnitude spectrum."""
    bands = multiband_split(spectrum, freqs, config.crossover_freqs)
    n_params = len(config.per_band_params)
    output = np.zeros_like(spectrum)
    for idx, band in enumerate(bands):
        params = config.per_band_params[min(idx, n_params - 1)]
        energy = float(np.sum(band ** 2))
        if energy < 1e-12:
            output += band
            continue
        input_db = 10.0 * math.log10(energy + 1e-12)
        out_db = static_characteristic(input_db, params) + params.makeup_gain_db
        gain = 10.0 ** ((out_db - input_db) / 20.0)
        output += band * gain
    return output


def auto_makeup_gain(params: CompressorParams, avg_input_db: float = -18.0) -> float:
    """Estimate makeup gain to compensate for average gain reduction."""
    out_db = static_characteristic(avg_input_db, params)
    return avg_input_db - out_db
