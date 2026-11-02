"""Perceptual audio coding models — masking thresholds, bit allocation, perceptual entropy."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class MaskingThreshold:
    freq_hz: np.ndarray
    threshold_db: np.ndarray
    quiet_threshold_db: np.ndarray


@dataclass
class BitAllocation:
    band_index: np.ndarray
    bits_per_sample: np.ndarray
    total_bits: int
    perceptual_entropy: float


@dataclass
class DistortionMeasure:
    noise_to_mask_ratio: float
    perceptual_snr_db: float
    bands_above_mask: int


QUIET_THRESHOLD_COEFFS = (3.64, -6.5, 6.0e-4)


def quiet_threshold_db(f_hz: np.ndarray) -> np.ndarray:
    f_khz = np.clip(f_hz / 1000.0, 0.02, 20.0)
    a, b, c = QUIET_THRESHOLD_COEFFS
    return (a * f_khz ** (-0.8)
            + b * np.exp(-0.6 * (f_khz - 3.3) ** 2)
            + c * f_khz ** 4)


def bark_freq(f_hz: float) -> float:
    f = f_hz / 1000.0
    return 13.0 * np.arctan(0.76 * f) + 3.5 * np.arctan((f / 7.5) ** 2)


def bark_band_edges(n_bands: int = 25, sr: float = 44100.0) -> np.ndarray:
    max_bark = bark_freq(sr / 2.0)
    bark_edges = np.linspace(0.0, max_bark, n_bands + 1)
    edges_hz = np.zeros(n_bands + 1)
    for i, b in enumerate(bark_edges):
        lo, hi = 20.0, sr / 2.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if bark_freq(mid) < b:
                lo = mid
            else:
                hi = mid
        edges_hz[i] = (lo + hi) / 2.0
    return edges_hz


def simultaneous_masking(
    masker_level_db: float, masker_bark: float, target_bark: float,
) -> float:
    dz = target_bark - masker_bark
    if dz >= 0:
        slope = 27.0 - 0.37 * max(masker_level_db - 40.0, 0.0)
        spread = masker_level_db - slope * dz
    else:
        spread = masker_level_db + 18.0 * dz
    return spread


def temporal_masking_pre(
    masker_level_db: float, dt_ms: float,
) -> float:
    if dt_ms < 0 or dt_ms > 20.0:
        return -100.0
    return masker_level_db - 20.0 * (dt_ms / 20.0)


def temporal_masking_post(
    masker_level_db: float, dt_ms: float,
) -> float:
    if dt_ms < 0 or dt_ms > 200.0:
        return -100.0
    decay_rate = 5.0
    return masker_level_db - decay_rate * np.log10(max(dt_ms, 0.5)) * 10.0


def compute_masking_threshold(
    spectrum_db: np.ndarray,
    freq_hz: np.ndarray,
    n_bands: int = 25,
) -> MaskingThreshold:
    edges = bark_band_edges(n_bands, sr=2.0 * freq_hz[-1])
    band_centers = (edges[:-1] + edges[1:]) / 2.0
    band_levels = np.zeros(n_bands)
    for i in range(n_bands):
        mask = (freq_hz >= edges[i]) & (freq_hz < edges[i + 1])
        if np.any(mask):
            band_levels[i] = 10.0 * np.log10(np.sum(10.0 ** (spectrum_db[mask] / 10.0)) + 1e-30)
        else:
            band_levels[i] = -100.0
    bark_centers = np.array([bark_freq(f) for f in band_centers])
    threshold = np.full(n_bands, -100.0)
    for i in range(n_bands):
        contributions = []
        for j in range(n_bands):
            if band_levels[j] > -80.0:
                m = simultaneous_masking(band_levels[j], bark_centers[j], bark_centers[i])
                contributions.append(m)
        if contributions:
            threshold[i] = 10.0 * np.log10(np.sum(10.0 ** (np.array(contributions) / 10.0)))
    qt = quiet_threshold_db(band_centers)
    combined = np.maximum(threshold, qt)
    return MaskingThreshold(
        freq_hz=band_centers, threshold_db=combined, quiet_threshold_db=qt,
    )


def signal_to_mask_ratio(
    signal_db: np.ndarray, mask_db: np.ndarray,
) -> np.ndarray:
    return signal_db - mask_db


def perceptual_entropy(
    spectrum_db: np.ndarray, mask_threshold_db: np.ndarray,
) -> float:
    smr = spectrum_db - mask_threshold_db
    bits = np.maximum(smr / 6.02, 0.0)
    return float(np.sum(bits))


def allocate_bits(
    smr_db: np.ndarray, total_budget: int,
) -> BitAllocation:
    raw_bits = np.maximum(smr_db / 6.02, 0.0)
    total_raw = np.sum(raw_bits)
    if total_raw < 1e-12:
        alloc = np.zeros(len(smr_db), dtype=int)
    else:
        scale = total_budget / total_raw
        alloc = np.clip(np.round(raw_bits * scale).astype(int), 0, 16)
        while np.sum(alloc) > total_budget and np.any(alloc > 0):
            idx = np.argmin(smr_db + (alloc == 0) * 1e6)
            alloc[idx] = max(alloc[idx] - 1, 0)
        while np.sum(alloc) < total_budget:
            idx = np.argmax(smr_db * (alloc < 16))
            alloc[idx] += 1
    pe = perceptual_entropy(smr_db + 60.0, np.full(len(smr_db), 60.0))
    return BitAllocation(
        band_index=np.arange(len(smr_db)),
        bits_per_sample=alloc,
        total_bits=int(np.sum(alloc)),
        perceptual_entropy=pe,
    )


def noise_shaping_curve(
    mask_threshold_db: np.ndarray,
    quantization_noise_db: np.ndarray,
) -> np.ndarray:
    return mask_threshold_db - quantization_noise_db


def perceptual_distortion(
    original_db: np.ndarray,
    coded_db: np.ndarray,
    mask_threshold_db: np.ndarray,
) -> DistortionMeasure:
    noise_db = original_db - coded_db
    nmr = noise_db - mask_threshold_db
    avg_nmr = float(np.mean(nmr))
    signal_power = np.sum(10.0 ** (original_db / 10.0))
    noise_power = np.sum(10.0 ** (np.maximum(noise_db, -100.0) / 10.0))
    psnr = 10.0 * np.log10(signal_power / max(noise_power, 1e-30))
    above = int(np.sum(nmr > 0))
    return DistortionMeasure(
        noise_to_mask_ratio=avg_nmr,
        perceptual_snr_db=float(psnr),
        bands_above_mask=above,
    )


def psychoacoustic_model_1(
    spectrum_db: np.ndarray, freq_hz: np.ndarray,
) -> np.ndarray:
    mt = compute_masking_threshold(spectrum_db, freq_hz)
    interp_threshold = np.interp(freq_hz, mt.freq_hz, mt.threshold_db)
    return interp_threshold


def coding_gain_db(
    original_db: np.ndarray, mask_threshold_db: np.ndarray,
) -> float:
    smr = original_db - mask_threshold_db
    uniform_bits = np.mean(np.maximum(original_db + 96.0, 0.0) / 6.02)
    shaped_bits = np.mean(np.maximum(smr, 0.0) / 6.02)
    if shaped_bits < 1e-12:
        return 0.0
    return float(10.0 * np.log10(uniform_bits / shaped_bits)) if shaped_bits > 0 else 0.0
