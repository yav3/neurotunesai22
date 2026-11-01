"""Frequency-domain compensation for hearing loss: audiogram-guided remapping,
spectral compression, frequency lowering, and perceptual loudness correction."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


STANDARD_FREQS = np.array([250, 500, 1000, 2000, 4000, 8000], dtype=float)


@dataclass
class AudiogramPoint:
    frequency_hz: float
    threshold_db_hl: float


@dataclass
class RemappingConfig:
    source_range: Tuple[float, float] = (250.0, 8000.0)
    target_range: Tuple[float, float] = (250.0, 4000.0)
    compression_ratio: float = 2.0
    crossover_hz: float = 2000.0
    gain_limit_db: float = 30.0


def interpolate_audiogram(
    points: List[AudiogramPoint], freqs: np.ndarray
) -> np.ndarray:
    """Interpolate audiogram thresholds at arbitrary frequencies (log-linear)."""
    if len(points) < 2:
        return np.full(len(freqs), points[0].threshold_db_hl if points else 0.0)
    pt_f = np.array([p.frequency_hz for p in points])
    pt_t = np.array([p.threshold_db_hl for p in points])
    order = np.argsort(pt_f)
    pt_f, pt_t = pt_f[order], pt_t[order]
    return np.interp(np.log2(freqs), np.log2(pt_f), pt_t)


def hearing_loss_gain(
    audiogram: List[AudiogramPoint], freqs: np.ndarray, *, prescriptive: str = "nal_nl2"
) -> np.ndarray:
    """Compute prescriptive gain (dB) per frequency bin.

    prescriptive: 'nal_nl2' (half-gain rule variant) or 'dsl' (desired sensation level).
    """
    thresholds = interpolate_audiogram(audiogram, freqs)
    if prescriptive == "dsl":
        gain = thresholds * 0.6
    else:
        gain = thresholds * 0.46 + 0.2 * np.clip(thresholds - 40, 0, None)
    return np.clip(gain, 0.0, 30.0)


def apply_frequency_gain(
    magnitude_spectrum: np.ndarray,
    gain_db: np.ndarray,
) -> np.ndarray:
    """Apply per-bin gain (dB) to a magnitude spectrum."""
    linear_gain = 10 ** (gain_db / 20.0)
    return magnitude_spectrum * linear_gain


def frequency_lowering(
    spectrum: np.ndarray,
    sr: float,
    n_fft: int,
    *,
    source_cutoff_hz: float = 4000.0,
    target_band: Tuple[float, float] = (1500.0, 3000.0),
) -> np.ndarray:
    """Non-linear frequency compression: lower high-frequency energy into an audible band."""
    freqs = np.linspace(0, sr / 2, len(spectrum))
    out = spectrum.copy()
    src_mask = freqs >= source_cutoff_hz
    tgt_lo, tgt_hi = target_band
    n_src = int(src_mask.sum())
    if n_src == 0:
        return out
    tgt_idx_lo = int(tgt_lo / (sr / 2) * len(spectrum))
    tgt_idx_hi = int(tgt_hi / (sr / 2) * len(spectrum))
    n_tgt = max(tgt_idx_hi - tgt_idx_lo, 1)
    src_energy = spectrum[src_mask]
    compressed = np.interp(
        np.linspace(0, 1, n_tgt), np.linspace(0, 1, n_src), src_energy
    )
    out[tgt_idx_lo:tgt_idx_hi] = np.maximum(
        out[tgt_idx_lo:tgt_idx_hi], compressed * 0.5
    )
    out[src_mask] *= 0.1
    return out


def spectral_envelope_compression(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    crossover_hz: float = 2000.0,
    ratio: float = 2.0,
) -> np.ndarray:
    """Compress the spectral envelope above a crossover frequency.

    ratio > 1 reduces the dynamic range of high-frequency energy relative to low.
    """
    out = spectrum.copy()
    above = freqs >= crossover_hz
    if not np.any(above):
        return out
    ref = np.mean(spectrum[above]) + 1e-12
    db = 20 * np.log10(spectrum[above] / ref + 1e-12)
    compressed_db = db / ratio
    out[above] = ref * 10 ** (compressed_db / 20.0)
    return out


def equal_loudness_weight(freqs: np.ndarray, phon: float = 60.0) -> np.ndarray:
    """Approximate ISO 226 equal-loudness contour weights at a given phon level.

    Returns linear gain multipliers that flatten perceived loudness across frequency.
    """
    f_norm = freqs / 1000.0
    base_curve = (
        -6.0 * np.exp(-0.6 * (np.log2(f_norm + 0.01)) ** 2)
        + 3.0 * np.exp(-((np.log2(f_norm) - 2.5) ** 2) / 1.5)
    )
    phon_offset = (phon - 60.0) * 0.15
    correction_db = -(base_curve + phon_offset)
    return 10 ** (correction_db / 20.0)


@dataclass
class RemappingResult:
    original_spectrum: np.ndarray
    compensated_spectrum: np.ndarray
    gain_applied_db: np.ndarray
    frequency_lowered: bool
    loudness_weighted: bool


def full_remapping_pipeline(
    spectrum: np.ndarray,
    sr: float,
    audiogram: List[AudiogramPoint],
    config: RemappingConfig | None = None,
    *,
    apply_loudness: bool = True,
    apply_lowering: bool = False,
    phon: float = 60.0,
) -> RemappingResult:
    """Full pipeline: prescriptive gain → optional frequency lowering → loudness weighting."""
    cfg = config or RemappingConfig()
    n = len(spectrum)
    freqs = np.linspace(0, sr / 2, n)

    gain = hearing_loss_gain(audiogram, freqs)
    gain = np.clip(gain, 0, cfg.gain_limit_db)
    result = apply_frequency_gain(spectrum, gain)

    lowered = False
    if apply_lowering:
        result = frequency_lowering(result, sr, n * 2, source_cutoff_hz=cfg.crossover_hz)
        lowered = True

    result = spectral_envelope_compression(
        result, freqs, crossover_hz=cfg.crossover_hz, ratio=cfg.compression_ratio
    )

    weighted = False
    if apply_loudness:
        w = equal_loudness_weight(freqs, phon=phon)
        result = result * w
        weighted = True

    return RemappingResult(
        original_spectrum=spectrum,
        compensated_spectrum=result,
        gain_applied_db=gain,
        frequency_lowered=lowered,
        loudness_weighted=weighted,
    )


def compensation_error(
    target_threshold: np.ndarray, achieved_threshold: np.ndarray
) -> float:
    """RMS error (dB) between target and achieved hearing thresholds."""
    return float(np.sqrt(np.mean((target_threshold - achieved_threshold) ** 2)))


def bandwidth_utilization(spectrum: np.ndarray, sr: float, *, cutoff_db: float = -40.0) -> float:
    """Fraction of the spectrum above cutoff_db relative to peak."""
    db = 20 * np.log10(spectrum / (np.max(spectrum) + 1e-12) + 1e-12)
    return float(np.mean(db > cutoff_db))
