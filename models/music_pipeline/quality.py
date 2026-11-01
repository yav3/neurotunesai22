"""Signal quality assessment: clipping, silence, SNR, spectral quality metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QualityReport:
    """Quality assessment of an audio signal."""
    duration_s: float
    peak_amplitude: float
    rms_level: float
    rms_db: float
    clipping_ratio: float
    silence_ratio: float
    snr_estimate_db: float
    dc_offset: float
    crest_factor: float
    dynamic_range_db: float
    spectral_bandwidth: float
    is_valid: bool
    issues: list[str]


def peak_amplitude(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return 0.0
    return float(np.max(np.abs(signal)))


def rms_level(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def rms_to_db(rms: float, ref: float = 1.0) -> float:
    if rms < 1e-10:
        return -100.0
    return float(20.0 * np.log10(rms / ref))


def clipping_ratio(signal: np.ndarray, threshold: float = 0.99) -> float:
    if len(signal) == 0:
        return 0.0
    clipped = np.sum(np.abs(signal) >= threshold)
    return float(clipped / len(signal))


def silence_ratio(
    signal: np.ndarray,
    sr: int = 22050,
    frame_length: int = 1024,
    threshold_db: float = -60.0,
) -> float:
    if len(signal) < frame_length:
        r = rms_level(signal)
        return 1.0 if rms_to_db(r) < threshold_db else 0.0

    n_frames = len(signal) // frame_length
    silent_frames = 0
    for i in range(n_frames):
        frame = signal[i * frame_length:(i + 1) * frame_length]
        r = rms_level(frame)
        if rms_to_db(r) < threshold_db:
            silent_frames += 1
    return float(silent_frames / n_frames) if n_frames > 0 else 0.0


def estimate_snr(
    signal: np.ndarray,
    sr: int = 22050,
    frame_length: int = 1024,
    noise_percentile: float = 10.0,
) -> float:
    """Estimate SNR by comparing signal RMS to noise floor (lowest-energy frames)."""
    if len(signal) < frame_length:
        return 0.0

    n_frames = len(signal) // frame_length
    frame_rms = np.array([
        rms_level(signal[i * frame_length:(i + 1) * frame_length])
        for i in range(n_frames)
    ])

    if len(frame_rms) == 0:
        return 0.0

    noise_floor = np.percentile(frame_rms, noise_percentile)
    signal_level = np.percentile(frame_rms, 90)

    if noise_floor < 1e-10:
        return 60.0
    return float(20.0 * np.log10(signal_level / noise_floor))


def dc_offset(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return 0.0
    return float(np.mean(signal))


def crest_factor(signal: np.ndarray) -> float:
    rms = rms_level(signal)
    peak = peak_amplitude(signal)
    if rms < 1e-10:
        return 0.0
    return float(peak / rms)


def dynamic_range(
    signal: np.ndarray,
    sr: int = 22050,
    frame_length: int = 1024,
) -> float:
    if len(signal) < frame_length:
        return 0.0

    n_frames = len(signal) // frame_length
    frame_rms = np.array([
        rms_level(signal[i * frame_length:(i + 1) * frame_length])
        for i in range(n_frames)
    ])

    frame_rms = frame_rms[frame_rms > 1e-10]
    if len(frame_rms) < 2:
        return 0.0

    loud = np.percentile(frame_rms, 95)
    quiet = np.percentile(frame_rms, 5)
    if quiet < 1e-10:
        return 60.0
    return float(20.0 * np.log10(loud / quiet))


def spectral_quality(
    signal: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
) -> float:
    """Spectral bandwidth as fraction of Nyquist — proxy for recording quality."""
    if len(signal) < n_fft:
        return 0.0
    mag = np.abs(np.fft.rfft(signal[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total = np.sum(mag)
    if total < 1e-10:
        return 0.0
    centroid = np.sum(freqs * mag) / total
    nyquist = sr / 2.0
    return float(centroid / nyquist)


def assess_quality(
    signal: np.ndarray,
    sr: int = 22050,
    frame_length: int = 1024,
) -> QualityReport:
    issues: list[str] = []
    dur = len(signal) / sr if sr > 0 else 0.0
    peak = peak_amplitude(signal)
    rms = rms_level(signal)
    rms_db = rms_to_db(rms)
    clip = clipping_ratio(signal)
    silence = silence_ratio(signal, sr, frame_length)
    snr = estimate_snr(signal, sr, frame_length)
    dc = dc_offset(signal)
    crest = crest_factor(signal)
    dyn_range = dynamic_range(signal, sr, frame_length)
    spec_bw = spectral_quality(signal, sr)

    if clip > 0.01:
        issues.append("clipping")
    if silence > 0.8:
        issues.append("mostly_silent")
    if snr < 10.0:
        issues.append("low_snr")
    if abs(dc) > 0.05:
        issues.append("dc_offset")
    if dur < 0.5:
        issues.append("very_short")
    if rms_db < -40:
        issues.append("very_quiet")

    return QualityReport(
        duration_s=dur,
        peak_amplitude=peak,
        rms_level=rms,
        rms_db=rms_db,
        clipping_ratio=clip,
        silence_ratio=silence,
        snr_estimate_db=snr,
        dc_offset=dc,
        crest_factor=crest,
        dynamic_range_db=dyn_range,
        spectral_bandwidth=spec_bw,
        is_valid=len(issues) == 0,
        issues=issues,
    )
