"""Audiological models: equal-loudness, presbycusis, hearing aid prescription, CI vocoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "Audiogram", "HearingAidPrescription", "CIVocoderConfig",
    "iso226_equal_loudness", "presbycusis_threshold_shift",
    "simulate_audiogram", "hearing_aid_gain",
    "cochlear_implant_vocoder", "noise_induced_threshold_shift",
    "speech_intelligibility_index", "audiogram_severity",
    "frequency_compression", "dynamic_range_compression",
]


ISO226_FREQUENCIES = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500,
])

ISO226_THRESHOLDS_0_PHON = np.array([
    78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
    14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7,
    -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
])


def iso226_equal_loudness(phon_level: float) -> Tuple[np.ndarray, np.ndarray]:
    freqs = ISO226_FREQUENCIES.copy()
    base = ISO226_THRESHOLDS_0_PHON.copy()
    if phon_level <= 0:
        return freqs, base
    scaling = np.ones_like(base)
    low_idx = freqs < 200
    mid_idx = (freqs >= 200) & (freqs <= 4000)
    high_idx = freqs > 4000
    scaling[low_idx] = 0.8
    scaling[mid_idx] = 1.0
    scaling[high_idx] = 0.9
    spl = base + phon_level * scaling
    return freqs, spl


@dataclass
class Audiogram:
    frequencies: np.ndarray
    thresholds_left: np.ndarray
    thresholds_right: np.ndarray

    @property
    def pta_left(self) -> float:
        target = [500, 1000, 2000, 4000]
        vals = []
        for f in target:
            idx = np.argmin(np.abs(self.frequencies - f))
            vals.append(self.thresholds_left[idx])
        return float(np.mean(vals))

    @property
    def pta_right(self) -> float:
        target = [500, 1000, 2000, 4000]
        vals = []
        for f in target:
            idx = np.argmin(np.abs(self.frequencies - f))
            vals.append(self.thresholds_right[idx])
        return float(np.mean(vals))


def audiogram_severity(pta: float) -> str:
    if pta <= 25:
        return "normal"
    elif pta <= 40:
        return "mild"
    elif pta <= 55:
        return "moderate"
    elif pta <= 70:
        return "moderately_severe"
    elif pta <= 90:
        return "severe"
    return "profound"


def presbycusis_threshold_shift(
    frequencies: np.ndarray,
    age: float,
    sex: str = "male",
) -> np.ndarray:
    f = np.asarray(frequencies, dtype=float)
    if sex == "male":
        a, b = 0.00015, 2.1
    else:
        a, b = 0.00010, 1.8
    shift = a * f ** 1.2 * (age / 20.0) ** b
    return np.clip(shift, 0, 120)


def noise_induced_threshold_shift(
    frequencies: np.ndarray,
    exposure_level_db: float,
    exposure_years: float,
) -> np.ndarray:
    f = np.asarray(frequencies, dtype=float)
    excess = max(exposure_level_db - 85, 0)
    notch_center = 4000.0
    notch_width = 2000.0
    notch = np.exp(-0.5 * ((f - notch_center) / notch_width) ** 2)
    shift = excess * 0.5 * np.sqrt(exposure_years) * notch
    return np.clip(shift, 0, 80)


def simulate_audiogram(
    age: float = 60,
    sex: str = "male",
    noise_exposure_db: float = 0,
    noise_years: float = 0,
    frequencies: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Audiogram:
    if frequencies is None:
        frequencies = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000], dtype=float)
    rng = np.random.default_rng(seed)
    pres = presbycusis_threshold_shift(frequencies, age, sex)
    nihl = noise_induced_threshold_shift(frequencies, noise_exposure_db, noise_years)
    base = pres + nihl
    left = base + rng.normal(0, 3, len(frequencies))
    right = base + rng.normal(0, 3, len(frequencies))
    return Audiogram(
        frequencies=frequencies,
        thresholds_left=np.clip(left, -10, 120),
        thresholds_right=np.clip(right, -10, 120),
    )


@dataclass
class HearingAidPrescription:
    frequencies: np.ndarray
    insertion_gain: np.ndarray
    output_limit: np.ndarray
    compression_ratio: np.ndarray


def hearing_aid_gain(
    audiogram_thresholds: np.ndarray,
    frequencies: np.ndarray,
    target_input_level: float = 65.0,
) -> HearingAidPrescription:
    hl = np.asarray(audiogram_thresholds, dtype=float)
    gain = np.zeros_like(hl)
    for i, h in enumerate(hl):
        if h <= 20:
            gain[i] = 0.0
        elif h <= 60:
            gain[i] = h * 0.46 + 1
        else:
            gain[i] = h * 0.65 - 1
    f = np.asarray(frequencies, dtype=float)
    high_freq_boost = np.clip((f - 1000) / 4000, 0, 1) * 5
    gain += high_freq_boost
    output_limit = 105 + 0.2 * hl
    cr = np.ones_like(hl)
    for i, h in enumerate(hl):
        if h <= 40:
            cr[i] = 1.5
        elif h <= 60:
            cr[i] = 2.0
        elif h <= 80:
            cr[i] = 2.5
        else:
            cr[i] = 3.0
    return HearingAidPrescription(
        frequencies=frequencies,
        insertion_gain=np.clip(gain, 0, 80),
        output_limit=np.clip(output_limit, 90, 140),
        compression_ratio=cr,
    )


@dataclass
class CIVocoderConfig:
    n_channels: int = 22
    low_freq: float = 200.0
    high_freq: float = 8000.0
    envelope_cutoff: float = 400.0


def cochlear_implant_vocoder(
    audio: np.ndarray,
    sr: int = 22050,
    config: CIVocoderConfig = CIVocoderConfig(),
) -> np.ndarray:
    n = len(audio)
    log_freqs = np.logspace(
        np.log10(config.low_freq), np.log10(config.high_freq),
        config.n_channels + 1,
    )
    output = np.zeros(n)
    fft_size = min(1024, n)
    hop = fft_size // 2
    for start in range(0, n - fft_size, hop):
        frame = audio[start:start + fft_size]
        spectrum = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
        for ch in range(config.n_channels):
            band_mask = (freqs >= log_freqs[ch]) & (freqs < log_freqs[ch + 1])
            if not np.any(band_mask):
                continue
            band_energy = np.sqrt(np.mean(np.abs(spectrum[band_mask]) ** 2))
            center_freq = np.sqrt(log_freqs[ch] * log_freqs[ch + 1])
            t = np.arange(fft_size) / sr
            carrier = np.sin(2 * np.pi * center_freq * t) * band_energy
            window = np.hanning(fft_size)
            output[start:start + fft_size] += carrier * window
    mx = np.abs(output).max()
    if mx > 1e-9:
        output /= mx
    return output


def speech_intelligibility_index(
    audiogram_thresholds: np.ndarray,
    frequencies: np.ndarray,
    speech_level_db: float = 65.0,
    noise_level_db: float = 30.0,
) -> float:
    sii_bands = np.array([250, 500, 1000, 2000, 4000])
    importance = np.array([0.0617, 0.0671, 0.0781, 0.1178, 0.0923])
    total = 0.0
    for i, bf in enumerate(sii_bands):
        idx = np.argmin(np.abs(frequencies - bf))
        threshold = audiogram_thresholds[idx]
        audible_speech = max(0, speech_level_db - max(threshold, noise_level_db))
        band_audibility = min(audible_speech / 30.0, 1.0)
        total += importance[i] * band_audibility
    return float(np.clip(total / importance.sum(), 0, 1))


def frequency_compression(
    audio: np.ndarray,
    sr: int,
    kneepoint_hz: float = 2000.0,
    ratio: float = 2.0,
) -> np.ndarray:
    n = len(audio)
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    new_freqs = np.where(
        freqs <= kneepoint_hz,
        freqs,
        kneepoint_hz + (freqs - kneepoint_hz) / ratio,
    )
    phase = np.angle(spectrum)
    magnitude = np.abs(spectrum)
    new_spectrum = np.zeros_like(spectrum)
    for i in range(len(freqs)):
        target_bin = int(round(new_freqs[i] * n / sr))
        if 0 <= target_bin < len(new_spectrum):
            new_spectrum[target_bin] += magnitude[i] * np.exp(1j * phase[i])
    return np.fft.irfft(new_spectrum, n=n)


def dynamic_range_compression(
    audio: np.ndarray,
    threshold_db: float = -20.0,
    ratio: float = 3.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    sr: int = 22050,
) -> np.ndarray:
    out = audio.copy()
    attack_coeff = 1 - np.exp(-1.0 / (attack_ms * sr / 1000))
    release_coeff = 1 - np.exp(-1.0 / (release_ms * sr / 1000))
    envelope = 0.0
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    for i in range(len(out)):
        level = abs(out[i])
        if level > envelope:
            envelope += attack_coeff * (level - envelope)
        else:
            envelope += release_coeff * (level - envelope)
        if envelope > threshold_lin:
            gain_db = threshold_db + (20 * np.log10(max(envelope, 1e-15)) - threshold_db) / ratio
            gain = 10.0 ** (gain_db / 20.0) / max(envelope, 1e-15)
            out[i] *= gain
    return out
