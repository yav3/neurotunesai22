"""Hearing profile modeling and compensation for accessible music."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


STANDARD_AUDIOGRAM_FREQUENCIES: list[int] = [250, 500, 1000, 2000, 4000, 8000]


class HearingLossType(Enum):
    CONDUCTIVE = "conductive"
    SENSORINEURAL = "sensorineural"
    MIXED = "mixed"
    NORMAL = "normal"


@dataclass
class HearingProfile:
    loss_type: HearingLossType
    audiogram_db: dict[int, float] = field(default_factory=dict)
    tinnitus_freq: float | None = None
    recruitment: bool = False

    def average_loss(self) -> float:
        if not self.audiogram_db:
            return 0.0
        return sum(self.audiogram_db.values()) / len(self.audiogram_db)

    def severity(self) -> str:
        avg = self.average_loss()
        if avg <= 20:
            return "normal"
        if avg <= 40:
            return "mild"
        if avg <= 55:
            return "moderate"
        if avg <= 70:
            return "moderately severe"
        if avg <= 90:
            return "severe"
        return "profound"


def frequency_response_compensation(profile: HearingProfile, n_bands: int = 10) -> np.ndarray:
    """Per-band gain curve (dB) to partially compensate audiogram loss."""
    freqs = np.logspace(np.log10(125), np.log10(16000), n_bands)
    gains = np.zeros(n_bands)
    if not profile.audiogram_db:
        return gains
    aud_freqs = sorted(profile.audiogram_db.keys())
    aud_vals = [profile.audiogram_db[f] for f in aud_freqs]
    for i, f in enumerate(freqs):
        loss = float(np.interp(f, aud_freqs, aud_vals))
        if profile.recruitment:
            gains[i] = loss * 0.5
        else:
            gains[i] = min(loss * 0.7, 40.0)
    return gains


def tinnitus_notch_filter(
    center_freq: float,
    bandwidth_oct: float = 0.5,
    depth_db: float = -20.0,
    sr: int = 44100,
    n_taps: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Design a notch filter to attenuate tinnitus frequency."""
    low = center_freq * 2 ** (-bandwidth_oct / 2)
    high = center_freq * 2 ** (bandwidth_oct / 2)
    freqs = np.fft.rfftfreq(n_taps, 1.0 / sr)
    H = np.ones(len(freqs))
    mask = (freqs >= low) & (freqs <= high)
    H[mask] = 10 ** (depth_db / 20.0)
    h = np.fft.irfft(H, n_taps)
    h *= np.hanning(n_taps)
    return h, freqs


def speech_intelligibility_index(
    profile: HearingProfile,
    signal_spectrum: np.ndarray,
    noise_spectrum: np.ndarray,
) -> float:
    """Simplified SII (ANSI S3.5): weighted audibility across bands."""
    n_bands = len(signal_spectrum)
    if n_bands == 0 or len(noise_spectrum) != n_bands:
        return 0.0
    band_weights = np.ones(n_bands) / n_bands
    sii = 0.0
    for i in range(n_bands):
        snr = signal_spectrum[i] - noise_spectrum[i]
        audibility = np.clip((snr + 15) / 30.0, 0.0, 1.0)
        sii += band_weights[i] * audibility
    return float(np.clip(sii, 0.0, 1.0))


def comfortable_listening_level(profile: HearingProfile) -> float:
    """Recommended listening level in dB SPL."""
    avg_loss = profile.average_loss()
    base = 65.0
    return base + avg_loss * 0.5
