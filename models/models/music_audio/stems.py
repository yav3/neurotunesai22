"""Stem separation: spectral estimation, remixing, isolation."""
from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class StemType(Enum):
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    PIANO = "piano"
    GUITAR = "guitar"


@runtime_checkable
class StemSeparator(Protocol):
    def separate(self, audio: np.ndarray, sr: int = 22050) -> dict[StemType, np.ndarray]: ...


class SpectralStemEstimator:
    """Frequency-band heuristic stem estimator (no ML dependencies)."""

    BAND_LIMITS = {
        StemType.BASS: (20, 250),
        StemType.VOCALS: (250, 4000),
        StemType.OTHER: (4000, 20000),
    }

    def separate(self, audio: np.ndarray, sr: int = 22050) -> dict[StemType, np.ndarray]:
        sig = np.asarray(audio, dtype=np.float64)
        n = len(sig)
        spectrum = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        stems: dict[StemType, np.ndarray] = {}
        for stem_type, (lo, hi) in self.BAND_LIMITS.items():
            mask = (freqs >= lo) & (freqs < hi)
            band = np.zeros_like(spectrum)
            band[mask] = spectrum[mask]
            stems[stem_type] = np.fft.irfft(band, n=n).astype(np.float32)
        return stems

    def separate_detailed(
        self, audio: np.ndarray, sr: int = 22050,
        bands: dict[StemType, tuple[float, float]] | None = None,
    ) -> dict[StemType, np.ndarray]:
        bands = bands or self.BAND_LIMITS
        sig = np.asarray(audio, dtype=np.float64)
        n = len(sig)
        spectrum = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        stems: dict[StemType, np.ndarray] = {}
        for stem_type, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            band = np.zeros_like(spectrum)
            band[mask] = spectrum[mask]
            stems[stem_type] = np.fft.irfft(band, n=n).astype(np.float32)
        return stems


class StemRemixer:
    @staticmethod
    def remix(stems: dict[StemType, np.ndarray], weights: dict[StemType, float] | None = None) -> np.ndarray:
        if not stems:
            return np.array([], dtype=np.float32)
        weights = weights or {s: 1.0 for s in stems}
        max_len = max(len(v) for v in stems.values())
        out = np.zeros(max_len, dtype=np.float32)
        for stem_type, signal in stems.items():
            w = weights.get(stem_type, 1.0)
            out[:len(signal)] += signal * w
        return out

    @staticmethod
    def isolate(stems: dict[StemType, np.ndarray], keep: list[StemType]) -> np.ndarray:
        keep_set = set(keep)
        selected = {k: v for k, v in stems.items() if k in keep_set}
        return StemRemixer.remix(selected)

    @staticmethod
    def remove(stems: dict[StemType, np.ndarray], drop: list[StemType]) -> np.ndarray:
        drop_set = set(drop)
        remaining = {k: v for k, v in stems.items() if k not in drop_set}
        return StemRemixer.remix(remaining)
