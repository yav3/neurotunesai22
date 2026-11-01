"""Spectral analysis solvers — EEG and audio frequency-domain math.

EEG band power:
    δ: 1–4 Hz,  θ: 4–8 Hz,  α: 8–13 Hz,  β: 13–30 Hz,  γ: 30–100 Hz

Key derived measures:
    θ/α ratio = P_θ / P_α          (cognitive load / AD biomarker)
    Spectral entropy = -Σ p_i log(p_i)  (complexity)
    α peak frequency                     (processing speed)

Binaural beat math:
    f_beat = |f_left - f_right|
    Entrainment target: f_beat → desired EEG band

Cross-spectral coherence (inter-channel):
    C_xy(f) = |S_xy(f)|² / (S_xx(f) · S_yy(f))
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

THERAPEUTIC_GOALS = {
    "focus": "beta",
    "relaxation": "alpha",
    "meditation": "theta",
    "sleep": "delta",
    "creativity": "theta",
    "energy": "beta",
}


class SpectralSolver:

    @staticmethod
    def band_power(
        psd: np.ndarray,
        freqs: np.ndarray,
        band: tuple[float, float],
    ) -> float:
        """Integrate PSD over a frequency band.
        P = ∫_{f1}^{f2} S(f) df  ≈ Σ S(f_i) · Δf
        """
        mask = (freqs >= band[0]) & (freqs < band[1])
        if not mask.any():
            return 0.0
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        return float(np.sum(psd[mask]) * df)

    @staticmethod
    def all_band_powers(
        psd: np.ndarray, freqs: np.ndarray,
    ) -> dict[str, float]:
        """Compute power in all standard EEG bands."""
        return {
            name: SpectralSolver.band_power(psd, freqs, band)
            for name, band in EEG_BANDS.items()
        }

    @staticmethod
    def theta_alpha_ratio(powers: dict[str, float]) -> float:
        """θ/α ratio — elevated in AD, cognitive load."""
        alpha = powers.get("alpha", 1e-8)
        return powers.get("theta", 0.0) / max(alpha, 1e-8)

    @staticmethod
    def spectral_entropy(psd: np.ndarray) -> float:
        """H = -Σ p_i · log(p_i), where p_i = S(f_i) / Σ S(f)."""
        total = np.sum(psd)
        if total <= 0:
            return 0.0
        p = psd / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def alpha_peak_frequency(psd: np.ndarray, freqs: np.ndarray) -> float:
        """Frequency of maximum power in the α band."""
        mask = (freqs >= 8.0) & (freqs < 13.0)
        if not mask.any():
            return 10.0
        alpha_psd = psd[mask]
        alpha_freqs = freqs[mask]
        return float(alpha_freqs[np.argmax(alpha_psd)])

    @staticmethod
    def binaural_beat_frequency(
        target_band: str,
    ) -> tuple[float, float, float]:
        """Compute left/right ear frequencies for binaural entrainment.

        Returns (f_left, f_right, f_beat) where f_beat is in the target band.
        Carrier frequency is 200 Hz (standard).
        """
        band = EEG_BANDS.get(target_band, (8.0, 13.0))
        f_beat = (band[0] + band[1]) / 2
        carrier = 200.0
        return (carrier, carrier + f_beat, f_beat)

    @staticmethod
    def coherence(
        sxx: np.ndarray,
        syy: np.ndarray,
        sxy: np.ndarray,
    ) -> np.ndarray:
        """Cross-spectral coherence: C(f) = |Sxy|² / (Sxx · Syy)."""
        denom = sxx * syy
        denom = np.maximum(denom, 1e-12)
        return np.abs(sxy) ** 2 / denom

    @staticmethod
    def therapeutic_score(
        powers: dict[str, float],
        goal: str,
    ) -> float:
        """Score how well current EEG matches a therapeutic goal.
        Higher target-band power relative to total = better alignment.
        """
        target_band = THERAPEUTIC_GOALS.get(goal, "alpha")
        target_power = powers.get(target_band, 0.0)
        total = sum(powers.values()) or 1e-8
        return target_power / total
