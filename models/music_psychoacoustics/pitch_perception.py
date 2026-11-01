"""Pitch perception: virtual pitch, pitch salience, missing fundamental."""
from __future__ import annotations
import numpy as np


class VirtualPitch:
    def missing_fundamental(self, harmonics: list[float]) -> float:
        if not harmonics:
            return 0.0
        if len(harmonics) == 1:
            return harmonics[0]
        diffs = []
        for i in range(len(harmonics) - 1):
            diffs.append(harmonics[i + 1] - harmonics[i])
        if not diffs:
            return harmonics[0]
        gcd_estimate = np.median(diffs)
        if gcd_estimate < 20.0:
            return harmonics[0]
        return float(gcd_estimate)

    def harmonic_template_match(
        self, spectrum_freqs: np.ndarray, spectrum_mags: np.ndarray,
        f0_candidate: float, n_harmonics: int = 8,
    ) -> float:
        if f0_candidate <= 0:
            return 0.0
        score = 0.0
        for h in range(1, n_harmonics + 1):
            target = f0_candidate * h
            diffs = np.abs(spectrum_freqs - target)
            closest = np.argmin(diffs)
            if diffs[closest] < f0_candidate * 0.03:
                score += spectrum_mags[closest] / h
        return float(score)

    def estimate_f0(
        self, spectrum_freqs: np.ndarray, spectrum_mags: np.ndarray,
        f0_min: float = 50.0, f0_max: float = 2000.0, resolution: int = 200,
    ) -> tuple[float, float]:
        candidates = np.linspace(f0_min, f0_max, resolution)
        scores = np.array([
            self.harmonic_template_match(spectrum_freqs, spectrum_mags, c)
            for c in candidates
        ])
        best = np.argmax(scores)
        return float(candidates[best]), float(scores[best])


class PitchSalience:
    def salience(self, autocorrelation: np.ndarray, sr: int) -> list[tuple[float, float]]:
        if len(autocorrelation) < 3:
            return []
        peaks = []
        for i in range(1, len(autocorrelation) - 1):
            if (autocorrelation[i] > autocorrelation[i - 1] and
                    autocorrelation[i] > autocorrelation[i + 1]):
                freq = sr / i if i > 0 else 0.0
                if 50.0 <= freq <= 4000.0:
                    peaks.append((freq, float(autocorrelation[i])))
        return sorted(peaks, key=lambda x: -x[1])

    def pitch_strength(self, autocorrelation: np.ndarray) -> float:
        if len(autocorrelation) < 2:
            return 0.0
        peak = float(np.max(autocorrelation[1:]))
        return max(min(peak / (abs(autocorrelation[0]) + 1e-12), 1.0), 0.0)


def estimate_pitch(
    spectrum_freqs: np.ndarray, spectrum_mags: np.ndarray,
) -> tuple[float, float]:
    return VirtualPitch().estimate_f0(spectrum_freqs, spectrum_mags)
