"""Spectral texture analysis: contrast, roughness, polyphony, tonal power ratio."""
from __future__ import annotations

import numpy as np


class TextureAnalyzer:
    """Spectral and temporal texture descriptors."""

    def spectral_contrast(
        self, spectrum: np.ndarray, freqs: np.ndarray, n_bands: int = 6
    ) -> np.ndarray:
        if len(spectrum) < 2:
            return np.zeros(n_bands)
        f_min, f_max = max(freqs[0], 20.0), freqs[-1]
        if f_min >= f_max:
            return np.zeros(n_bands)
        edges = f_min * np.power(f_max / f_min, np.arange(n_bands + 1) / n_bands)
        contrast = np.zeros(n_bands)
        for i in range(n_bands):
            mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
            band = spectrum[mask]
            if len(band) < 2:
                continue
            sorted_band = np.sort(band)
            n_neigh = max(1, len(band) // 4)
            peak = sorted_band[-n_neigh:].mean()
            valley = sorted_band[:n_neigh].mean()
            contrast[i] = np.log1p(peak) - np.log1p(valley)
        return contrast

    def spectral_valley_energy(
        self, spectrum: np.ndarray, freqs: np.ndarray, n_bands: int = 6
    ) -> np.ndarray:
        if len(spectrum) < 2:
            return np.zeros(n_bands)
        f_min, f_max = max(freqs[0], 20.0), freqs[-1]
        if f_min >= f_max:
            return np.zeros(n_bands)
        edges = f_min * np.power(f_max / f_min, np.arange(n_bands + 1) / n_bands)
        valleys = np.zeros(n_bands)
        for i in range(n_bands):
            mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
            band = spectrum[mask]
            if len(band) == 0:
                continue
            n_neigh = max(1, len(band) // 4)
            valleys[i] = np.sort(band)[:n_neigh].mean()
        return valleys

    def tonal_power_ratio(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        f0: float,
        n_harmonics: int = 10,
    ) -> float:
        if f0 <= 0 or len(spectrum) == 0:
            return 0.0
        total = np.sum(spectrum ** 2)
        if total < 1e-30:
            return 0.0
        tonal = 0.0
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        half_bin = max(1, int(0.5 * f0 * 0.05 / max(freq_res, 1e-12)))
        for h in range(1, n_harmonics + 1):
            hf = h * f0
            idx = np.argmin(np.abs(freqs - hf))
            lo = max(0, idx - half_bin)
            hi = min(len(spectrum), idx + half_bin + 1)
            tonal += np.sum(spectrum[lo:hi] ** 2)
        return float(min(tonal / total, 1.0))

    def roughness_sethares(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        amplitudes: np.ndarray | None = None,
    ) -> float:
        if len(spectrum) < 2:
            return 0.0
        threshold = np.max(spectrum) * 0.01
        peak_idx = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
                if spectrum[i] > threshold:
                    peak_idx.append(i)
        if len(peak_idx) < 2:
            return 0.0
        pf = freqs[peak_idx]
        pa = spectrum[peak_idx] if amplitudes is None else amplitudes[peak_idx]
        pa = pa / (np.max(pa) + 1e-30)
        d_star = 0.24
        a, b = 3.5, 5.75
        roughness = 0.0
        for i in range(len(pf)):
            for j in range(i + 1, len(pf)):
                f_lo, f_hi = min(pf[i], pf[j]), max(pf[i], pf[j])
                s = d_star / (0.0207 * f_lo + 18.96)
                x = s * (f_hi - f_lo)
                r = pa[i] * pa[j] * (np.exp(-a * x) - np.exp(-b * x))
                roughness += max(r, 0.0)
        return float(roughness)

    def spectral_flux_texture(
        self, frames: np.ndarray, order: int = 2
    ) -> np.ndarray:
        if frames.ndim != 2 or frames.shape[0] < 2:
            return np.array([0.0])
        diff = np.diff(frames, axis=0)
        flux = np.power(np.sum(np.abs(diff) ** order, axis=1), 1.0 / order)
        return flux

    def polyphony_estimate(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        threshold: float = 0.1,
    ) -> int:
        if len(spectrum) < 3:
            return 0
        norm = spectrum / (np.max(spectrum) + 1e-30)
        count = 0
        for i in range(1, len(norm) - 1):
            if norm[i] > norm[i - 1] and norm[i] > norm[i + 1] and norm[i] > threshold:
                count += 1
        return count

    def harmonic_energy_ratio(
        self, spectrum: np.ndarray, freqs: np.ndarray, f0: float
    ) -> float:
        return self.tonal_power_ratio(spectrum, freqs, f0, n_harmonics=16)

    def noise_to_harmonic_ratio(
        self,
        spectrum: np.ndarray,
        freqs: np.ndarray,
        f0: float,
        n_harmonics: int = 8,
    ) -> float:
        tpr = self.tonal_power_ratio(spectrum, freqs, f0, n_harmonics)
        if tpr >= 1.0:
            return 0.0
        return float((1.0 - tpr) / max(tpr, 1e-12))

    def temporal_flatness(self, rms_envelope: np.ndarray) -> float:
        if len(rms_envelope) < 2:
            return 0.0
        env = np.abs(rms_envelope) + 1e-30
        geo = np.exp(np.mean(np.log(env)))
        arith = np.mean(env)
        return float(geo / max(arith, 1e-30))

    def onset_density(
        self, onset_env: np.ndarray, sr: int, hop: int
    ) -> float:
        if len(onset_env) < 3 or sr <= 0 or hop <= 0:
            return 0.0
        threshold = np.mean(onset_env) + np.std(onset_env)
        count = 0
        for i in range(1, len(onset_env) - 1):
            if (
                onset_env[i] > onset_env[i - 1]
                and onset_env[i] > onset_env[i + 1]
                and onset_env[i] > threshold
            ):
                count += 1
        duration = len(onset_env) * hop / sr
        if duration < 1e-9:
            return 0.0
        return float(count / duration)
