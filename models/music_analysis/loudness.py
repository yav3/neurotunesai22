"""Perceptual loudness and dynamic range analysis."""
from __future__ import annotations
import numpy as np


class LoudnessAnalyzer:
    def rms(self, signal: np.ndarray) -> float:
        if len(signal) == 0:
            return 0.0
        return float(np.sqrt(np.mean(signal ** 2)))

    def peak(self, signal: np.ndarray) -> float:
        if len(signal) == 0:
            return 0.0
        return float(np.max(np.abs(signal)))

    def rms_db(self, signal: np.ndarray) -> float:
        r = self.rms(signal)
        return float(20 * np.log10(r)) if r > 1e-12 else -120.0

    def peak_db(self, signal: np.ndarray) -> float:
        p = self.peak(signal)
        return float(20 * np.log10(p)) if p > 1e-12 else -120.0

    def lufs_approximate(self, signal: np.ndarray, sr: int) -> float:
        if len(signal) < sr // 10:
            return -70.0
        filtered = self._k_weight(signal, sr)
        block_size = int(0.4 * sr)
        hop = int(0.1 * sr)
        if block_size < 1:
            return -70.0
        powers = []
        for start in range(0, len(filtered) - block_size + 1, hop):
            block = filtered[start:start + block_size]
            powers.append(float(np.mean(block ** 2)))
        if not powers:
            return -70.0
        abs_threshold = -70.0
        gated = [p for p in powers if p > 10 ** (abs_threshold / 10)]
        if not gated:
            return -70.0
        mean_power = np.mean(gated)
        rel_threshold = -0.691 + 10 * np.log10(mean_power) - 10
        final = [p for p in gated if 10 * np.log10(p + 1e-30) > rel_threshold]
        if not final:
            return -70.0
        return float(-0.691 + 10 * np.log10(np.mean(final)))

    def _k_weight(self, signal: np.ndarray, sr: int) -> np.ndarray:
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
        freqs = np.maximum(freqs, 1.0)
        shelf = 1.0 + (freqs / 1500) ** 2 / (1 + (freqs / 1500) ** 2)
        highpass = (freqs / 38) ** 2 / (1 + (freqs / 38) ** 2)
        weighted = spectrum * shelf * highpass
        return np.fft.irfft(weighted, n=len(signal))


class DynamicRangeAnalyzer:
    def crest_factor(self, signal: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(signal ** 2)))
        peak = float(np.max(np.abs(signal)))
        if rms < 1e-12:
            return 0.0
        return float(20 * np.log10(peak / rms))

    def lra_estimate(self, signal: np.ndarray, sr: int, block_s: float = 3.0) -> float:
        block_size = int(block_s * sr)
        if block_size < 1 or len(signal) < block_size:
            return 0.0
        hop = block_size
        loudnesses = []
        for start in range(0, len(signal) - block_size + 1, hop):
            block = signal[start:start + block_size]
            power = float(np.mean(block ** 2))
            if power > 1e-20:
                loudnesses.append(10 * np.log10(power))
        if len(loudnesses) < 2:
            return 0.0
        loudnesses.sort()
        low = loudnesses[int(0.10 * len(loudnesses))]
        high = loudnesses[int(0.95 * len(loudnesses))]
        return float(max(0, high - low))


def rms_loudness(signal: np.ndarray) -> float:
    return LoudnessAnalyzer().rms(signal)


def lufs_approximate(signal: np.ndarray, sr: int) -> float:
    return LoudnessAnalyzer().lufs_approximate(signal, sr)
