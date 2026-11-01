"""Production analysis: spectral balance, dynamics profiling, stereo field, reference comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class SpectralBalance:
    LOW_CUTOFF = 250.0
    MID_CUTOFF = 4000.0

    @staticmethod
    def band_energy(signal: np.ndarray, sr: int, low: float, high: float) -> float:
        n = len(signal) if signal.ndim == 1 else signal.shape[0]
        if n == 0:
            return 0.0
        mono = signal if signal.ndim == 1 else signal.mean(axis=1)
        fft_mag = np.abs(np.fft.rfft(mono.astype(np.float64)))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(fft_mag[mask] ** 2))

    @classmethod
    def analyze(cls, signal: np.ndarray, sr: int = 44100) -> dict[str, float]:
        low = cls.band_energy(signal, sr, 20.0, cls.LOW_CUTOFF)
        mid = cls.band_energy(signal, sr, cls.LOW_CUTOFF, cls.MID_CUTOFF)
        high = cls.band_energy(signal, sr, cls.MID_CUTOFF, sr / 2.0)
        total = low + mid + high
        if total < 1e-20:
            return {"low": 0.0, "mid": 0.0, "high": 0.0, "tilt_db": 0.0, "centroid_hz": 0.0}
        return {
            "low": low / total,
            "mid": mid / total,
            "high": high / total,
            "tilt_db": float(10.0 * np.log10(max(low, 1e-20) / max(high, 1e-20))),
            "centroid_hz": cls._spectral_centroid(signal, sr),
        }

    @staticmethod
    def _spectral_centroid(signal: np.ndarray, sr: int) -> float:
        mono = signal if signal.ndim == 1 else signal.mean(axis=1)
        n = len(mono)
        if n == 0:
            return 0.0
        fft_mag = np.abs(np.fft.rfft(mono.astype(np.float64)))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        total = np.sum(fft_mag)
        if total < 1e-12:
            return 0.0
        return float(np.sum(freqs * fft_mag) / total)


class DynamicProfile:
    @staticmethod
    def analyze(signal: np.ndarray) -> dict[str, float]:
        if signal.size == 0:
            return {"crest_factor": 0.0, "dynamic_range_db": 0.0, "rms": 0.0, "peak": 0.0}
        flat = signal.ravel().astype(np.float64)
        rms = float(np.sqrt(np.mean(flat ** 2)))
        peak = float(np.max(np.abs(flat)))
        if rms < 1e-12 or peak < 1e-12:
            return {"crest_factor": 0.0, "dynamic_range_db": 0.0, "rms": rms, "peak": peak}
        crest = peak / rms
        dr_db = 20.0 * np.log10(peak / rms)
        return {
            "crest_factor": float(crest),
            "dynamic_range_db": float(dr_db),
            "rms": rms,
            "peak": peak,
        }

    @staticmethod
    def loudness_envelope(signal: np.ndarray, window_ms: float = 50.0, sr: int = 44100) -> np.ndarray:
        mono = signal if signal.ndim == 1 else signal.mean(axis=1)
        win = max(1, int(sr * window_ms / 1000.0))
        n = len(mono)
        if n == 0:
            return np.array([], dtype=np.float32)
        squared = mono.astype(np.float64) ** 2
        kernel = np.ones(win) / win
        env = np.convolve(squared, kernel, mode="same")
        return np.sqrt(np.maximum(env, 0.0)).astype(np.float32)


class StereoAnalysis:
    @staticmethod
    def analyze(stereo: np.ndarray) -> dict[str, float]:
        if stereo.ndim != 2 or stereo.shape[1] < 2:
            return {"correlation": 1.0, "width": 0.0, "balance_db": 0.0, "phase_issues": False}
        left = stereo[:, 0].astype(np.float64)
        right = stereo[:, 1].astype(np.float64)
        l_rms = np.sqrt(np.mean(left ** 2))
        r_rms = np.sqrt(np.mean(right ** 2))
        if l_rms < 1e-12 or r_rms < 1e-12:
            return {"correlation": 0.0, "width": 0.0, "balance_db": 0.0, "phase_issues": False}
        corr = float(np.mean(left * right) / (l_rms * r_rms))
        width = float(1.0 - abs(corr))
        balance = float(20.0 * np.log10(max(r_rms, 1e-12) / max(l_rms, 1e-12)))
        mid = left + right
        side = left - right
        mid_energy = np.mean(mid ** 2)
        side_energy = np.mean(side ** 2)
        phase_issues = bool(side_energy > mid_energy * 4.0)
        return {
            "correlation": corr,
            "width": width,
            "balance_db": balance,
            "phase_issues": phase_issues,
        }


@dataclass
class ReferenceComparison:
    target_lufs: float = -14.0
    target_crest: float = 8.0
    target_low_pct: float = 0.35
    target_mid_pct: float = 0.40
    target_high_pct: float = 0.25

    def compare(self, signal: np.ndarray, sr: int = 44100) -> dict[str, Any]:
        from models.music_production.mastering import LoudnessMeter

        lufs = LoudnessMeter.measure_lufs(signal, sr)
        dynamics = DynamicProfile.analyze(signal)
        spectral = SpectralBalance.analyze(signal, sr)
        return {
            "lufs_diff": lufs - self.target_lufs,
            "crest_diff": dynamics["crest_factor"] - self.target_crest,
            "low_diff": spectral["low"] - self.target_low_pct,
            "mid_diff": spectral["mid"] - self.target_mid_pct,
            "high_diff": spectral["high"] - self.target_high_pct,
            "measured": {"lufs": lufs, "crest": dynamics["crest_factor"], "spectral": spectral},
        }
