"""Mastering chain: loudness metering, limiting, stereo widening, M/S encoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class LoudnessMeter:
    @staticmethod
    def measure_rms(signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))

    @staticmethod
    def measure_peak(signal: np.ndarray) -> float:
        return float(np.max(np.abs(signal)))

    @staticmethod
    def measure_lufs(signal: np.ndarray, sr: int = 44100) -> float:
        rms = LoudnessMeter.measure_rms(signal)
        if rms < 1e-12:
            return -np.inf
        return float(20.0 * np.log10(rms) - 0.691)

    @staticmethod
    def measure_dynamic_range(signal: np.ndarray) -> float:
        peak = LoudnessMeter.measure_peak(signal)
        rms = LoudnessMeter.measure_rms(signal)
        if rms < 1e-12 or peak < 1e-12:
            return 0.0
        return float(20.0 * np.log10(peak / rms))

    @staticmethod
    def measure_crest_factor(signal: np.ndarray) -> float:
        rms = LoudnessMeter.measure_rms(signal)
        peak = LoudnessMeter.measure_peak(signal)
        if rms < 1e-12:
            return 0.0
        return float(peak / rms)


class StereoWidener:
    @staticmethod
    def mid_side_encode(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if stereo.ndim == 1:
            return stereo.copy(), np.zeros_like(stereo)
        left = stereo[:, 0].astype(np.float64)
        right = stereo[:, 1].astype(np.float64)
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        return mid.astype(np.float32), side.astype(np.float32)

    @staticmethod
    def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
        m = mid.astype(np.float64)
        s = side.astype(np.float64)
        left = m + s
        right = m - s
        return np.column_stack([left, right]).astype(np.float32)

    @staticmethod
    def process(stereo: np.ndarray, width_factor: float = 1.0) -> np.ndarray:
        mid, side = StereoWidener.mid_side_encode(stereo)
        side = side * width_factor
        return StereoWidener.mid_side_decode(mid, side)


class BrickwallLimiter:
    @staticmethod
    def process(
        signal: np.ndarray,
        ceiling_db: float = -0.3,
        release_ms: float = 50.0,
    ) -> np.ndarray:
        ceiling_lin = 10.0 ** (ceiling_db / 20.0)
        out = signal.astype(np.float64).copy()
        abs_sig = np.abs(out)
        over = abs_sig > ceiling_lin
        if not np.any(over):
            return out.astype(np.float32)
        gain_reduction = np.ones_like(out)
        gain_reduction[over] = ceiling_lin / abs_sig[over]
        out *= gain_reduction
        return np.clip(out, -ceiling_lin, ceiling_lin).astype(np.float32)


class MasteringStage(Protocol):
    def process(self, signal: np.ndarray, sr: int) -> np.ndarray: ...


@dataclass
class EQStage:
    low_shelf_db: float = 0.0
    high_shelf_db: float = 0.0
    low_freq: float = 100.0
    high_freq: float = 8000.0

    def process(self, signal: np.ndarray, sr: int) -> np.ndarray:
        n = len(signal) if signal.ndim == 1 else signal.shape[0]
        if n == 0:
            return signal.copy()
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        if signal.ndim == 1:
            fft = np.fft.rfft(signal.astype(np.float64))
            gains = np.ones_like(freqs)
            low_mask = freqs < self.low_freq
            high_mask = freqs > self.high_freq
            gains[low_mask] *= 10.0 ** (self.low_shelf_db / 20.0)
            gains[high_mask] *= 10.0 ** (self.high_shelf_db / 20.0)
            fft *= gains
            return np.fft.irfft(fft, n=n).astype(np.float32)
        else:
            out = signal.astype(np.float64).copy()
            for ch in range(signal.shape[1]):
                fft = np.fft.rfft(out[:, ch])
                gains = np.ones_like(freqs)
                low_mask = freqs < self.low_freq
                high_mask = freqs > self.high_freq
                gains[low_mask] *= 10.0 ** (self.low_shelf_db / 20.0)
                gains[high_mask] *= 10.0 ** (self.high_shelf_db / 20.0)
                fft *= gains
                out[:, ch] = np.fft.irfft(fft, n=n)
            return out.astype(np.float32)


@dataclass
class CompressionStage:
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0

    def process(self, signal: np.ndarray, sr: int) -> np.ndarray:
        threshold_lin = 10.0 ** (self.threshold_db / 20.0)
        out = signal.astype(np.float64).copy()
        flat = out.ravel()
        abs_sig = np.abs(flat)
        over = abs_sig > threshold_lin
        if not np.any(over):
            return out.astype(np.float32)
        excess_db = 20.0 * np.log10(np.maximum(abs_sig[over], 1e-12) / threshold_lin)
        compressed_db = excess_db / self.ratio
        target_db = 20.0 * np.log10(threshold_lin) + compressed_db
        target_lin = 10.0 ** (target_db / 20.0)
        gain = target_lin / np.maximum(abs_sig[over], 1e-12)
        flat[over] *= gain
        return out.astype(np.float32)


class MasteringChain:
    def __init__(self, stages: list[Any] | None = None) -> None:
        self._stages: list[Any] = list(stages or [])
        self._before: dict[str, float] = {}
        self._after: dict[str, float] = {}

    def add_stage(self, stage: Any) -> None:
        self._stages.append(stage)

    @property
    def stages(self) -> list[Any]:
        return list(self._stages)

    def process(self, signal: np.ndarray, sr: int = 44100) -> np.ndarray:
        meter = LoudnessMeter()
        self._before = {
            "lufs": meter.measure_lufs(signal, sr),
            "peak": meter.measure_peak(signal),
            "rms": meter.measure_rms(signal),
        }
        out = signal.copy()
        for stage in self._stages:
            out = stage.process(out, sr)
        self._after = {
            "lufs": meter.measure_lufs(out, sr),
            "peak": meter.measure_peak(out),
            "rms": meter.measure_rms(out),
        }
        return out

    def analyze_before_after(self) -> dict[str, dict[str, float]]:
        return {"before": dict(self._before), "after": dict(self._after)}
