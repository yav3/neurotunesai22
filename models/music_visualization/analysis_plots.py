"""Analysis plot data: loudness, dynamic range, frequency response, stereo field."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class LoudnessCurve:
    @staticmethod
    def compute(
        signal: np.ndarray,
        sr: int = 44100,
        window_ms: float = 400.0,
        hop_ms: float = 100.0,
    ) -> dict[str, np.ndarray]:
        mono = signal if signal.ndim == 1 else signal.mean(axis=-1)
        mono = mono.astype(np.float64)
        win_samples = max(1, int(sr * window_ms / 1000.0))
        hop_samples = max(1, int(sr * hop_ms / 1000.0))
        n = len(mono)
        if n < win_samples:
            return {
                "times": np.array([0.0], dtype=np.float32),
                "lufs": np.array([-70.0], dtype=np.float32),
            }
        n_frames = 1 + (n - win_samples) // hop_samples
        times = np.zeros(n_frames, dtype=np.float32)
        lufs = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_samples
            end = start + win_samples
            chunk = mono[start:end]
            rms_sq = float(np.mean(chunk ** 2))
            if rms_sq < 1e-20:
                lufs[i] = -70.0
            else:
                lufs[i] = float(-0.691 + 10.0 * np.log10(rms_sq))
            times[i] = float(start + win_samples // 2) / sr
        return {"times": times, "lufs": lufs}

    @staticmethod
    def integrated(signal: np.ndarray, sr: int = 44100) -> float:
        mono = signal if signal.ndim == 1 else signal.mean(axis=-1)
        mono = mono.astype(np.float64)
        rms_sq = float(np.mean(mono ** 2))
        if rms_sq < 1e-20:
            return -70.0
        return float(-0.691 + 10.0 * np.log10(rms_sq))


class DynamicRangePlot:
    @staticmethod
    def compute(
        signal: np.ndarray,
        sr: int = 44100,
        window_ms: float = 50.0,
    ) -> dict[str, np.ndarray]:
        mono = signal if signal.ndim == 1 else signal.mean(axis=-1)
        mono = mono.astype(np.float64)
        win = max(1, int(sr * window_ms / 1000.0))
        n = len(mono)
        if n == 0:
            return {
                "times": np.array([], dtype=np.float32),
                "peak": np.array([], dtype=np.float32),
                "rms": np.array([], dtype=np.float32),
            }
        n_frames = max(1, n // win)
        times = np.zeros(n_frames, dtype=np.float32)
        peak = np.zeros(n_frames, dtype=np.float32)
        rms = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            start = i * win
            end = min(start + win, n)
            chunk = mono[start:end]
            peak[i] = float(np.max(np.abs(chunk)))
            rms[i] = float(np.sqrt(np.mean(chunk ** 2)))
            times[i] = float(start + (end - start) // 2) / sr
        return {"times": times, "peak": peak, "rms": rms}

    @staticmethod
    def crest_factor_series(
        signal: np.ndarray, sr: int = 44100, window_ms: float = 50.0
    ) -> dict[str, np.ndarray]:
        data = DynamicRangePlot.compute(signal, sr, window_ms)
        safe_rms = np.maximum(data["rms"], 1e-10)
        crest = data["peak"] / safe_rms
        return {"times": data["times"], "crest": crest.astype(np.float32)}


class FrequencyResponse:
    @staticmethod
    def compute(
        signal: np.ndarray,
        sr: int = 44100,
        n_fft: int = 4096,
    ) -> dict[str, np.ndarray]:
        mono = signal if signal.ndim == 1 else signal.mean(axis=-1)
        mono = mono.astype(np.float64)
        n = len(mono)
        if n < n_fft:
            mono = np.pad(mono, (0, n_fft - n))
        win = np.hanning(n_fft)
        spectrum = np.abs(np.fft.rfft(mono[:n_fft] * win))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        mag_db = 20.0 * np.log10(np.maximum(spectrum, 1e-10))
        return {
            "frequencies": freqs.astype(np.float32),
            "magnitude_db": mag_db.astype(np.float32),
        }

    @staticmethod
    def smoothed(
        signal: np.ndarray, sr: int = 44100, n_fft: int = 4096, octave_fraction: float = 3.0
    ) -> dict[str, np.ndarray]:
        raw = FrequencyResponse.compute(signal, sr, n_fft)
        freqs = raw["frequencies"]
        mag = raw["magnitude_db"]
        smoothed = np.copy(mag)
        for i in range(1, len(freqs)):
            f = freqs[i]
            if f < 20.0:
                continue
            ratio = 2.0 ** (1.0 / octave_fraction)
            lo = f / ratio
            hi = f * ratio
            mask = (freqs >= lo) & (freqs <= hi)
            if np.any(mask):
                smoothed[i] = float(np.mean(mag[mask]))
        return {
            "frequencies": freqs,
            "magnitude_db": smoothed.astype(np.float32),
        }


class StereoFieldPlot:
    @staticmethod
    def lissajous(
        stereo: np.ndarray, max_points: int = 10000
    ) -> dict[str, np.ndarray]:
        if stereo.ndim != 2 or stereo.shape[1] < 2:
            return {"x": np.array([], dtype=np.float32), "y": np.array([], dtype=np.float32)}
        left = stereo[:, 0].astype(np.float64)
        right = stereo[:, 1].astype(np.float64)
        if len(left) > max_points:
            step = len(left) // max_points
            left = left[::step]
            right = right[::step]
        x = (left + right) / 2.0
        y = (left - right) / 2.0
        return {
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
        }

    @staticmethod
    def goniometer(
        stereo: np.ndarray, max_points: int = 10000
    ) -> dict[str, np.ndarray]:
        data = StereoFieldPlot.lissajous(stereo, max_points)
        x, y = data["x"].astype(np.float64), data["y"].astype(np.float64)
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return {
            "r": r.astype(np.float32),
            "theta": theta.astype(np.float32),
        }

    @staticmethod
    def correlation(stereo: np.ndarray) -> float:
        if stereo.ndim != 2 or stereo.shape[1] < 2 or stereo.shape[0] == 0:
            return 0.0
        left = stereo[:, 0].astype(np.float64)
        right = stereo[:, 1].astype(np.float64)
        l_std = float(np.std(left))
        r_std = float(np.std(right))
        if l_std < 1e-12 or r_std < 1e-12:
            return 0.0
        return float(np.mean((left - np.mean(left)) * (right - np.mean(right))) / (l_std * r_std))

    @staticmethod
    def width_estimate(stereo: np.ndarray) -> float:
        corr = StereoFieldPlot.correlation(stereo)
        return max(0.0, 1.0 - abs(corr))
