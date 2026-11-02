"""Spectrogram visualization: STFT, mel-scale, log-frequency, colormapping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class SpectrogramRenderer:
    @staticmethod
    def compute(
        signal: np.ndarray,
        sr: int = 44100,
        n_fft: int = 2048,
        hop: int = 512,
        window: str = "hann",
    ) -> np.ndarray:
        mono = signal if signal.ndim == 1 else signal.mean(axis=-1)
        mono = mono.astype(np.float64)
        n = len(mono)
        if n < n_fft:
            mono = np.pad(mono, (0, n_fft - n))
            n = n_fft
        if window == "hann":
            win = np.hanning(n_fft)
        elif window == "hamming":
            win = np.hamming(n_fft)
        elif window == "blackman":
            win = np.blackman(n_fft)
        else:
            win = np.ones(n_fft)
        n_frames = 1 + (n - n_fft) // hop
        freq_bins = n_fft // 2 + 1
        spec = np.zeros((freq_bins, n_frames), dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            frame = mono[start : start + n_fft] * win
            spec[:, i] = np.abs(np.fft.rfft(frame))
        return spec.astype(np.float32)

    @staticmethod
    def to_db(spec: np.ndarray, ref: float = 1.0, floor_db: float = -80.0) -> np.ndarray:
        mag = np.maximum(spec, 1e-10)
        db = 20.0 * np.log10(mag / max(ref, 1e-10))
        return np.maximum(db, floor_db).astype(np.float32)

    @staticmethod
    def frequency_axis(n_fft: int = 2048, sr: int = 44100) -> np.ndarray:
        return np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    @staticmethod
    def time_axis(n_frames: int, hop: int = 512, sr: int = 44100) -> np.ndarray:
        return (np.arange(n_frames) * hop / sr).astype(np.float32)


class MelSpectrogram:
    @staticmethod
    def mel_filterbank(sr: int = 44100, n_fft: int = 2048, n_mels: int = 128) -> np.ndarray:
        freq_bins = n_fft // 2 + 1
        f_min = 0.0
        f_max = sr / 2.0
        mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        fb = np.zeros((n_mels, freq_bins), dtype=np.float64)
        for m in range(n_mels):
            left = bin_points[m]
            center = bin_points[m + 1]
            right = bin_points[m + 2]
            for k in range(left, center):
                if center > left:
                    fb[m, k] = (k - left) / (center - left)
            for k in range(center, right):
                if right > center:
                    fb[m, k] = (right - k) / (right - center)
        return fb.astype(np.float32)

    @staticmethod
    def compute(
        signal: np.ndarray,
        sr: int = 44100,
        n_fft: int = 2048,
        hop: int = 512,
        n_mels: int = 128,
    ) -> np.ndarray:
        spec = SpectrogramRenderer.compute(signal, sr, n_fft, hop)
        fb = MelSpectrogram.mel_filterbank(sr, n_fft, n_mels)
        power = spec.astype(np.float64) ** 2
        mel = fb @ power
        return np.maximum(mel, 1e-10).astype(np.float32)


class LogSpectrogram:
    @staticmethod
    def compute(
        spec: np.ndarray,
        sr: int = 44100,
        n_fft: int = 2048,
        n_log_bins: int = 256,
        f_min: float = 20.0,
        f_max: float | None = None,
    ) -> np.ndarray:
        if f_max is None:
            f_max = sr / 2.0
        freq_bins, n_frames = spec.shape
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        log_freqs = np.logspace(
            np.log10(max(f_min, 1.0)), np.log10(f_max), n_log_bins
        )
        log_spec = np.zeros((n_log_bins, n_frames), dtype=np.float32)
        for i, f in enumerate(log_freqs):
            idx = int(np.argmin(np.abs(freqs - f)))
            idx = min(idx, freq_bins - 1)
            log_spec[i, :] = spec[idx, :]
        return log_spec


class SpectrogramColormap:
    COLORMAPS = {
        "viridis": np.array([
            [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
            [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
            [121, 209, 81], [189, 222, 38], [253, 231, 37],
        ], dtype=np.uint8),
        "magma": np.array([
            [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
            [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 127],
            [252, 253, 191],
        ], dtype=np.uint8),
        "inferno": np.array([
            [0, 0, 4], [40, 11, 84], [101, 0, 167], [159, 42, 99],
            [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
        ], dtype=np.uint8),
        "plasma": np.array([
            [13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150],
            [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40],
            [240, 249, 33],
        ], dtype=np.uint8),
    }

    @staticmethod
    def apply(
        magnitude: np.ndarray,
        colormap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> np.ndarray:
        lut = SpectrogramColormap.COLORMAPS.get(
            colormap, SpectrogramColormap.COLORMAPS["viridis"]
        )
        n_colors = len(lut)
        if magnitude.size == 0:
            return np.zeros((*magnitude.shape, 3), dtype=np.uint8)
        if vmin is None:
            vmin = float(np.min(magnitude))
        if vmax is None:
            vmax = float(np.max(magnitude))
        rng = vmax - vmin
        if rng < 1e-12:
            norm = np.zeros_like(magnitude, dtype=np.float64)
        else:
            norm = np.clip((magnitude.astype(np.float64) - vmin) / rng, 0.0, 1.0)
        indices = norm * (n_colors - 1)
        lo = np.floor(indices).astype(int)
        lo = np.clip(lo, 0, n_colors - 2)
        hi = lo + 1
        frac = (indices - lo)[..., np.newaxis]
        rgb = (1.0 - frac) * lut[lo] + frac * lut[hi]
        return np.clip(rgb, 0, 255).astype(np.uint8)
