"""Waveform visualization: peak envelopes, region annotation, thumbnails."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WaveformRegion:
    start_time: float
    end_time: float
    label: str = ""
    color: tuple[int, int, int] = (100, 149, 237)

    def contains(self, t: float) -> bool:
        return self.start_time <= t <= self.end_time

    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class WaveformRenderer:
    @staticmethod
    def peak_envelope(
        signal: np.ndarray, num_bins: int = 800
    ) -> tuple[np.ndarray, np.ndarray]:
        mono = signal if signal.ndim == 1 else signal.mean(axis=-1)
        n = len(mono)
        if n == 0 or num_bins <= 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        num_bins = min(num_bins, n)
        bin_size = n / num_bins
        pos = np.zeros(num_bins, dtype=np.float32)
        neg = np.zeros(num_bins, dtype=np.float32)
        for i in range(num_bins):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            end = max(end, start + 1)
            chunk = mono[start:end]
            pos[i] = float(np.max(chunk))
            neg[i] = float(np.min(chunk))
        return pos, neg

    @staticmethod
    def rms_envelope(
        signal: np.ndarray, num_bins: int = 800
    ) -> np.ndarray:
        mono = signal if signal.ndim == 1 else signal.mean(axis=-1)
        n = len(mono)
        if n == 0 or num_bins <= 0:
            return np.zeros(0, dtype=np.float32)
        num_bins = min(num_bins, n)
        bin_size = n / num_bins
        rms = np.zeros(num_bins, dtype=np.float32)
        for i in range(num_bins):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            end = max(end, start + 1)
            chunk = mono[start:end].astype(np.float64)
            rms[i] = float(np.sqrt(np.mean(chunk ** 2)))
        return rms

    @staticmethod
    def segment_colors_by_energy(
        signal: np.ndarray,
        num_bins: int = 800,
        low_color: tuple[int, int, int] = (0, 0, 255),
        high_color: tuple[int, int, int] = (255, 0, 0),
    ) -> np.ndarray:
        rms = WaveformRenderer.rms_envelope(signal, num_bins)
        if len(rms) == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        mx = float(np.max(rms))
        if mx < 1e-12:
            t = np.zeros(len(rms))
        else:
            t = rms / mx
        colors = np.zeros((len(rms), 3), dtype=np.uint8)
        for c in range(3):
            colors[:, c] = np.clip(
                low_color[c] + t * (high_color[c] - low_color[c]), 0, 255
            ).astype(np.uint8)
        return colors


class WaveformThumbnail:
    @staticmethod
    def generate(
        signal: np.ndarray, width: int = 200, height: int = 50
    ) -> np.ndarray:
        if signal.size == 0 or width <= 0 or height <= 0:
            return np.zeros((height, width), dtype=np.uint8)
        pos, neg = WaveformRenderer.peak_envelope(signal, width)
        img = np.zeros((height, width), dtype=np.uint8)
        mid = height // 2
        for x in range(len(pos)):
            top = mid - int(pos[x] * mid)
            bot = mid - int(neg[x] * mid)
            top = max(0, min(top, height - 1))
            bot = max(0, min(bot, height - 1))
            if top > bot:
                top, bot = bot, top
            img[top : bot + 1, x] = 255
        return img
