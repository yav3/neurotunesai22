"""Tempo utilities: beat grids, BPM estimation, groove templates, click tracks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BeatGrid:
    bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)
    offset_s: float = 0.0

    def beat_times(self, duration_s: float) -> np.ndarray:
        if self.bpm <= 0 or duration_s <= 0:
            return np.array([], dtype=np.float64)
        interval = 60.0 / self.bpm
        times = np.arange(self.offset_s, duration_s, interval)
        return times

    def bar_times(self, duration_s: float) -> np.ndarray:
        beats_per_bar = self.time_signature[0]
        beat_interval = 60.0 / self.bpm
        bar_interval = beat_interval * beats_per_bar
        if bar_interval <= 0 or duration_s <= 0:
            return np.array([], dtype=np.float64)
        return np.arange(self.offset_s, duration_s, bar_interval)

    def snap_to_grid(self, time_s: float) -> float:
        if self.bpm <= 0:
            return time_s
        interval = 60.0 / self.bpm
        beats_elapsed = round((time_s - self.offset_s) / interval)
        return self.offset_s + beats_elapsed * interval

    def beat_number(self, time_s: float) -> int:
        if self.bpm <= 0:
            return 0
        interval = 60.0 / self.bpm
        return max(0, int((time_s - self.offset_s) / interval))

    def is_downbeat(self, time_s: float) -> bool:
        beat_idx = self.beat_number(time_s)
        return beat_idx % self.time_signature[0] == 0


class TempoEstimator:
    @staticmethod
    def onset_strength(signal: np.ndarray, sr: int = 44100, hop: int = 512) -> np.ndarray:
        mono = signal if signal.ndim == 1 else signal.mean(axis=1)
        n = len(mono)
        if n < hop * 2:
            return np.array([], dtype=np.float32)
        n_frames = n // hop
        strengths = np.zeros(n_frames, dtype=np.float64)
        prev_fft = np.zeros(hop // 2 + 1)
        for i in range(n_frames):
            frame = mono[i * hop : i * hop + hop].astype(np.float64)
            if len(frame) < hop:
                frame = np.pad(frame, (0, hop - len(frame)))
            fft_mag = np.abs(np.fft.rfft(frame * np.hanning(hop)))
            diff = fft_mag - prev_fft
            strengths[i] = float(np.sum(np.maximum(diff, 0.0)))
            prev_fft = fft_mag
        return strengths.astype(np.float32)

    @staticmethod
    def estimate_bpm(
        signal: np.ndarray,
        sr: int = 44100,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
    ) -> float:
        odf = TempoEstimator.onset_strength(signal, sr)
        if len(odf) < 4:
            return 0.0
        odf = odf - np.mean(odf)
        corr = np.correlate(odf, odf, mode="full")
        corr = corr[len(corr) // 2 :]
        hop = 512
        min_lag = max(1, int(sr * 60.0 / (max_bpm * hop)))
        max_lag = min(len(corr) - 1, int(sr * 60.0 / (min_bpm * hop)))
        if min_lag >= max_lag:
            return 0.0
        search = corr[min_lag : max_lag + 1]
        best_idx = int(np.argmax(search))
        best_lag = min_lag + best_idx
        if best_lag <= 0:
            return 0.0
        return float(sr * 60.0 / (best_lag * hop))


@dataclass
class GrooveTemplate:
    name: str = "straight"
    swing_amount: float = 0.0
    humanize_ms: float = 0.0

    def apply(self, beat_times: np.ndarray, seed: int | None = None) -> np.ndarray:
        if len(beat_times) == 0:
            return beat_times.copy()
        out = beat_times.astype(np.float64).copy()
        if self.swing_amount > 0:
            for i in range(1, len(out), 2):
                interval = out[i] - out[i - 1] if i > 0 else 0.0
                out[i] += interval * self.swing_amount * 0.33
        if self.humanize_ms > 0:
            rng = np.random.default_rng(seed)
            jitter = rng.normal(0.0, self.humanize_ms / 1000.0, size=len(out))
            jitter[0] = 0.0
            out += jitter
        return out


class ClickTrack:
    @staticmethod
    def generate(
        bpm: float,
        duration_s: float,
        sr: int = 44100,
        time_signature: tuple[int, int] = (4, 4),
        accent_downbeat: bool = True,
    ) -> np.ndarray:
        if bpm <= 0 or duration_s <= 0:
            return np.zeros(0, dtype=np.float32)
        grid = BeatGrid(bpm=bpm, time_signature=time_signature)
        beats = grid.beat_times(duration_s)
        n_samples = int(sr * duration_s)
        out = np.zeros(n_samples, dtype=np.float64)
        click_dur = min(0.02, 60.0 / bpm / 4.0)
        click_samples = int(sr * click_dur)
        for i, t in enumerate(beats):
            idx = int(t * sr)
            if idx + click_samples > n_samples:
                break
            is_down = i % time_signature[0] == 0
            freq = 1500.0 if (accent_downbeat and is_down) else 1000.0
            amp = 0.8 if (accent_downbeat and is_down) else 0.5
            t_click = np.arange(click_samples) / sr
            click = amp * np.sin(2.0 * np.pi * freq * t_click)
            envelope = np.exp(-t_click * 80.0)
            out[idx : idx + click_samples] += click * envelope
        return np.clip(out, -1.0, 1.0).astype(np.float32)
