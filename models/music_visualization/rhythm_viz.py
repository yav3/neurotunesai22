"""Rhythm visualization: beat grids, onset plots, tempo graphs, groove maps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class RhythmGrid:
    @staticmethod
    def create(
        n_beats: int = 16,
        subdivisions: int = 4,
        active: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        grid = np.zeros((n_beats, subdivisions), dtype=np.float32)
        if active:
            for beat, sub in active:
                if 0 <= beat < n_beats and 0 <= sub < subdivisions:
                    grid[beat, sub] = 1.0
        return grid

    @staticmethod
    def from_onsets(
        onset_times: np.ndarray,
        bpm: float = 120.0,
        n_beats: int = 16,
        subdivisions: int = 4,
    ) -> np.ndarray:
        grid = np.zeros((n_beats, subdivisions), dtype=np.float32)
        beat_dur = 60.0 / bpm
        sub_dur = beat_dur / subdivisions
        for t in onset_times:
            total_sub = t / sub_dur
            beat = int(total_sub) // subdivisions
            sub = int(total_sub) % subdivisions
            if 0 <= beat < n_beats and 0 <= sub < subdivisions:
                grid[beat, sub] = 1.0
        return grid

    @staticmethod
    def density(grid: np.ndarray) -> float:
        if grid.size == 0:
            return 0.0
        return float(np.sum(grid > 0)) / grid.size


class OnsetPlot:
    @staticmethod
    def create(
        onset_strengths: np.ndarray,
        beat_times: np.ndarray | None = None,
        sr: int = 44100,
        hop: int = 512,
    ) -> dict[str, Any]:
        n = len(onset_strengths)
        times = np.arange(n) * hop / sr
        result: dict[str, Any] = {
            "times": times.astype(np.float32),
            "strengths": onset_strengths.astype(np.float32),
        }
        if beat_times is not None:
            result["beat_markers"] = beat_times.astype(np.float32)
        return result

    @staticmethod
    def peak_onsets(
        strengths: np.ndarray, threshold: float = 0.3
    ) -> np.ndarray:
        if len(strengths) < 3:
            return np.array([], dtype=int)
        mx = float(np.max(strengths))
        if mx < 1e-12:
            return np.array([], dtype=int)
        norm = strengths / mx
        peaks = []
        for i in range(1, len(norm) - 1):
            if norm[i] > norm[i - 1] and norm[i] > norm[i + 1] and norm[i] >= threshold:
                peaks.append(i)
        return np.array(peaks, dtype=int)


class TempoGraph:
    @staticmethod
    def from_beats(beat_times: np.ndarray) -> dict[str, np.ndarray]:
        if len(beat_times) < 2:
            return {"times": np.array([], dtype=np.float32), "bpm": np.array([], dtype=np.float32)}
        ioi = np.diff(beat_times)
        ioi = np.maximum(ioi, 1e-6)
        bpm = 60.0 / ioi
        times = beat_times[:-1]
        return {
            "times": times.astype(np.float32),
            "bpm": bpm.astype(np.float32),
        }

    @staticmethod
    def smoothed(
        beat_times: np.ndarray, window: int = 3
    ) -> dict[str, np.ndarray]:
        raw = TempoGraph.from_beats(beat_times)
        if len(raw["bpm"]) < window:
            return raw
        kernel = np.ones(window) / window
        smoothed_bpm = np.convolve(raw["bpm"], kernel, mode="same")
        return {
            "times": raw["times"],
            "bpm": smoothed_bpm.astype(np.float32),
        }

    @staticmethod
    def tempo_stability(beat_times: np.ndarray) -> float:
        if len(beat_times) < 3:
            return 1.0
        ioi = np.diff(beat_times)
        ioi = np.maximum(ioi, 1e-6)
        mean_ioi = float(np.mean(ioi))
        if mean_ioi < 1e-12:
            return 0.0
        cv = float(np.std(ioi)) / mean_ioi
        return max(0.0, 1.0 - cv)


class GrooveMap:
    @staticmethod
    def compute(
        onset_times: np.ndarray,
        bpm: float = 120.0,
        beats_per_bar: int = 4,
        n_bars: int | None = None,
    ) -> np.ndarray:
        if len(onset_times) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        beat_dur = 60.0 / bpm
        bar_dur = beat_dur * beats_per_bar
        if n_bars is None:
            n_bars = max(1, int(np.ceil(float(np.max(onset_times)) / bar_dur)))
        grid = np.full((n_bars, beats_per_bar), np.nan, dtype=np.float32)
        for t in onset_times:
            bar_idx = int(t / bar_dur)
            if bar_idx >= n_bars:
                continue
            beat_in_bar = (t - bar_idx * bar_dur) / beat_dur
            nearest_beat = int(round(beat_in_bar))
            if nearest_beat >= beats_per_bar:
                continue
            expected = bar_idx * bar_dur + nearest_beat * beat_dur
            deviation_ms = (t - expected) * 1000.0
            grid[bar_idx, nearest_beat] = deviation_ms
        return grid

    @staticmethod
    def mean_deviation(groove_map: np.ndarray) -> float:
        valid = groove_map[~np.isnan(groove_map)]
        if len(valid) == 0:
            return 0.0
        return float(np.mean(np.abs(valid)))
