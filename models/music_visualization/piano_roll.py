"""Piano roll visualization: MIDI grid, velocity mapping, overlays, labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass
class NoteLabel:
    @staticmethod
    def from_midi(midi_note: int, use_flats: bool = False) -> str:
        names = FLAT_NAMES if use_flats else SHARP_NAMES
        octave = (midi_note // 12) - 1
        pc = midi_note % 12
        return f"{names[pc]}{octave}"

    @staticmethod
    def labels_for_range(
        low: int = 21, high: int = 108, use_flats: bool = False
    ) -> list[str]:
        return [NoteLabel.from_midi(m, use_flats) for m in range(low, high + 1)]


class VelocityColorMap:
    @staticmethod
    def map_velocity(velocity: int, base_color: tuple[int, int, int] = (0, 120, 255)) -> tuple[int, int, int]:
        t = max(0, min(velocity, 127)) / 127.0
        return (
            int(base_color[0] * t),
            int(base_color[1] * t),
            int(base_color[2] * t),
        )

    @staticmethod
    def intensity(velocity: int) -> float:
        return max(0, min(velocity, 127)) / 127.0

    @staticmethod
    def gradient(n_steps: int = 128) -> np.ndarray:
        grad = np.zeros((n_steps, 3), dtype=np.uint8)
        for i in range(n_steps):
            grad[i] = VelocityColorMap.map_velocity(i)
        return grad


class PianoRollRenderer:
    @staticmethod
    def render(
        notes: list[dict[str, Any]],
        duration_s: float | None = None,
        time_resolution: float = 0.01,
        pitch_range: tuple[int, int] = (21, 108),
    ) -> np.ndarray:
        if not notes:
            h = pitch_range[1] - pitch_range[0] + 1
            w = 1
            return np.zeros((h, w), dtype=np.float32)
        if duration_s is None:
            duration_s = max(n.get("start", 0) + n.get("duration", 0) for n in notes)
            duration_s = max(duration_s, 0.01)
        n_cols = max(1, int(duration_s / time_resolution))
        n_rows = pitch_range[1] - pitch_range[0] + 1
        grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        for note in notes:
            pitch = note.get("pitch", 60)
            start = note.get("start", 0.0)
            dur = note.get("duration", 0.1)
            vel = note.get("velocity", 100)
            row = pitch - pitch_range[0]
            if row < 0 or row >= n_rows:
                continue
            col_start = int(start / time_resolution)
            col_end = int((start + dur) / time_resolution)
            col_start = max(0, min(col_start, n_cols - 1))
            col_end = max(col_start + 1, min(col_end, n_cols))
            intensity = VelocityColorMap.intensity(vel)
            grid[row, col_start:col_end] = intensity
        return grid

    @staticmethod
    def to_rgb(
        grid: np.ndarray,
        base_color: tuple[int, int, int] = (0, 120, 255),
        bg_color: tuple[int, int, int] = (20, 20, 30),
    ) -> np.ndarray:
        h, w = grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = np.where(
                grid > 0,
                np.clip(grid * base_color[c], 0, 255).astype(np.uint8),
                bg_color[c],
            )
        return rgb


class PianoRollOverlay:
    @staticmethod
    def beat_grid(
        n_cols: int, time_resolution: float = 0.01, bpm: float = 120.0
    ) -> np.ndarray:
        beat_interval = 60.0 / bpm
        markers = np.zeros(n_cols, dtype=np.float32)
        t = 0.0
        while True:
            col = int(t / time_resolution)
            if col >= n_cols:
                break
            markers[col] = 1.0
            t += beat_interval
        return markers

    @staticmethod
    def measure_lines(
        n_cols: int,
        time_resolution: float = 0.01,
        bpm: float = 120.0,
        beats_per_measure: int = 4,
    ) -> np.ndarray:
        bar_interval = 60.0 / bpm * beats_per_measure
        markers = np.zeros(n_cols, dtype=np.float32)
        t = 0.0
        while True:
            col = int(t / time_resolution)
            if col >= n_cols:
                break
            markers[col] = 1.0
            t += bar_interval
        return markers

    @staticmethod
    def key_shading(
        n_rows: int, root: int = 0, pitch_low: int = 21
    ) -> np.ndarray:
        major_scale = {0, 2, 4, 5, 7, 9, 11}
        shading = np.zeros(n_rows, dtype=np.float32)
        for row in range(n_rows):
            midi = row + pitch_low
            pc = (midi - root) % 12
            shading[row] = 1.0 if pc in major_scale else 0.3
        return shading
