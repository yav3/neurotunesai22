"""Notation engraving: spacing, stems, beams, collision detection, layout."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class StaffMetrics:
    staff_height: float = 4.0
    line_spacing: float = 1.0
    stem_length: float = 3.5
    beam_thickness: float = 0.5
    note_spacing_base: float = 1.2


def compute_note_spacing(
    durations: np.ndarray,
    base_width: float = 1.2,
    stretch: float = 1.6,
) -> np.ndarray:
    if len(durations) == 0:
        return np.array([])
    log_dur = np.log2(np.maximum(durations, 1e-6))
    ref = np.log2(0.25)
    spacing = base_width * (stretch ** (log_dur - ref))
    return np.maximum(spacing, base_width * 0.4)


def stem_direction(pitches: np.ndarray, middle_line: int = 71) -> np.ndarray:
    if len(pitches) == 0:
        return np.array([], dtype=int)
    return np.where(pitches >= middle_line, -1, 1).astype(int)


def beam_slope(note_positions: np.ndarray, max_slope: float = 0.5) -> float:
    if len(note_positions) < 2:
        return 0.0
    x = np.arange(len(note_positions), dtype=float)
    x_mean = np.mean(x)
    y_mean = np.mean(note_positions)
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    slope = float(np.sum((x - x_mean) * (note_positions - y_mean)) / denom)
    return max(-max_slope, min(slope, max_slope))


def collision_detect(bboxes: List[Tuple[float, float, float, float]]) -> List[Tuple[int, int]]:
    collisions = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        for j in range(i + 1, len(bboxes)):
            ax1, ay1, ax2, ay2 = bboxes[j]
            if x1 < ax2 and x2 > ax1 and y1 < ay2 and y2 > ay1:
                collisions.append((i, j))
    return collisions


def lyric_hyphenation_points(
    syllables: List[str],
    note_x_positions: np.ndarray,
) -> np.ndarray:
    n = min(len(syllables), len(note_x_positions))
    return note_x_positions[:n].copy()


def page_break_penalty(
    measures_per_system: List[int],
    ideal: int = 4,
) -> float:
    if not measures_per_system:
        return 0.0
    deviations = [(m - ideal) ** 2 for m in measures_per_system]
    return float(np.mean(deviations))


def accidental_stacking(accidental_pitches: np.ndarray) -> np.ndarray:
    if len(accidental_pitches) == 0:
        return np.array([])
    sorted_p = np.sort(accidental_pitches)[::-1]
    columns = np.zeros(len(sorted_p), dtype=int)
    placed: list[list[float]] = [[]]
    for i, p in enumerate(sorted_p):
        col_found = False
        for c, col_pitches in enumerate(placed):
            if all(abs(p - pp) >= 2 for pp in col_pitches):
                columns[i] = c
                col_pitches.append(float(p))
                col_found = True
                break
        if not col_found:
            columns[i] = len(placed)
            placed.append([float(p)])
    return columns


def tie_arc_height(distance: float, min_height: float = 0.5) -> float:
    return max(min_height, 0.3 * np.sqrt(max(distance, 0.0)))


def optimal_system_breaks(
    measure_widths: np.ndarray,
    page_width: float,
    penalty_exp: float = 2.0,
) -> List[int]:
    n = len(measure_widths)
    if n == 0:
        return []
    breaks = []
    current_width = 0.0
    for i in range(n):
        if current_width + measure_widths[i] > page_width and current_width > 0:
            breaks.append(i)
            current_width = measure_widths[i]
        else:
            current_width += measure_widths[i]
    return breaks
