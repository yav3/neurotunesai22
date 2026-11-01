"""Circular visualizations: circle of fifths, pitch-class distributions, interval radials."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]


@dataclass
class CirclePoint:
    pc: int
    angle_rad: float
    x: float
    y: float
    label: str


@dataclass
class ArcPath:
    start_pc: int
    end_pc: int
    start_angle: float
    end_angle: float
    weight: float = 1.0


def circle_of_fifths_layout(radius: float = 1.0) -> list[CirclePoint]:
    points = []
    for i, pc in enumerate(FIFTHS_ORDER):
        angle = 2 * np.pi * i / 12 - np.pi / 2
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append(CirclePoint(pc, float(angle), float(x), float(y), PITCH_NAMES[pc]))
    return points


def chromatic_circle_layout(radius: float = 1.0) -> list[CirclePoint]:
    points = []
    for pc in range(12):
        angle = 2 * np.pi * pc / 12 - np.pi / 2
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append(CirclePoint(pc, float(angle), float(x), float(y), PITCH_NAMES[pc]))
    return points


def pitch_class_distribution(pitches: list[int], use_fifths: bool = False) -> np.ndarray:
    hist = np.zeros(12)
    for p in pitches:
        hist[p % 12] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    if use_fifths:
        reordered = np.zeros(12)
        for i, pc in enumerate(FIFTHS_ORDER):
            reordered[i] = hist[pc]
        return reordered
    return hist


def key_region_highlight(key_tonic: int, is_minor: bool = False) -> list[int]:
    if is_minor:
        intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]
    return [(key_tonic + iv) % 12 for iv in intervals]


def chord_progression_arcs(chord_roots: list[int],
                           use_fifths: bool = True) -> list[ArcPath]:
    arcs = []
    order = FIFTHS_ORDER if use_fifths else list(range(12))
    for i in range(len(chord_roots) - 1):
        src = chord_roots[i] % 12
        dst = chord_roots[i + 1] % 12
        src_idx = order.index(src)
        dst_idx = order.index(dst)
        src_angle = 2 * np.pi * src_idx / 12 - np.pi / 2
        dst_angle = 2 * np.pi * dst_idx / 12 - np.pi / 2
        arcs.append(ArcPath(src, dst, float(src_angle), float(dst_angle)))
    return arcs


def interval_vector_radial(pcs: list[int]) -> np.ndarray:
    ic_vector = np.zeros(6)
    unique_pcs = sorted(set(p % 12 for p in pcs))
    for i in range(len(unique_pcs)):
        for j in range(i + 1, len(unique_pcs)):
            diff = (unique_pcs[j] - unique_pcs[i]) % 12
            ic = min(diff, 12 - diff)
            if 1 <= ic <= 6:
                ic_vector[ic - 1] += 1
    return ic_vector


def fifths_distance(pc_a: int, pc_b: int) -> int:
    pos_a = FIFTHS_ORDER.index(pc_a % 12)
    pos_b = FIFTHS_ORDER.index(pc_b % 12)
    d = abs(pos_a - pos_b)
    return min(d, 12 - d)


def angular_distance(pc_a: int, pc_b: int, use_fifths: bool = True) -> float:
    order = FIFTHS_ORDER if use_fifths else list(range(12))
    pos_a = order.index(pc_a % 12)
    pos_b = order.index(pc_b % 12)
    d = abs(pos_a - pos_b)
    d = min(d, 12 - d)
    return float(2 * np.pi * d / 12)
