"""Tonnetz visualization: pitch-class coordinates, chord trajectories, region coloring."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

FIFTH_AXIS = 7
MAJOR_THIRD_AXIS = 4
MINOR_THIRD_AXIS = 3


@dataclass
class TonnetzPoint:
    pc: int
    x: float
    y: float
    label: str = ""


@dataclass
class TonnetzEdge:
    src: int
    dst: int
    interval: int
    label: str = ""


@dataclass
class ChordRegion:
    pcs: tuple[int, ...]
    centroid_x: float
    centroid_y: float
    quality: str


def tonnetz_coordinates(pc: int) -> tuple[float, float]:
    fifth_steps = 0
    thirds_steps = 0
    current = 0
    for _ in range(12):
        if current == pc % 12:
            break
        if (current + FIFTH_AXIS) % 12 == pc % 12:
            fifth_steps += 1
            break
        fifth_steps += 1
        current = (current + FIFTH_AXIS) % 12
    x = (pc * 7) % 12
    y = (pc * 4) % 12
    return (float(x), float(y))


def chord_tonnetz_centroid(pcs: list[int]) -> tuple[float, float]:
    if not pcs:
        return (0.0, 0.0)
    coords = [tonnetz_coordinates(pc) for pc in pcs]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (float(np.mean(xs)), float(np.mean(ys)))


def chord_trajectory(chord_sequence: list[list[int]]) -> list[tuple[float, float]]:
    return [chord_tonnetz_centroid(ch) for ch in chord_sequence]


def tonnetz_distance(pcs_a: list[int], pcs_b: list[int]) -> float:
    ca = chord_tonnetz_centroid(pcs_a)
    cb = chord_tonnetz_centroid(pcs_b)
    dx = ca[0] - cb[0]
    dy = ca[1] - cb[1]
    return float(np.sqrt(dx ** 2 + dy ** 2))


def tonnetz_grid(n_fifths: int = 6, n_thirds: int = 4) -> list[TonnetzPoint]:
    points = []
    for i in range(-n_fifths, n_fifths + 1):
        for j in range(-n_thirds, n_thirds + 1):
            pc = (i * FIFTH_AXIS + j * MAJOR_THIRD_AXIS) % 12
            x = float(i)
            y = float(j)
            points.append(TonnetzPoint(pc, x, y, PITCH_NAMES[pc]))
    return points


def neo_riemannian_edges(pcs_from: list[int], transform: str) -> list[int]:
    s = set(pcs_from)
    if len(s) != 3:
        return list(pcs_from)
    root = min(s)
    intervals = sorted([(p - root) % 12 for p in s])
    if intervals == [0, 3, 7]:
        quality = "minor"
    elif intervals == [0, 4, 7]:
        quality = "major"
    else:
        return list(pcs_from)
    if transform.upper() == "P":
        if quality == "major":
            return [root, (root + 3) % 12, (root + 7) % 12]
        return [root, (root + 4) % 12, (root + 7) % 12]
    if transform.upper() == "R":
        if quality == "major":
            return [(root + 9) % 12, root, (root + 4) % 12]
        return [(root + 3) % 12, (root + 7) % 12, (root + 10) % 12]
    if transform.upper() == "L":
        if quality == "major":
            return [(root + 4) % 12, (root + 7) % 12, (root + 11) % 12]
        return [(root - 1) % 12, (root + 3) % 12, (root + 7) % 12]
    return list(pcs_from)


def harmonic_function_color(pcs: list[int], key_tonic: int = 0) -> str:
    if not pcs:
        return "neutral"
    root = (min(pcs) - key_tonic) % 12
    if root in (0, 4, 9):
        return "tonic"
    if root in (5, 2):
        return "subdominant"
    if root in (7, 11):
        return "dominant"
    return "chromatic"


def region_data(chord_sequence: list[list[int]], key_tonic: int = 0) -> list[ChordRegion]:
    regions = []
    for ch in chord_sequence:
        cx, cy = chord_tonnetz_centroid(ch)
        color = harmonic_function_color(ch, key_tonic)
        regions.append(ChordRegion(tuple(ch), cx, cy, color))
    return regions
