"""Harmonic circle visualizations: circle of fifths, chroma wheel, Tonnetz."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PITCH_NAMES = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
CHROMATIC_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {"Db": "C#", "Eb": "D#", "Ab": "G#", "Bb": "A#"}
FIFTHS_SEMITONES = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]


class CircleOfFifths:
    @staticmethod
    def positions() -> list[dict[str, Any]]:
        result = []
        for i, name in enumerate(PITCH_NAMES):
            angle = 2.0 * math.pi * i / 12 - math.pi / 2
            result.append({
                "name": name,
                "index": i,
                "semitone": FIFTHS_SEMITONES[i],
                "angle_rad": angle,
                "x": math.cos(angle),
                "y": math.sin(angle),
            })
        return result

    @staticmethod
    def highlight(active_notes: list[str]) -> list[dict[str, Any]]:
        pos = CircleOfFifths.positions()
        normalized = set()
        for n in active_notes:
            normalized.add(ENHARMONIC.get(n, n))
        for p in pos:
            canon = ENHARMONIC.get(p["name"], p["name"])
            p["active"] = canon in normalized
        return pos

    @staticmethod
    def chord_positions(root: str, quality: str = "major") -> list[dict[str, Any]]:
        root_canon = ENHARMONIC.get(root, root)
        if root_canon not in CHROMATIC_ORDER:
            return []
        root_idx = CHROMATIC_ORDER.index(root_canon)
        if quality == "major":
            intervals = [0, 4, 7]
        elif quality == "minor":
            intervals = [0, 3, 7]
        elif quality == "diminished":
            intervals = [0, 3, 6]
        elif quality == "augmented":
            intervals = [0, 4, 8]
        else:
            intervals = [0, 4, 7]
        notes = [CHROMATIC_ORDER[(root_idx + iv) % 12] for iv in intervals]
        return CircleOfFifths.highlight(notes)


class ChromaWheel:
    @staticmethod
    def map_chroma(chroma: np.ndarray, radius: float = 1.0) -> list[dict[str, Any]]:
        if len(chroma) != 12:
            return []
        result = []
        for i in range(12):
            angle = 2.0 * math.pi * i / 12 - math.pi / 2
            intensity = float(chroma[i])
            result.append({
                "pitch_class": CHROMATIC_ORDER[i],
                "x": radius * math.cos(angle),
                "y": radius * math.sin(angle),
                "intensity": intensity,
                "angle_rad": angle,
            })
        return result

    @staticmethod
    def centroid(chroma: np.ndarray) -> tuple[float, float]:
        if len(chroma) != 12:
            return 0.0, 0.0
        total = float(np.sum(chroma))
        if total < 1e-12:
            return 0.0, 0.0
        cx, cy = 0.0, 0.0
        for i in range(12):
            angle = 2.0 * math.pi * i / 12 - math.pi / 2
            w = float(chroma[i]) / total
            cx += w * math.cos(angle)
            cy += w * math.sin(angle)
        return cx, cy


class TonnetzProjection:
    @staticmethod
    def pitch_to_tonnetz(pitch_class: int) -> tuple[float, float, float]:
        fifths = (pitch_class * 7) % 12
        major_thirds = (pitch_class * 4) % 12
        minor_thirds = (pitch_class * 3) % 12
        return float(fifths), float(major_thirds), float(minor_thirds)

    @staticmethod
    def pitch_to_2d(pitch_class: int) -> tuple[float, float]:
        fifth_x = math.cos(2.0 * math.pi * ((pitch_class * 7) % 12) / 12)
        third_y = math.sin(2.0 * math.pi * ((pitch_class * 4) % 12) / 12)
        return fifth_x, third_y

    @staticmethod
    def chord_triangle(root: int, quality: str = "major") -> list[tuple[float, float]]:
        if quality == "major":
            pitches = [root, (root + 4) % 12, (root + 7) % 12]
        elif quality == "minor":
            pitches = [root, (root + 3) % 12, (root + 7) % 12]
        else:
            pitches = [root, (root + 4) % 12, (root + 7) % 12]
        return [TonnetzProjection.pitch_to_2d(p) for p in pitches]

    @staticmethod
    def project_chroma(chroma: np.ndarray) -> list[dict[str, Any]]:
        if len(chroma) != 12:
            return []
        result = []
        for pc in range(12):
            x, y = TonnetzProjection.pitch_to_2d(pc)
            result.append({
                "pitch_class": pc,
                "name": CHROMATIC_ORDER[pc],
                "x": x,
                "y": y,
                "intensity": float(chroma[pc]),
            })
        return result


class KeyRegion:
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

    @staticmethod
    def get_region(root: str, mode: str = "major") -> list[dict[str, Any]]:
        root_canon = ENHARMONIC.get(root, root)
        if root_canon not in CHROMATIC_ORDER:
            return []
        root_idx = CHROMATIC_ORDER.index(root_canon)
        scale = KeyRegion.MAJOR_SCALE if mode == "major" else KeyRegion.MINOR_SCALE
        scale_pcs = set((root_idx + s) % 12 for s in scale)
        positions = CircleOfFifths.positions()
        for p in positions:
            p["in_key"] = p["semitone"] in scale_pcs
        return positions
