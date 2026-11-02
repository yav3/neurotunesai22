"""Harmonic generation: voice leading, chord progressions, accompaniment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


COMMON_PROGRESSIONS = {
    "I-IV-V-I": [0, 5, 7, 0],
    "I-V-vi-IV": [0, 7, 9, 5],
    "ii-V-I": [2, 7, 0],
    "I-vi-IV-V": [0, 9, 5, 7],
    "vi-IV-I-V": [9, 5, 0, 7],
    "I-IV-vi-V": [0, 5, 9, 7],
    "i-bVII-bVI-V": [0, 10, 8, 7],
    "i-iv-V-i": [0, 5, 7, 0],
    "I-iii-IV-V": [0, 4, 5, 7],
    "I-bVII-IV-I": [0, 10, 5, 0],
    "12-bar-blues": [0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 0, 7],
}


class VoiceLeadingRules:
    @staticmethod
    def has_parallel_fifths(voice_a: list[int], voice_b: list[int]) -> bool:
        for i in range(len(voice_a) - 1):
            interval_1 = abs(voice_a[i] - voice_b[i]) % 12
            interval_2 = abs(voice_a[i + 1] - voice_b[i + 1]) % 12
            if interval_1 == 7 and interval_2 == 7:
                if voice_a[i] != voice_a[i + 1] or voice_b[i] != voice_b[i + 1]:
                    return True
        return False

    @staticmethod
    def has_voice_crossing(voices: list[list[int]]) -> bool:
        for step in range(len(voices[0])):
            pitches = [v[step] for v in voices]
            if pitches != sorted(pitches, reverse=True):
                return True
        return False

    @staticmethod
    def smooth_voice_leading_cost(chord_a: list[int], chord_b: list[int]) -> float:
        if len(chord_a) != len(chord_b):
            return float("inf")
        return float(sum(abs(a - b) for a, b in zip(chord_a, chord_b)))


class ChordProgressionGenerator:
    def __init__(self, root: int = 0, mode: str = "major"):
        self._root = root
        self._mode = mode

    def generate_from_template(self, template_name: str, n_repeats: int = 1) -> list[int]:
        pattern = COMMON_PROGRESSIONS.get(template_name, [0, 5, 7, 0])
        return [(self._root + deg) % 12 for deg in pattern * n_repeats]

    def generate_random(
        self, n_chords: int, seed: int | None = None
    ) -> list[int]:
        rng = np.random.default_rng(seed)
        if self._mode == "major":
            pool = [0, 2, 4, 5, 7, 9]
        else:
            pool = [0, 2, 3, 5, 7, 8, 10]
        roots = [(self._root + pool[rng.integers(len(pool))]) % 12 for _ in range(n_chords)]
        return roots

    def available_templates(self) -> list[str]:
        return list(COMMON_PROGRESSIONS.keys())


class HarmonySuggester:
    MAJOR_CHORDS = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}

    def suggest_chords(
        self,
        melody_notes: list[int],
        key_root: int = 0,
        beats_per_chord: int = 4,
    ) -> list[dict[str, Any]]:
        if not melody_notes:
            return []

        suggestions = []
        for i in range(0, len(melody_notes), beats_per_chord):
            segment = melody_notes[i:i + beats_per_chord]
            pitch_classes = [(n % 12 - key_root) % 12 for n in segment]
            best_root, best_score = 0, -1
            for root_deg, quality in self.MAJOR_CHORDS.items():
                if quality == "maj":
                    chord_pcs = {root_deg, (root_deg + 4) % 12, (root_deg + 7) % 12}
                elif quality == "min":
                    chord_pcs = {root_deg, (root_deg + 3) % 12, (root_deg + 7) % 12}
                else:
                    chord_pcs = {root_deg, (root_deg + 3) % 12, (root_deg + 6) % 12}
                score = sum(1 for pc in pitch_classes if pc in chord_pcs)
                if score > best_score:
                    best_score = score
                    best_root = root_deg
                    best_quality = quality
            suggestions.append({
                "root_degree": best_root,
                "quality": best_quality,
                "absolute_root": (key_root + best_root) % 12,
                "fit_score": best_score / max(len(segment), 1),
            })
        return suggestions
