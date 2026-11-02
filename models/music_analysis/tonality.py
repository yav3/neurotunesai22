"""Tonal analysis: mode mixture, chromatic content, harmonic rhythm."""
from __future__ import annotations
import numpy as np


MAJOR_SCALE_PCS = {0, 2, 4, 5, 7, 9, 11}


class TonalityAnalyzer:
    def chromatic_ratio(self, chroma: np.ndarray, key_root: int = 0) -> float:
        if chroma.ndim == 2:
            profile = np.mean(chroma, axis=1) if chroma.shape[0] == 12 else np.mean(chroma, axis=0)
        else:
            profile = np.asarray(chroma, dtype=float)
        if len(profile) != 12 or profile.sum() < 1e-12:
            return 0.0
        profile = profile / profile.sum()
        diatonic_pcs = {(pc + key_root) % 12 for pc in MAJOR_SCALE_PCS}
        chromatic_energy = sum(profile[pc] for pc in range(12) if pc not in diatonic_pcs)
        return float(chromatic_energy)

    def mode_mixture_score(self, chroma: np.ndarray, key_root: int = 0) -> float:
        if chroma.ndim == 2:
            profile = np.mean(chroma, axis=1) if chroma.shape[0] == 12 else np.mean(chroma, axis=0)
        else:
            profile = np.asarray(chroma, dtype=float)
        if len(profile) != 12 or profile.sum() < 1e-12:
            return 0.0
        profile = profile / profile.sum()
        major_third = (key_root + 4) % 12
        minor_third = (key_root + 3) % 12
        major_seventh = (key_root + 11) % 12
        minor_seventh = (key_root + 10) % 12
        mixture = profile[minor_third] + profile[minor_seventh]
        pure = profile[major_third] + profile[major_seventh]
        total = mixture + pure
        return float(mixture / total) if total > 1e-12 else 0.0


class HarmonicRhythm:
    def change_rate(self, chroma: np.ndarray) -> float:
        if chroma.ndim != 2:
            return 0.0
        if chroma.shape[0] == 12:
            chroma = chroma.T
        n_frames = chroma.shape[0]
        if n_frames < 2:
            return 0.0
        changes = 0
        for i in range(1, n_frames):
            a = chroma[i - 1]
            b = chroma[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12:
                cos_sim = float(np.dot(a, b) / (na * nb))
                if cos_sim < 0.9:
                    changes += 1
        return float(changes / (n_frames - 1))

    def average_chord_duration(self, chroma: np.ndarray, fps: float = 10.0) -> float:
        rate = self.change_rate(chroma)
        if rate < 1e-12:
            return float("inf")
        return 1.0 / (rate * fps)


def chromatic_ratio(chroma: np.ndarray, key_root: int = 0) -> float:
    return TonalityAnalyzer().chromatic_ratio(chroma, key_root)


def harmonic_change_rate(chroma: np.ndarray) -> float:
    return HarmonicRhythm().change_rate(chroma)
