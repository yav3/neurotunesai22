"""Key detection via Krumhansl-Schmuckler pitch-class profile correlation."""
from __future__ import annotations
import numpy as np

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class KeyProfile:
    def __init__(self, chroma: np.ndarray):
        if chroma.ndim == 2:
            self.distribution = np.mean(chroma, axis=1) if chroma.shape[0] == 12 else np.mean(chroma, axis=0)
        else:
            self.distribution = np.asarray(chroma, dtype=float)
        if len(self.distribution) != 12:
            self.distribution = np.zeros(12)

    def normalize(self) -> np.ndarray:
        s = self.distribution.sum()
        return self.distribution / s if s > 1e-12 else self.distribution


class KeyDetector:
    def __init__(self):
        self._major = MAJOR_PROFILE
        self._minor = MINOR_PROFILE

    def detect(self, chroma: np.ndarray) -> tuple[str, str, float]:
        kp = KeyProfile(chroma)
        dist = kp.normalize()
        best_key, best_mode, best_corr = "C", "major", -1.0
        for shift in range(12):
            rotated = np.roll(dist, -shift)
            for mode, template in [("major", self._major), ("minor", self._minor)]:
                c = _pearson(rotated, template)
                if c > best_corr:
                    best_corr = c
                    best_key = PITCH_CLASSES[shift]
                    best_mode = mode
        return best_key, best_mode, float(best_corr)

    def all_correlations(self, chroma: np.ndarray) -> list[tuple[str, str, float]]:
        kp = KeyProfile(chroma)
        dist = kp.normalize()
        results = []
        for shift in range(12):
            rotated = np.roll(dist, -shift)
            for mode, template in [("major", self._major), ("minor", self._minor)]:
                results.append((PITCH_CLASSES[shift], mode, float(_pearson(rotated, template))))
        results.sort(key=lambda x: x[2], reverse=True)
        return results


def detect_key(chroma: np.ndarray) -> tuple[str, str, float]:
    return KeyDetector().detect(chroma)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = float(np.dot(a_c, b_c))
    den = float(np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c)))
    return num / den if den > 1e-12 else 0.0
