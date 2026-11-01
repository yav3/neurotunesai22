"""Groove analysis: syncopation, micro-timing, swing measurement."""
from __future__ import annotations
import numpy as np

METRIC_WEIGHTS_4_4 = [4, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]


class GrooveAnalyzer:
    def syncopation_index(
        self, onsets: np.ndarray, meter: tuple[int, int] = (4, 4),
        tempo: float = 120.0, sr: int = 1,
    ) -> float:
        if len(onsets) < 2:
            return 0.0
        beats_per_bar = meter[0]
        subdivisions = beats_per_bar * 4
        beat_dur = 60.0 / tempo
        subdiv_dur = beat_dur / 4
        weights = (METRIC_WEIGHTS_4_4 * ((subdivisions // 16) + 1))[:subdivisions]
        syncopation = 0.0
        count = 0
        for onset in onsets:
            t = float(onset) / sr if sr > 1 else float(onset)
            pos = int(round(t / subdiv_dur)) % subdivisions
            if pos < len(weights):
                w = weights[pos]
                if w <= 2:
                    syncopation += (4 - w) / 3.0
                count += 1
        return float(syncopation / count) if count > 0 else 0.0

    def grid_deviation(self, onsets: np.ndarray, tempo: float, sr: int = 1) -> np.ndarray:
        if len(onsets) == 0:
            return np.array([])
        beat_dur = 60.0 / tempo
        times = onsets / sr if sr > 1 else onsets.astype(float)
        deviations = []
        for t in times:
            nearest_beat = round(t / beat_dur) * beat_dur
            deviations.append(t - nearest_beat)
        return np.array(deviations)


class MicrotimingProfile:
    def analyze(self, onsets: np.ndarray, tempo: float, sr: int = 1) -> dict[str, float]:
        ga = GrooveAnalyzer()
        devs = ga.grid_deviation(onsets, tempo, sr)
        if len(devs) == 0:
            return {"mean_deviation": 0.0, "std_deviation": 0.0, "tendency": 0.0}
        return {
            "mean_deviation": float(np.mean(np.abs(devs))),
            "std_deviation": float(np.std(devs)),
            "tendency": float(np.mean(devs)),
        }


def syncopation_index(
    onsets: np.ndarray, meter: tuple[int, int] = (4, 4), tempo: float = 120.0
) -> float:
    return GrooveAnalyzer().syncopation_index(onsets, meter, tempo)


def swing_ratio(onset_pairs: np.ndarray) -> float:
    if len(onset_pairs) < 2:
        return 1.0
    pairs = np.asarray(onset_pairs, dtype=float)
    if pairs.ndim == 1:
        iois = np.diff(pairs)
        if len(iois) < 2:
            return 1.0
        ratios = []
        for i in range(0, len(iois) - 1, 2):
            short = min(iois[i], iois[i + 1])
            if short > 1e-12:
                ratios.append(max(iois[i], iois[i + 1]) / short)
        return float(np.mean(ratios)) if ratios else 1.0
    ratios = []
    for row in pairs:
        if len(row) >= 2 and row[1] > 1e-12:
            ratios.append(row[0] / row[1])
    return float(np.mean(ratios)) if ratios else 1.0
