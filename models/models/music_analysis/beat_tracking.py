"""Beat tracking via onset detection and autocorrelation tempo estimation."""
from __future__ import annotations
import numpy as np


class TempoEstimator:
    def __init__(self, min_bpm: float = 40, max_bpm: float = 240):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def estimate(self, onset_env: np.ndarray, sr: int, hop_length: int = 512) -> float:
        if len(onset_env) < 4:
            return 120.0
        onset_env = onset_env - onset_env.mean()
        acf = np.correlate(onset_env, onset_env, mode="full")
        acf = acf[len(acf) // 2:]
        fps = sr / hop_length
        min_lag = max(1, int(fps * 60.0 / self.max_bpm))
        max_lag = min(len(acf) - 1, int(fps * 60.0 / self.min_bpm))
        if min_lag >= max_lag:
            return 120.0
        search = acf[min_lag:max_lag + 1]
        best_lag = min_lag + int(np.argmax(search))
        if best_lag < 1:
            return 120.0
        return float(60.0 * fps / best_lag)


class BeatTracker:
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        self._estimator = TempoEstimator()

    def track(self, onset_env: np.ndarray, tempo: float | None = None) -> np.ndarray:
        if tempo is None:
            tempo = self._estimator.estimate(onset_env, self.sr, self.hop_length)
        fps = self.sr / self.hop_length
        period_frames = fps * 60.0 / tempo
        if period_frames < 1 or len(onset_env) < 2:
            return np.array([])
        beats = []
        pos = 0.0
        while pos < len(onset_env):
            idx = int(round(pos))
            if idx >= len(onset_env):
                break
            window = max(0, idx - 2), min(len(onset_env), idx + 3)
            local = onset_env[window[0]:window[1]]
            if len(local) > 0:
                refined = window[0] + int(np.argmax(local))
            else:
                refined = idx
            beats.append(refined)
            pos += period_frames
        return np.unique(np.array(beats))

    def beats_to_times(self, beats: np.ndarray) -> np.ndarray:
        return beats * self.hop_length / self.sr


def estimate_tempo(onset_env: np.ndarray, sr: int, hop_length: int = 512) -> float:
    return TempoEstimator().estimate(onset_env, sr, hop_length)


def track_beats(onset_env: np.ndarray, sr: int, tempo: float, hop_length: int = 512) -> np.ndarray:
    return BeatTracker(sr=sr, hop_length=hop_length).track(onset_env, tempo)
