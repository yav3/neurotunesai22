"""Tempo expression: rubato, accelerando/ritardando, agogic accents, metrical strength."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class RubatoDetector:
    @staticmethod
    def inter_onset_intervals(beat_times: np.ndarray) -> np.ndarray:
        if len(beat_times) < 2:
            return np.array([], dtype=np.float64)
        return np.diff(beat_times.astype(np.float64))

    @staticmethod
    def rubato_index(beat_times: np.ndarray) -> float:
        ioi = RubatoDetector.inter_onset_intervals(beat_times)
        if len(ioi) < 2:
            return 0.0
        mean_ioi = np.mean(ioi)
        if mean_ioi < 1e-12:
            return 0.0
        return float(np.std(ioi) / mean_ioi)

    @staticmethod
    def classify(rubato_idx: float) -> str:
        if rubato_idx < 0.03:
            return "strict"
        elif rubato_idx < 0.10:
            return "moderate"
        else:
            return "free"

    @staticmethod
    def analyze_sections(
        beat_times: np.ndarray, section_beats: int = 8
    ) -> list[dict]:
        if len(beat_times) < section_beats + 1:
            return []
        results: list[dict] = []
        for i in range(0, len(beat_times) - section_beats, section_beats):
            section = beat_times[i : i + section_beats + 1]
            idx = RubatoDetector.rubato_index(section)
            results.append(
                {
                    "start_beat": i,
                    "end_beat": i + section_beats,
                    "start_time": float(section[0]),
                    "end_time": float(section[-1]),
                    "rubato_index": idx,
                    "classification": RubatoDetector.classify(idx),
                }
            )
        return results


class AccelerandoDetector:
    @staticmethod
    def detect(
        beat_times: np.ndarray, min_beats: int = 4, threshold: float = 0.05
    ) -> list[dict]:
        ioi = RubatoDetector.inter_onset_intervals(beat_times)
        if len(ioi) < min_beats:
            return []

        results: list[dict] = []
        i = 0
        while i < len(ioi) - min_beats + 1:
            segment_ioi = ioi[i : i + min_beats]
            diffs = np.diff(segment_ioi)

            if np.all(diffs < 0):
                end_idx = i + min_beats
                while end_idx < len(ioi) and ioi[end_idx] < ioi[end_idx - 1]:
                    end_idx += 1
                seg = ioi[i:end_idx]
                change = (seg[0] - seg[-1]) / max(seg[0], 1e-12)
                if change > threshold:
                    results.append(
                        {
                            "type": "accelerando",
                            "start_beat": i,
                            "end_beat": end_idx,
                            "start_time": float(beat_times[i]),
                            "end_time": float(beat_times[min(end_idx, len(beat_times) - 1)]),
                            "tempo_change_ratio": float(change),
                        }
                    )
                i = end_idx
            elif np.all(diffs > 0):
                end_idx = i + min_beats
                while end_idx < len(ioi) and ioi[end_idx] > ioi[end_idx - 1]:
                    end_idx += 1
                seg = ioi[i:end_idx]
                change = (seg[-1] - seg[0]) / max(seg[0], 1e-12)
                if change > threshold:
                    results.append(
                        {
                            "type": "ritardando",
                            "start_beat": i,
                            "end_beat": end_idx,
                            "start_time": float(beat_times[i]),
                            "end_time": float(beat_times[min(end_idx, len(beat_times) - 1)]),
                            "tempo_change_ratio": float(change),
                        }
                    )
                i = end_idx
            else:
                i += 1
        return results

    @staticmethod
    def tempo_curve(beat_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ioi = RubatoDetector.inter_onset_intervals(beat_times)
        if len(ioi) == 0:
            return np.array([]), np.array([])
        bpm = 60.0 / np.maximum(ioi, 1e-12)
        mid_times = (beat_times[:-1] + beat_times[1:]) / 2.0
        return mid_times, bpm


class AgogicAccent:
    @staticmethod
    def detect(
        note_durations: np.ndarray,
        nominal_duration: float,
        threshold_ratio: float = 1.15,
    ) -> np.ndarray:
        if len(note_durations) == 0 or nominal_duration <= 0:
            return np.array([], dtype=bool)
        ratios = note_durations / nominal_duration
        return ratios >= threshold_ratio

    @staticmethod
    def accent_strength(
        note_durations: np.ndarray, nominal_duration: float
    ) -> np.ndarray:
        if len(note_durations) == 0 or nominal_duration <= 0:
            return np.array([], dtype=np.float64)
        ratios = note_durations / nominal_duration
        return np.maximum(ratios - 1.0, 0.0)


class MetricalStrength:
    @staticmethod
    def beat_emphasis(
        onset_amplitudes: np.ndarray, beats_per_bar: int = 4
    ) -> dict:
        if len(onset_amplitudes) == 0 or beats_per_bar < 1:
            return {"strong_mean": 0.0, "weak_mean": 0.0, "emphasis_ratio": 1.0}

        amps = onset_amplitudes.astype(np.float64)
        strong_indices = np.arange(0, len(amps), beats_per_bar)
        weak_indices = np.array(
            [i for i in range(len(amps)) if i % beats_per_bar != 0]
        )

        strong_mean = float(np.mean(amps[strong_indices])) if len(strong_indices) > 0 else 0.0
        weak_mean = float(np.mean(amps[weak_indices])) if len(weak_indices) > 0 else 0.0

        ratio = strong_mean / max(weak_mean, 1e-12)
        return {
            "strong_mean": strong_mean,
            "weak_mean": weak_mean,
            "emphasis_ratio": float(ratio),
        }

    @staticmethod
    def downbeat_pattern(
        onset_amplitudes: np.ndarray, beats_per_bar: int = 4
    ) -> list[float]:
        if len(onset_amplitudes) == 0 or beats_per_bar < 1:
            return []
        amps = onset_amplitudes.astype(np.float64)
        pattern: list[float] = []
        for pos in range(beats_per_bar):
            indices = np.arange(pos, len(amps), beats_per_bar)
            if len(indices) > 0:
                pattern.append(float(np.mean(amps[indices])))
            else:
                pattern.append(0.0)
        total = sum(pattern)
        if total > 1e-12:
            pattern = [p / total for p in pattern]
        return pattern
