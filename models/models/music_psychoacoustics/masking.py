"""Auditory masking: simultaneous masking, temporal masking, spread of masking."""
from __future__ import annotations
import numpy as np
from models.music_psychoacoustics.critical_bands import hz_to_bark


class SimultaneousMasking:
    def masking_threshold(
        self, masker_freq: float, masker_level: float, probe_freq: float,
    ) -> float:
        z_m = float(hz_to_bark(masker_freq))
        z_p = float(hz_to_bark(probe_freq))
        dz = z_p - z_m
        if dz >= 0:
            slope = 27.0 - 0.37 * max(masker_level - 40.0, 0.0)
            spread = slope * dz
        else:
            spread = -27.0 * dz
        threshold = masker_level - spread
        return max(threshold, 0.0)

    def masked(
        self, masker_freq: float, masker_level: float,
        probe_freq: float, probe_level: float,
    ) -> bool:
        threshold = self.masking_threshold(masker_freq, masker_level, probe_freq)
        return probe_level < threshold

    def composite_threshold(
        self, maskers: list[tuple[float, float]], probe_freq: float,
    ) -> float:
        if not maskers:
            return 0.0
        thresholds = [
            self.masking_threshold(f, l, probe_freq) for f, l in maskers
        ]
        linear = sum(10.0 ** (t / 10.0) for t in thresholds)
        return 10.0 * np.log10(max(linear, 1e-12))


class TemporalMasking:
    def forward_masking(self, delay_ms: float, masker_level: float) -> float:
        if delay_ms <= 0:
            return masker_level
        decay_rate = 10.0
        threshold = masker_level - decay_rate * np.log10(max(delay_ms, 0.1))
        return max(threshold, 0.0)

    def backward_masking(self, advance_ms: float, masker_level: float) -> float:
        if advance_ms <= 0:
            return masker_level
        if advance_ms > 20.0:
            return 0.0
        return masker_level * max(1.0 - advance_ms / 20.0, 0.0)


class MaskingModel:
    def __init__(self):
        self.simultaneous = SimultaneousMasking()
        self.temporal = TemporalMasking()

    def audibility(
        self, probe_freq: float, probe_level: float, probe_time: float,
        maskers: list[tuple[float, float, float]],
    ) -> float:
        sim_maskers = [(f, l) for f, l, _ in maskers]
        sim_threshold = self.simultaneous.composite_threshold(sim_maskers, probe_freq)
        temp_thresholds = []
        for _, level, time in maskers:
            dt = (probe_time - time) * 1000.0
            if dt >= 0:
                temp_thresholds.append(self.temporal.forward_masking(dt, level))
            else:
                temp_thresholds.append(self.temporal.backward_masking(-dt, level))
        temp_threshold = max(temp_thresholds) if temp_thresholds else 0.0
        effective_threshold = max(sim_threshold, temp_threshold)
        if probe_level <= effective_threshold:
            return 0.0
        return min((probe_level - effective_threshold) / max(probe_level, 1e-12), 1.0)
