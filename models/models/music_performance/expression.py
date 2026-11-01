"""Expression analysis: dynamics, vibrato, portamento, articulation detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

import numpy as np


class DynamicLevel(Enum):
    PP = "pp"
    P = "p"
    MP = "mp"
    MF = "mf"
    F = "f"
    FF = "ff"


DYNAMIC_THRESHOLDS: list[tuple[float, DynamicLevel]] = [
    (0.05, DynamicLevel.PP),
    (0.15, DynamicLevel.P),
    (0.30, DynamicLevel.MP),
    (0.50, DynamicLevel.MF),
    (0.75, DynamicLevel.F),
]


@dataclass
class ExpressionCurve:
    times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    label: str = "expression"

    def duration(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return float(self.times[-1] - self.times[0])

    def mean_value(self) -> float:
        if len(self.values) == 0:
            return 0.0
        return float(np.mean(self.values))

    def range_value(self) -> float:
        if len(self.values) == 0:
            return 0.0
        return float(np.max(self.values) - np.min(self.values))

    def interpolate(self, t: float) -> float:
        if len(self.times) == 0:
            return 0.0
        if len(self.times) == 1:
            return float(self.values[0])
        return float(np.interp(t, self.times, self.values))

    def smoothed(self, window: int = 5) -> "ExpressionCurve":
        if len(self.values) < window:
            return ExpressionCurve(
                times=self.times.copy(), values=self.values.copy(), label=self.label
            )
        kernel = np.ones(window) / window
        smoothed = np.convolve(self.values, kernel, mode="same")
        return ExpressionCurve(
            times=self.times.copy(), values=smoothed, label=self.label
        )


class DynamicsAnalyzer:
    @staticmethod
    def classify_level(rms: float) -> DynamicLevel:
        for threshold, level in DYNAMIC_THRESHOLDS:
            if rms < threshold:
                return level
        return DynamicLevel.FF

    @staticmethod
    def amplitude_envelope(
        signal: np.ndarray, sr: int = 44100, hop_ms: float = 10.0
    ) -> ExpressionCurve:
        mono = signal if signal.ndim == 1 else signal.mean(axis=1)
        hop = max(1, int(sr * hop_ms / 1000.0))
        n_frames = max(1, len(mono) // hop)
        times = np.arange(n_frames) * hop / sr
        values = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            end = min(start + hop, len(mono))
            frame = mono[start:end].astype(np.float64)
            values[i] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0
        return ExpressionCurve(times=times, values=values, label="dynamics")

    @staticmethod
    def detect_crescendo(
        envelope: ExpressionCurve, min_duration_s: float = 0.5, min_ratio: float = 1.5
    ) -> list[dict]:
        if len(envelope.values) < 3:
            return []
        results: list[dict] = []
        smoothed = envelope.smoothed(window=7)
        vals = smoothed.values
        times = smoothed.times
        i = 0
        while i < len(vals) - 1:
            if vals[i + 1] > vals[i]:
                start_idx = i
                while i < len(vals) - 1 and vals[i + 1] >= vals[i]:
                    i += 1
                end_idx = i
                dur = times[end_idx] - times[start_idx]
                start_val = max(vals[start_idx], 1e-12)
                ratio = vals[end_idx] / start_val
                if dur >= min_duration_s and ratio >= min_ratio:
                    results.append(
                        {
                            "type": "crescendo",
                            "start_time": float(times[start_idx]),
                            "end_time": float(times[end_idx]),
                            "start_level": float(vals[start_idx]),
                            "end_level": float(vals[end_idx]),
                            "ratio": float(ratio),
                        }
                    )
            i += 1
        return results

    @staticmethod
    def detect_sforzando(
        envelope: ExpressionCurve, threshold_ratio: float = 3.0
    ) -> list[dict]:
        if len(envelope.values) < 3:
            return []
        results: list[dict] = []
        vals = envelope.values
        times = envelope.times
        for i in range(1, len(vals) - 1):
            prev_val = max(vals[i - 1], 1e-12)
            if vals[i] / prev_val >= threshold_ratio and vals[i] > vals[i + 1]:
                results.append(
                    {
                        "type": "sforzando",
                        "time": float(times[i]),
                        "level": float(vals[i]),
                        "ratio": float(vals[i] / prev_val),
                    }
                )
        return results

    @staticmethod
    def dynamic_profile(
        envelope: ExpressionCurve, window_frames: int = 10
    ) -> list[dict]:
        if len(envelope.values) < window_frames:
            return []
        results: list[dict] = []
        for i in range(0, len(envelope.values) - window_frames + 1, window_frames):
            chunk = envelope.values[i : i + window_frames]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            level = DynamicsAnalyzer.classify_level(rms)
            t_start = float(envelope.times[i])
            t_end = float(envelope.times[min(i + window_frames - 1, len(envelope.times) - 1)])
            results.append(
                {"start_time": t_start, "end_time": t_end, "rms": rms, "level": level.value}
            )
        return results


class VibratoAnalyzer:
    @staticmethod
    def detect(
        pitch_hz: np.ndarray,
        sr_pitch: float = 100.0,
        min_rate: float = 4.0,
        max_rate: float = 8.0,
    ) -> dict:
        if len(pitch_hz) < 8:
            return {"detected": False, "rate_hz": 0.0, "extent_cents": 0.0, "classification": "none"}

        valid = pitch_hz > 0
        if np.sum(valid) < 8:
            return {"detected": False, "rate_hz": 0.0, "extent_cents": 0.0, "classification": "none"}

        pitch_valid = pitch_hz[valid].astype(np.float64)
        mean_pitch = np.mean(pitch_valid)
        cents_deviation = 1200.0 * np.log2(pitch_valid / mean_pitch)
        cents_deviation -= np.mean(cents_deviation)

        fft = np.abs(np.fft.rfft(cents_deviation * np.hanning(len(cents_deviation))))
        freqs = np.fft.rfftfreq(len(cents_deviation), d=1.0 / sr_pitch)

        mask = (freqs >= min_rate) & (freqs <= max_rate)
        if not np.any(mask):
            return {"detected": False, "rate_hz": 0.0, "extent_cents": 0.0, "classification": "none"}

        fft_band = fft.copy()
        fft_band[~mask] = 0.0
        peak_idx = int(np.argmax(fft_band))
        peak_power = fft_band[peak_idx]
        total_power = np.sum(fft) + 1e-12

        if peak_power / total_power < 0.1:
            return {"detected": False, "rate_hz": 0.0, "extent_cents": 0.0, "classification": "none"}

        rate = float(freqs[peak_idx])
        extent = float(np.std(cents_deviation) * 2.0)

        if extent < 10:
            cls = "narrow"
        elif extent < 40:
            cls = "medium"
        else:
            cls = "wide"

        return {"detected": True, "rate_hz": rate, "extent_cents": extent, "classification": cls}

    @staticmethod
    def vibrato_curve(
        pitch_hz: np.ndarray, sr_pitch: float = 100.0
    ) -> ExpressionCurve:
        if len(pitch_hz) == 0:
            return ExpressionCurve(label="vibrato")
        valid = pitch_hz > 0
        mean_p = np.mean(pitch_hz[valid]) if np.any(valid) else 440.0
        cents = np.where(
            pitch_hz > 0, 1200.0 * np.log2(np.maximum(pitch_hz, 1.0) / mean_p), 0.0
        )
        times = np.arange(len(cents)) / sr_pitch
        return ExpressionCurve(times=times, values=cents, label="vibrato_cents")


class ArticulationDetector:
    @staticmethod
    def classify(
        onset_sharpness: float, sustain_ratio: float, gap_ratio: float = 0.0
    ) -> str:
        if onset_sharpness > 0.8 and sustain_ratio < 0.3:
            return "staccato"
        if onset_sharpness > 0.8 and sustain_ratio >= 0.3:
            return "marcato"
        if onset_sharpness > 0.6:
            return "accent"
        if sustain_ratio > 0.9 and gap_ratio < 0.05:
            return "legato"
        if sustain_ratio > 0.85:
            return "tenuto"
        return "normal"

    @staticmethod
    def analyze_notes(
        onsets: np.ndarray,
        offsets: np.ndarray,
        amplitudes: np.ndarray,
        nominal_duration: float | None = None,
    ) -> list[dict]:
        n = min(len(onsets), len(offsets), len(amplitudes))
        if n == 0:
            return []

        results: list[dict] = []
        for i in range(n):
            dur = offsets[i] - onsets[i]
            if dur <= 0:
                continue
            nom = nominal_duration if nominal_duration else dur
            sustain_ratio = min(dur / max(nom, 1e-9), 1.0)

            if i > 0:
                gap = onsets[i] - offsets[i - 1]
                gap_ratio = max(gap, 0.0) / max(nom, 1e-9)
            else:
                gap_ratio = 0.0

            peak_amp = float(amplitudes[i])
            if i > 0:
                prev_amp = max(float(amplitudes[i - 1]), 1e-12)
                onset_sharpness = min(peak_amp / prev_amp, 1.0) if peak_amp <= prev_amp else min(peak_amp / prev_amp / 3.0, 1.0)
            else:
                onset_sharpness = 0.5

            art = ArticulationDetector.classify(onset_sharpness, sustain_ratio, gap_ratio)
            results.append(
                {
                    "note_index": i,
                    "onset": float(onsets[i]),
                    "duration": float(dur),
                    "sustain_ratio": float(sustain_ratio),
                    "onset_sharpness": float(onset_sharpness),
                    "articulation": art,
                }
            )
        return results
