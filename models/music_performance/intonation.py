"""Intonation analysis: pitch tracking, temperament comparison, microtone detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

EQUAL_TEMPERAMENT_CENTS = np.arange(12) * 100.0

JUST_INTONATION_RATIOS = np.array([
    1.0, 16 / 15, 9 / 8, 6 / 5, 5 / 4, 4 / 3,
    45 / 32, 3 / 2, 8 / 5, 5 / 3, 9 / 5, 15 / 8,
])
JUST_INTONATION_CENTS = 1200.0 * np.log2(JUST_INTONATION_RATIOS)

PYTHAGOREAN_RATIOS = np.array([
    1.0, 256 / 243, 9 / 8, 32 / 27, 81 / 64, 4 / 3,
    729 / 512, 3 / 2, 128 / 81, 27 / 16, 16 / 9, 243 / 128,
])
PYTHAGOREAN_CENTS = 1200.0 * np.log2(PYTHAGOREAN_RATIOS)


class PitchTracker:
    @staticmethod
    def autocorrelation_pitch(
        frame: np.ndarray, sr: int = 44100, min_hz: float = 50.0, max_hz: float = 2000.0
    ) -> float:
        if len(frame) < int(sr / min_hz):
            return 0.0
        mono = frame.astype(np.float64)
        mono = mono - np.mean(mono)
        if np.max(np.abs(mono)) < 1e-10:
            return 0.0

        min_lag = max(1, int(sr / max_hz))
        max_lag = min(len(mono) - 1, int(sr / min_hz))
        if min_lag >= max_lag:
            return 0.0

        corr = np.correlate(mono, mono, mode="full")
        corr = corr[len(corr) // 2 :]
        corr_norm = corr / (corr[0] + 1e-12)
        search = corr_norm[min_lag : max_lag + 1]

        if len(search) == 0:
            return 0.0
        peak = int(np.argmax(search))
        if search[peak] < 0.3:
            return 0.0
        lag = min_lag + peak
        return float(sr / lag) if lag > 0 else 0.0

    @staticmethod
    def track(
        signal: np.ndarray,
        sr: int = 44100,
        hop_ms: float = 10.0,
        frame_ms: float = 40.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        mono = signal if signal.ndim == 1 else signal.mean(axis=1)
        hop = max(1, int(sr * hop_ms / 1000.0))
        frame_len = max(1, int(sr * frame_ms / 1000.0))
        n_frames = max(0, (len(mono) - frame_len) // hop + 1)
        times = np.arange(n_frames) * hop / sr
        pitches = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            frame = mono[start : start + frame_len]
            pitches[i] = PitchTracker.autocorrelation_pitch(frame, sr)
        return times, pitches


class IntonationAnalyzer:
    @staticmethod
    def hz_to_cents_from_ref(freq_hz: float, ref_hz: float = 440.0) -> float:
        if freq_hz <= 0 or ref_hz <= 0:
            return 0.0
        return float(1200.0 * np.log2(freq_hz / ref_hz))

    @staticmethod
    def nearest_equal_temperament(freq_hz: float, a4: float = 440.0) -> dict:
        if freq_hz <= 0:
            return {"note": "?", "midi": 0, "deviation_cents": 0.0}
        midi = 69 + 12 * np.log2(freq_hz / a4)
        midi_round = int(round(midi))
        deviation = (midi - midi_round) * 100.0
        note_name = PITCH_NAMES[midi_round % 12]
        octave = midi_round // 12 - 1
        return {
            "note": f"{note_name}{octave}",
            "midi": midi_round,
            "deviation_cents": float(deviation),
        }

    @staticmethod
    def deviation_from_temperament(
        freq_hz: float,
        temperament: str = "equal",
        a4: float = 440.0,
    ) -> float:
        if freq_hz <= 0:
            return 0.0
        cents_from_a4 = 1200.0 * np.log2(freq_hz / a4)
        semitones_from_c = (cents_from_a4 + 900) % 1200
        pitch_class = int(round(semitones_from_c / 100.0)) % 12

        if temperament == "just":
            target_cents = JUST_INTONATION_CENTS[pitch_class]
        elif temperament == "pythagorean":
            target_cents = PYTHAGOREAN_CENTS[pitch_class]
        else:
            target_cents = EQUAL_TEMPERAMENT_CENTS[pitch_class]

        et_cents = pitch_class * 100.0
        deviation_from_et = semitones_from_c - et_cents
        target_offset = target_cents - et_cents
        return float(deviation_from_et - target_offset)

    @staticmethod
    def analyze_pitches(
        pitch_hz: np.ndarray, temperament: str = "equal", a4: float = 440.0
    ) -> dict:
        valid = pitch_hz[pitch_hz > 0]
        if len(valid) == 0:
            return {"mean_deviation": 0.0, "std_deviation": 0.0, "max_deviation": 0.0, "n_notes": 0}
        deviations = np.array(
            [IntonationAnalyzer.deviation_from_temperament(f, temperament, a4) for f in valid]
        )
        return {
            "mean_deviation": float(np.mean(np.abs(deviations))),
            "std_deviation": float(np.std(deviations)),
            "max_deviation": float(np.max(np.abs(deviations))),
            "n_notes": len(valid),
        }


class TemperamentComparator:
    @staticmethod
    def compare(
        temperament_a: str = "equal", temperament_b: str = "just"
    ) -> list[dict]:
        def get_cents(name: str) -> np.ndarray:
            if name == "just":
                return JUST_INTONATION_CENTS
            elif name == "pythagorean":
                return PYTHAGOREAN_CENTS
            return EQUAL_TEMPERAMENT_CENTS

        a = get_cents(temperament_a)
        b = get_cents(temperament_b)
        results: list[dict] = []
        for i in range(12):
            results.append(
                {
                    "pitch_class": PITCH_NAMES[i],
                    f"{temperament_a}_cents": float(a[i]),
                    f"{temperament_b}_cents": float(b[i]),
                    "difference_cents": float(b[i] - a[i]),
                }
            )
        return results

    @staticmethod
    def max_difference(temperament_a: str = "equal", temperament_b: str = "just") -> float:
        rows = TemperamentComparator.compare(temperament_a, temperament_b)
        if not rows:
            return 0.0
        return max(abs(r["difference_cents"]) for r in rows)


class MicrotoneDetector:
    @staticmethod
    def detect_quarter_tones(
        pitch_hz: np.ndarray, a4: float = 440.0, tolerance_cents: float = 15.0
    ) -> list[dict]:
        valid = pitch_hz[pitch_hz > 0]
        if len(valid) == 0:
            return []

        results: list[dict] = []
        for i, freq in enumerate(valid):
            info = IntonationAnalyzer.nearest_equal_temperament(freq, a4)
            dev = abs(info["deviation_cents"])
            if 50 - tolerance_cents <= dev <= 50 + tolerance_cents:
                results.append(
                    {
                        "index": i,
                        "freq_hz": float(freq),
                        "nearest_note": info["note"],
                        "deviation_cents": info["deviation_cents"],
                        "type": "quarter_tone",
                    }
                )
        return results

    @staticmethod
    def classify_microtones(
        detections: list[dict], consistency_threshold: int = 3
    ) -> dict:
        if not detections:
            return {"intentional": [], "errors": []}
        from collections import Counter

        note_counts: Counter = Counter()
        for d in detections:
            note_counts[d["nearest_note"]] += 1

        intentional = []
        errors = []
        for d in detections:
            if note_counts[d["nearest_note"]] >= consistency_threshold:
                intentional.append(d)
            else:
                errors.append(d)
        return {"intentional": intentional, "errors": errors}
