"""Ensemble analysis: synchrony, blend, balance, tuning between performers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SynchronyAnalyzer:
    @staticmethod
    def pairwise_asynchrony(
        onsets_a: np.ndarray, onsets_b: np.ndarray, tolerance_s: float = 0.1
    ) -> dict:
        if len(onsets_a) == 0 or len(onsets_b) == 0:
            return {"mean_ms": 0.0, "std_ms": 0.0, "matched_pairs": 0}
        a = onsets_a.astype(np.float64)
        b = onsets_b.astype(np.float64)
        diffs: list[float] = []
        used_b: set[int] = set()
        for t_a in a:
            distances = np.abs(b - t_a)
            best = int(np.argmin(distances))
            if distances[best] <= tolerance_s and best not in used_b:
                diffs.append(float(b[best] - t_a))
                used_b.add(best)
        if not diffs:
            return {"mean_ms": 0.0, "std_ms": 0.0, "matched_pairs": 0}
        arr = np.array(diffs) * 1000.0
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "matched_pairs": len(diffs),
        }

    @staticmethod
    def group_synchrony(
        onset_arrays: list[np.ndarray], tolerance_s: float = 0.1
    ) -> dict:
        n = len(onset_arrays)
        if n < 2:
            return {"overall_mean_ms": 0.0, "overall_std_ms": 0.0, "n_pairs": 0}
        all_diffs: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                result = SynchronyAnalyzer.pairwise_asynchrony(
                    onset_arrays[i], onset_arrays[j], tolerance_s
                )
                if result["matched_pairs"] > 0:
                    all_diffs.append(abs(result["mean_ms"]))
        if not all_diffs:
            return {"overall_mean_ms": 0.0, "overall_std_ms": 0.0, "n_pairs": 0}
        return {
            "overall_mean_ms": float(np.mean(all_diffs)),
            "overall_std_ms": float(np.std(all_diffs)),
            "n_pairs": len(all_diffs),
        }


class BlendAnalyzer:
    @staticmethod
    def spectral_envelope(
        signal: np.ndarray, sr: int = 44100, n_bands: int = 32
    ) -> np.ndarray:
        mono = signal if signal.ndim == 1 else signal.mean(axis=1)
        if len(mono) == 0:
            return np.zeros(n_bands, dtype=np.float64)
        fft_mag = np.abs(np.fft.rfft(mono.astype(np.float64)))
        n_freq = len(fft_mag)
        band_size = max(1, n_freq // n_bands)
        envelope = np.zeros(n_bands, dtype=np.float64)
        for i in range(n_bands):
            start = i * band_size
            end = min(start + band_size, n_freq)
            if start < n_freq:
                envelope[i] = float(np.mean(fft_mag[start:end] ** 2))
        return envelope

    @staticmethod
    def overlap_coefficient(env_a: np.ndarray, env_b: np.ndarray) -> float:
        if len(env_a) == 0 or len(env_b) == 0:
            return 0.0
        a = env_a.astype(np.float64)
        b = env_b.astype(np.float64)
        minimum = np.minimum(a, b)
        denom = min(np.sum(a), np.sum(b))
        if denom < 1e-12:
            return 0.0
        return float(np.sum(minimum) / denom)

    @staticmethod
    def blend_score(signals: list[np.ndarray], sr: int = 44100) -> float:
        if len(signals) < 2:
            return 1.0
        envelopes = [BlendAnalyzer.spectral_envelope(s, sr) for s in signals]
        scores: list[float] = []
        for i in range(len(envelopes)):
            for j in range(i + 1, len(envelopes)):
                scores.append(BlendAnalyzer.overlap_coefficient(envelopes[i], envelopes[j]))
        return float(np.mean(scores)) if scores else 0.0


class BalanceAnalyzer:
    @staticmethod
    def loudness_per_band(
        signal: np.ndarray, sr: int = 44100, band_edges: list[float] | None = None
    ) -> np.ndarray:
        if band_edges is None:
            band_edges = [0, 250, 500, 1000, 2000, 4000, 8000, 16000, sr / 2]
        mono = signal if signal.ndim == 1 else signal.mean(axis=1)
        if len(mono) == 0:
            return np.zeros(len(band_edges) - 1, dtype=np.float64)
        fft_mag = np.abs(np.fft.rfft(mono.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)
        n_bands = len(band_edges) - 1
        result = np.zeros(n_bands, dtype=np.float64)
        for i in range(n_bands):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            if np.any(mask):
                result[i] = float(np.sqrt(np.mean(fft_mag[mask] ** 2)))
        return result

    @staticmethod
    def detect_masking(
        signals: list[np.ndarray], sr: int = 44100, masking_threshold_db: float = 12.0
    ) -> list[dict]:
        if len(signals) < 2:
            return []
        band_profiles = [BalanceAnalyzer.loudness_per_band(s, sr) for s in signals]
        results: list[dict] = []
        for i in range(len(signals)):
            for j in range(len(signals)):
                if i == j:
                    continue
                for b in range(len(band_profiles[i])):
                    if band_profiles[i][b] < 1e-12 or band_profiles[j][b] < 1e-12:
                        continue
                    ratio_db = 20.0 * np.log10(
                        band_profiles[i][b] / max(band_profiles[j][b], 1e-12)
                    )
                    if ratio_db > masking_threshold_db:
                        results.append(
                            {
                                "masker": i,
                                "masked": j,
                                "band": b,
                                "ratio_db": float(ratio_db),
                            }
                        )
        return results


class TuningAnalyzer:
    @staticmethod
    def beat_frequency(freq_a: float, freq_b: float) -> float:
        if freq_a <= 0 or freq_b <= 0:
            return 0.0
        return abs(freq_a - freq_b)

    @staticmethod
    def unison_drift(
        pitch_a: np.ndarray, pitch_b: np.ndarray
    ) -> dict:
        valid = (pitch_a > 0) & (pitch_b > 0)
        if not np.any(valid):
            return {"mean_cents": 0.0, "std_cents": 0.0, "max_cents": 0.0, "n_frames": 0}
        a = pitch_a[valid].astype(np.float64)
        b = pitch_b[valid].astype(np.float64)
        cents_diff = 1200.0 * np.log2(a / b)
        return {
            "mean_cents": float(np.mean(np.abs(cents_diff))),
            "std_cents": float(np.std(cents_diff)),
            "max_cents": float(np.max(np.abs(cents_diff))),
            "n_frames": int(np.sum(valid)),
        }

    @staticmethod
    def drift_over_time(
        pitch_a: np.ndarray, pitch_b: np.ndarray, window: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = (pitch_a > 0) & (pitch_b > 0)
        cents_diff = np.where(
            valid,
            1200.0 * np.log2(np.maximum(pitch_a, 1.0) / np.maximum(pitch_b, 1.0)),
            0.0,
        )
        if len(cents_diff) < window:
            return np.arange(len(cents_diff), dtype=np.float64), cents_diff
        kernel = np.ones(window) / window
        smoothed = np.convolve(cents_diff, kernel, mode="same")
        return np.arange(len(smoothed), dtype=np.float64), smoothed
