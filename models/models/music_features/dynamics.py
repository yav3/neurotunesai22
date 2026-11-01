from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DynamicsFeatures:
    rms_mean: float = 0.0
    rms_std: float = 0.0
    rms_max: float = 0.0
    dynamic_range_db: float = 0.0
    loudness_lufs_approx: float = -70.0
    crest_factor: float = 0.0
    attack_time_mean: float = 0.0
    release_time_mean: float = 0.0
    dynamic_complexity: float = 0.0
    loudness_curve: list[float] = field(default_factory=list)


class DynamicsExtractor:
    def __init__(self, frame_length: int = 2048, hop_length: int = 512):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray, sr: int = 22050) -> DynamicsFeatures:
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float64)

        if len(audio) == 0:
            return DynamicsFeatures()

        rms_env = self._rms_envelope(audio)
        if len(rms_env) == 0:
            rms_env = np.array([np.sqrt(np.mean(audio ** 2))])

        rms_mean = float(np.mean(rms_env))
        rms_std = float(np.std(rms_env))
        rms_max = float(np.max(rms_env))
        dynamic_range = self._dynamic_range(rms_env)
        loudness = self._loudness_approximation(audio, sr)
        crest = self._crest_factor(audio, rms_mean)

        attacks = self._detect_attacks(rms_env)
        attack_time, release_time = self._attack_release_times(rms_env, attacks, sr)

        dynamic_complexity = self._dynamic_complexity(rms_env)
        loudness_curve = self._downsample_curve(rms_env, target_len=32)

        return DynamicsFeatures(
            rms_mean=rms_mean,
            rms_std=rms_std,
            rms_max=rms_max,
            dynamic_range_db=dynamic_range,
            loudness_lufs_approx=loudness,
            crest_factor=crest,
            attack_time_mean=attack_time,
            release_time_mean=release_time,
            dynamic_complexity=dynamic_complexity,
            loudness_curve=loudness_curve,
        )

    def _rms_envelope(self, audio: np.ndarray) -> np.ndarray:
        n_frames = 1 + (len(audio) - self.frame_length) // self.hop_length
        if n_frames < 1:
            return np.array([np.sqrt(np.mean(audio ** 2))])
        rms = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_length]
            rms[i] = np.sqrt(np.mean(frame ** 2))
        return rms

    def _dynamic_range(self, rms_env: np.ndarray) -> float:
        if len(rms_env) < 2:
            return 0.0
        p95 = np.percentile(rms_env, 95)
        p5 = np.percentile(rms_env, 5)
        if p5 <= 0:
            p5 = 1e-10
        if p95 <= 0:
            return 0.0
        return float(20.0 * np.log10(p95 / p5))

    def _loudness_approximation(self, audio: np.ndarray, sr: int) -> float:
        if len(audio) == 0:
            return -70.0
        mean_sq = np.mean(audio ** 2)
        if mean_sq <= 0:
            return -70.0
        lufs_approx = -0.691 + 10.0 * np.log10(mean_sq + 1e-10)
        return float(lufs_approx)

    def _crest_factor(self, audio: np.ndarray, rms: float) -> float:
        if rms <= 0:
            return 0.0
        peak = np.max(np.abs(audio))
        return float(peak / rms)

    def _detect_attacks(self, rms_env: np.ndarray) -> list[int]:
        if len(rms_env) < 3:
            return []
        diff = np.diff(rms_env)
        threshold = np.std(diff) * 1.5 if np.std(diff) > 0 else 0.01
        attacks = []
        for i in range(len(diff)):
            if diff[i] > threshold:
                if not attacks or i - attacks[-1] > 3:
                    attacks.append(i)
        return attacks

    def _attack_release_times(
        self, rms_env: np.ndarray, attacks: list[int], sr: int,
    ) -> tuple[float, float]:
        if not attacks or len(rms_env) < 3:
            return 0.0, 0.0
        frame_duration = self.hop_length / sr
        attack_times = []
        release_times = []

        for att in attacks:
            peak_idx = att
            for j in range(att, min(att + 20, len(rms_env))):
                if rms_env[j] >= rms_env[peak_idx]:
                    peak_idx = j
                else:
                    break
            attack_frames = peak_idx - att
            attack_times.append(attack_frames * frame_duration)

            release_end = peak_idx
            threshold = rms_env[peak_idx] * 0.1
            for j in range(peak_idx, min(peak_idx + 50, len(rms_env))):
                if rms_env[j] < threshold:
                    release_end = j
                    break
            release_frames = release_end - peak_idx
            release_times.append(release_frames * frame_duration)

        return float(np.mean(attack_times)), float(np.mean(release_times))

    def _dynamic_complexity(self, rms_env: np.ndarray) -> float:
        if len(rms_env) < 4:
            return 0.0
        diff = np.abs(np.diff(rms_env))
        mean_rms = np.mean(rms_env)
        if mean_rms <= 0:
            return 0.0
        return float(np.mean(diff) / mean_rms)

    def _downsample_curve(self, rms_env: np.ndarray, target_len: int = 32) -> list[float]:
        if len(rms_env) <= target_len:
            return [float(x) for x in rms_env]
        indices = np.linspace(0, len(rms_env) - 1, target_len, dtype=int)
        return [float(rms_env[i]) for i in indices]
