from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RhythmFeatures:
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    beat_positions: list[float] = field(default_factory=list)
    downbeat_positions: list[float] = field(default_factory=list)
    onset_rate: float = 0.0
    onset_strength_mean: float = 0.0
    onset_strength_std: float = 0.0
    tempo_stability: float = 0.0
    beat_regularity: float = 0.0
    swing_ratio: float = 0.0
    groove_consistency: float = 0.0


class RhythmExtractor:
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> RhythmFeatures:
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float64)

        if len(audio) < self.hop_length * 4:
            return RhythmFeatures()

        onset_env = self._onset_strength_envelope(audio)
        if onset_env.max() < 1e-10:
            duration = len(audio) / self.sr
            return RhythmFeatures(onset_rate=0.0)

        bpm, bpm_confidence = self._autocorrelation_tempo(onset_env)
        beat_positions = self._beat_tracking(onset_env, bpm)
        onset_rate = self._onset_rate(onset_env)
        onset_strength_mean = float(np.mean(onset_env))
        onset_strength_std = float(np.std(onset_env))
        tempo_stability = self._tempo_stability(beat_positions)
        beat_regularity = self._beat_regularity(beat_positions)
        swing_ratio = self._swing_ratio(beat_positions)
        groove_consistency = self._groove_consistency(onset_env, beat_positions)
        downbeat_positions = self._estimate_downbeats(beat_positions, bpm)

        return RhythmFeatures(
            bpm=bpm,
            bpm_confidence=bpm_confidence,
            beat_positions=beat_positions,
            downbeat_positions=downbeat_positions,
            onset_rate=onset_rate,
            onset_strength_mean=onset_strength_mean,
            onset_strength_std=onset_strength_std,
            tempo_stability=tempo_stability,
            beat_regularity=beat_regularity,
            swing_ratio=swing_ratio,
            groove_consistency=groove_consistency,
        )

    def _onset_strength_envelope(self, audio: np.ndarray) -> np.ndarray:
        n_fft = 2048
        n_frames = 1 + (len(audio) - n_fft) // self.hop_length
        if n_frames < 2:
            return np.zeros(1)
        window = np.hanning(n_fft)
        frames = np.stack([
            audio[i * self.hop_length: i * self.hop_length + n_fft]
            for i in range(n_frames)
        ])
        spec = np.abs(np.fft.rfft(frames * window, axis=1))
        n_mels = 40
        mel_spec = self._simple_mel_reduce(spec, n_mels)
        log_mel = np.log1p(mel_spec * 1000.0)
        diff = np.diff(log_mel, axis=0)
        onset_env = np.maximum(0.0, diff.sum(axis=1))
        onset_env = np.concatenate([[0.0], onset_env])
        mx = onset_env.max()
        if mx > 0:
            onset_env = onset_env / mx
        return onset_env

    def _simple_mel_reduce(self, spec: np.ndarray, n_mels: int) -> np.ndarray:
        n_freqs = spec.shape[1]
        indices = np.linspace(0, n_freqs, n_mels + 1, dtype=int)
        mel = np.zeros((spec.shape[0], n_mels))
        for m in range(n_mels):
            lo, hi = indices[m], indices[m + 1]
            if hi > lo:
                mel[:, m] = spec[:, lo:hi].mean(axis=1)
        return mel

    def _autocorrelation_tempo(self, onset_env: np.ndarray) -> tuple[float, float]:
        if len(onset_env) < 8:
            return 120.0, 0.0
        env = onset_env - np.mean(onset_env)
        corr = np.correlate(env, env, mode="full")
        corr = corr[len(corr) // 2:]
        if corr[0] > 0:
            corr = corr / corr[0]

        frame_rate = self.sr / self.hop_length
        bpm_min, bpm_max = 40.0, 240.0
        lag_min = max(1, int(frame_rate * 60.0 / bpm_max))
        lag_max = min(len(corr) - 1, int(frame_rate * 60.0 / bpm_min))

        if lag_min >= lag_max:
            return 120.0, 0.0

        search = corr[lag_min:lag_max + 1]
        if len(search) == 0:
            return 120.0, 0.0
        best_lag = lag_min + np.argmax(search)
        confidence = float(search.max())

        half_lag = best_lag // 2
        if half_lag >= lag_min:
            half_val = corr[half_lag]
            if half_val >= confidence * 0.8:
                best_lag = half_lag
                confidence = float(half_val)

        bpm = 60.0 * frame_rate / best_lag

        return float(bpm), float(np.clip(confidence, 0.0, 1.0))

    def _beat_tracking(self, onset_env: np.ndarray, bpm: float) -> list[float]:
        if bpm <= 0 or len(onset_env) < 4:
            return []
        frame_rate = self.sr / self.hop_length
        beat_period = frame_rate * 60.0 / bpm

        threshold = 0.3
        peaks = []
        for i in range(1, len(onset_env) - 1):
            if onset_env[i] > onset_env[i - 1] and onset_env[i] > onset_env[i + 1]:
                if onset_env[i] > threshold:
                    peaks.append(i)

        if not peaks:
            n_beats = int(len(onset_env) / beat_period)
            return [float(i * beat_period / frame_rate) for i in range(n_beats)]

        beats = [peaks[0]]
        for p in peaks[1:]:
            gap = p - beats[-1]
            if gap >= beat_period * 0.5:
                beats.append(p)

        return [float(b / frame_rate) for b in beats]

    def _onset_rate(self, onset_env: np.ndarray) -> float:
        threshold = 0.3
        peaks = 0
        for i in range(1, len(onset_env) - 1):
            if onset_env[i] > onset_env[i - 1] and onset_env[i] > onset_env[i + 1]:
                if onset_env[i] > threshold:
                    peaks += 1
        duration = len(onset_env) * self.hop_length / self.sr
        if duration <= 0:
            return 0.0
        return float(peaks / duration)

    def _tempo_stability(self, beats: list[float]) -> float:
        if len(beats) < 3:
            return 0.0
        ibis = np.diff(beats)
        if ibis.mean() <= 0:
            return 0.0
        cv = float(np.std(ibis) / np.mean(ibis))
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def _beat_regularity(self, beats: list[float]) -> float:
        if len(beats) < 3:
            return 0.0
        ibis = np.diff(beats)
        median_ibi = np.median(ibis)
        if median_ibi <= 0:
            return 0.0
        deviations = np.abs(ibis - median_ibi) / median_ibi
        return float(np.clip(1.0 - np.mean(deviations), 0.0, 1.0))

    def _swing_ratio(self, beats: list[float]) -> float:
        if len(beats) < 4:
            return 0.5
        ibis = np.diff(beats)
        if len(ibis) < 2:
            return 0.5
        even_ibis = ibis[0::2]
        odd_ibis = ibis[1::2]
        n = min(len(even_ibis), len(odd_ibis))
        if n == 0:
            return 0.5
        total = even_ibis[:n] + odd_ibis[:n]
        total = np.where(total == 0, 1.0, total)
        ratios = even_ibis[:n] / total
        return float(np.mean(ratios))

    def _groove_consistency(self, onset_env: np.ndarray, beats: list[float]) -> float:
        if len(beats) < 2:
            return 0.0
        frame_rate = self.sr / self.hop_length
        beat_frames = [int(b * frame_rate) for b in beats]
        strengths = []
        for bf in beat_frames:
            if 0 <= bf < len(onset_env):
                strengths.append(onset_env[bf])
        if not strengths:
            return 0.0
        mean_s = np.mean(strengths)
        if mean_s <= 0:
            return 0.0
        return float(np.clip(1.0 - np.std(strengths) / mean_s, 0.0, 1.0))

    def _estimate_downbeats(self, beats: list[float], bpm: float) -> list[float]:
        if len(beats) < 4:
            return beats[:1] if beats else []
        return beats[::4]
