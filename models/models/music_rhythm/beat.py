"""Beat tracking: dynamic programming, phase coherence, beat strength."""
from __future__ import annotations
import numpy as np


def inter_beat_intervals(beat_times: np.ndarray) -> np.ndarray:
    """Compute inter-beat intervals from beat times."""
    if len(beat_times) < 2:
        return np.array([])
    return np.diff(beat_times)


def beat_strength(
    signal: np.ndarray,
    beat_frames: np.ndarray,
    frame_size: int = 2048,
    hop: int = 512,
) -> np.ndarray:
    """Measure spectral flux at each beat position."""
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    flux = np.zeros(n_frames)
    prev_mag = None
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        mag = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
        if prev_mag is not None:
            diff = mag - prev_mag
            flux[i] = max(0.0, float(np.sum(diff[diff > 0])))
        prev_mag = mag

    strengths = np.zeros(len(beat_frames))
    for i, bf in enumerate(beat_frames):
        idx = int(bf)
        if 0 <= idx < len(flux):
            strengths[i] = flux[idx]
    return strengths


def beat_phase_coherence(beat_times: np.ndarray) -> float:
    """Measure regularity of beat spacing via circular variance.

    Returns value in [0, 1]; 1 = perfectly regular.
    """
    ibi = inter_beat_intervals(beat_times)
    if len(ibi) < 2:
        return 1.0
    median_ibi = np.median(ibi)
    if median_ibi <= 0:
        return 0.0
    phases = 2 * np.pi * (ibi / median_ibi)
    mean_vec = np.abs(np.mean(np.exp(1j * phases)))
    return float(mean_vec)


def dynamic_programming_beats(
    onset_env: np.ndarray,
    period: float,
    alpha: float = 100.0,
) -> np.ndarray:
    """Beat tracking via Viterbi-style dynamic programming.

    Args:
        onset_env: onset strength envelope (one value per frame).
        period: expected beat period in frames.
        alpha: transition penalty weight.

    Returns:
        Array of beat frame indices.
    """
    n = len(onset_env)
    if n == 0:
        return np.array([], dtype=int)
    if period < 1:
        period = 1.0

    score = np.full(n, -np.inf)
    backptr = np.full(n, -1, dtype=int)

    score[0] = onset_env[0]
    half_period = int(period / 2)
    search_range = max(1, half_period)

    for t in range(1, n):
        lo = max(0, t - int(period) - search_range)
        hi = max(0, t - int(period) + search_range)
        hi = min(hi, t)

        if lo >= hi:
            if score[t] == -np.inf:
                score[t] = onset_env[t]
            continue

        best_prev = lo
        best_val = -np.inf
        for p in range(lo, hi):
            deviation = abs((t - p) - period)
            penalty = alpha * (deviation / period) ** 2
            val = score[p] - penalty
            if val > best_val:
                best_val = val
                best_prev = p

        candidate = best_val + onset_env[t]
        if candidate > score[t]:
            score[t] = candidate
            backptr[t] = best_prev

    # backtrace from the best endpoint
    end = int(np.argmax(score))
    beats = []
    t = end
    while t >= 0:
        beats.append(t)
        t = backptr[t]
    beats.reverse()
    return np.array(beats, dtype=int)


class BeatTracker:
    """Beat tracker combining onset detection with DP alignment."""

    def __init__(
        self,
        sr: float = 22050.0,
        hop: int = 512,
        min_bpm: float = 40.0,
        max_bpm: float = 240.0,
    ) -> None:
        self.sr = sr
        self.hop = hop
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def _onset_envelope(self, signal: np.ndarray, frame_size: int = 2048) -> np.ndarray:
        n_frames = max(1, (len(signal) - frame_size) // self.hop + 1)
        envelope = np.zeros(n_frames)
        prev_mag = None
        for i in range(n_frames):
            start = i * self.hop
            frame = signal[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            mag = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
            if prev_mag is not None:
                diff = mag - prev_mag
                envelope[i] = max(0.0, float(np.sum(diff[diff > 0])))
            prev_mag = mag
        return envelope

    def _estimate_period(self, envelope: np.ndarray) -> float:
        min_lag = max(1, int(60.0 / self.max_bpm * self.sr / self.hop))
        max_lag = min(len(envelope) - 1, int(60.0 / self.min_bpm * self.sr / self.hop))
        if max_lag <= min_lag:
            return float(min_lag)

        env = envelope - np.mean(envelope)
        norm = np.dot(env, env)
        if norm < 1e-12:
            return float(min_lag)

        best_lag = min_lag
        best_val = -1.0
        for lag in range(min_lag, max_lag):
            val = np.dot(env[:len(env) - lag], env[lag:]) / norm
            if val > best_val:
                best_val = val
                best_lag = lag
        return float(best_lag)

    def track(self, signal: np.ndarray) -> dict:
        """Track beats in signal.

        Returns dict with 'beat_frames', 'beat_times', 'bpm', 'coherence'.
        """
        envelope = self._onset_envelope(signal)
        period = self._estimate_period(envelope)
        beat_frames = dynamic_programming_beats(envelope, period)
        beat_times = beat_frames * self.hop / self.sr
        bpm = 60.0 * self.sr / (period * self.hop) if period > 0 else 0.0
        coherence = beat_phase_coherence(beat_times)
        return {
            "beat_frames": beat_frames,
            "beat_times": beat_times,
            "bpm": bpm,
            "period_frames": period,
            "coherence": coherence,
        }
