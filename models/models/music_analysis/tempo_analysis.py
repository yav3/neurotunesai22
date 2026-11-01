"""Tempo analysis: multi-resolution estimation, tempo curves, beat strength."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TempoEstimate:
    bpm: float
    confidence: float
    octave_error_risk: float


def onset_strength_envelope(
    signal: np.ndarray, sr: int = 22050, hop: int = 512, n_fft: int = 2048,
) -> np.ndarray:
    from numpy.fft import rfft
    n = len(signal)
    n_frames = max(1, 1 + (n - n_fft) // hop)
    window = np.hanning(n_fft)
    prev_mag = None
    odf = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = np.zeros(n_fft)
        end = min(start + n_fft, n)
        frame[: end - start] = signal[start:end] * window[: end - start]
        mag = np.abs(rfft(frame))
        if prev_mag is not None:
            diff = mag - prev_mag
            odf[i] = float(np.sum(np.maximum(diff, 0.0)))
        prev_mag = mag
    return odf


def autocorrelation_tempo(
    odf: np.ndarray,
    sr: int = 22050,
    hop: int = 512,
    bpm_min: float = 40.0,
    bpm_max: float = 240.0,
) -> TempoEstimate:
    if len(odf) < 4:
        return TempoEstimate(bpm=120.0, confidence=0.0, octave_error_risk=1.0)
    odf_centered = odf - odf.mean()
    n = len(odf_centered)
    acf = np.correlate(odf_centered, odf_centered, mode="full")[n - 1 :]
    if acf[0] > 0:
        acf = acf / acf[0]

    fps = sr / hop
    lag_min = max(1, int(60.0 / bpm_max * fps))
    lag_max = min(n - 1, int(60.0 / bpm_min * fps))
    if lag_min >= lag_max:
        return TempoEstimate(bpm=120.0, confidence=0.0, octave_error_risk=1.0)

    search = acf[lag_min : lag_max + 1]
    best_idx = int(np.argmax(search))
    best_lag = lag_min + best_idx
    bpm = 60.0 * fps / best_lag
    confidence = float(search[best_idx])

    half_lag = best_lag // 2
    double_lag = best_lag * 2
    octave_risk = 0.0
    if lag_min <= half_lag < lag_max and half_lag < len(acf):
        octave_risk = max(octave_risk, float(acf[half_lag]))
    if lag_min <= double_lag < lag_max and double_lag < len(acf):
        octave_risk = max(octave_risk, float(acf[double_lag]))

    return TempoEstimate(bpm=bpm, confidence=confidence, octave_error_risk=octave_risk)


def tempo_curve(
    signal: np.ndarray,
    sr: int = 22050,
    hop: int = 512,
    window_sec: float = 8.0,
    step_sec: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    odf = onset_strength_envelope(signal, sr=sr, hop=hop)
    fps = sr / hop
    win_frames = max(4, int(window_sec * fps))
    step_frames = max(1, int(step_sec * fps))

    tempos = []
    times = []
    pos = 0
    while pos + win_frames <= len(odf):
        chunk = odf[pos : pos + win_frames]
        est = autocorrelation_tempo(chunk, sr=sr, hop=hop)
        center_time = (pos + win_frames / 2) / fps
        tempos.append(est.bpm)
        times.append(center_time)
        pos += step_frames
    return np.array(times), np.array(tempos)


def beat_strength(odf: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    strengths = np.zeros(len(beat_frames), dtype=np.float64)
    mx = odf.max() if len(odf) > 0 else 1.0
    if mx < 1e-12:
        return strengths
    for i, bf in enumerate(beat_frames):
        idx = int(bf)
        if 0 <= idx < len(odf):
            strengths[i] = odf[idx] / mx
    return strengths


def tempo_stability(tempos: np.ndarray) -> float:
    if len(tempos) < 2:
        return 1.0
    cv = float(np.std(tempos) / max(np.mean(tempos), 1e-12))
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def pulse_clarity(odf: np.ndarray, sr: int = 22050, hop: int = 512) -> float:
    if len(odf) < 8:
        return 0.0
    odf_c = odf - odf.mean()
    acf = np.correlate(odf_c, odf_c, mode="full")[len(odf_c) - 1 :]
    if acf[0] < 1e-12:
        return 0.0
    acf = acf / acf[0]
    fps = sr / hop
    lag_min = max(1, int(60.0 / 240.0 * fps))
    lag_max = min(len(acf) - 1, int(60.0 / 40.0 * fps))
    if lag_min >= lag_max:
        return 0.0
    return float(np.max(acf[lag_min : lag_max + 1]))
