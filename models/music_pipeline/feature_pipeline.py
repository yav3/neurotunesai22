"""End-to-end feature extraction pipeline.

Orchestrates feature extraction from raw audio signals through spectral,
rhythmic, harmonic, timbral, and emotional dimensions into a unified
feature vector suitable for downstream ML tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PipelineConfig:
    sr: int = 22050
    frame_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 13
    chroma_bins: int = 12
    enable_spectral: bool = True
    enable_rhythm: bool = True
    enable_harmonic: bool = True
    enable_timbral: bool = True
    enable_emotion: bool = True


@dataclass
class FeatureResult:
    spectral: dict[str, Any] = field(default_factory=dict)
    rhythm: dict[str, Any] = field(default_factory=dict)
    harmonic: dict[str, Any] = field(default_factory=dict)
    timbral: dict[str, Any] = field(default_factory=dict)
    emotion: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        parts = []
        for group in [self.spectral, self.rhythm, self.harmonic, self.timbral, self.emotion]:
            for v in group.values():
                if isinstance(v, (int, float)):
                    parts.append(float(v))
                elif isinstance(v, np.ndarray) and v.ndim == 0:
                    parts.append(float(v))
        return np.array(parts, dtype=np.float64)

    def dimension_names(self) -> list[str]:
        names = []
        for prefix, group in [
            ("spec", self.spectral), ("rhy", self.rhythm),
            ("har", self.harmonic), ("tim", self.timbral), ("emo", self.emotion),
        ]:
            for k, v in group.items():
                if isinstance(v, (int, float)) or (isinstance(v, np.ndarray) and v.ndim == 0):
                    names.append(f"{prefix}_{k}")
        return names


def extract_spectral_features(signal: np.ndarray, sr: int, config: PipelineConfig) -> dict[str, float]:
    n = len(signal)
    if n < config.frame_length:
        return {"centroid": 0.0, "rolloff": 0.0, "bandwidth": 0.0, "flatness": 0.0, "zcr": 0.0}

    fft = np.fft.rfft(signal[:config.frame_length])
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(config.frame_length, 1.0 / sr)

    total = mag.sum()
    if total < 1e-12:
        return {"centroid": 0.0, "rolloff": 0.0, "bandwidth": 0.0, "flatness": 0.0, "zcr": 0.0}

    centroid = float(np.sum(freqs * mag) / total)

    cumsum = np.cumsum(mag)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    bandwidth = float(np.sqrt(np.sum(mag * (freqs - centroid) ** 2) / total))

    log_mag = np.log(mag + 1e-12)
    flatness = float(np.exp(log_mag.mean()) / (mag.mean() + 1e-12))

    sign_changes = np.sum(np.diff(np.sign(signal)) != 0)
    zcr = float(sign_changes / (2 * n))

    return {"centroid": centroid, "rolloff": rolloff, "bandwidth": bandwidth, "flatness": flatness, "zcr": zcr}


def extract_rhythm_features(signal: np.ndarray, sr: int, config: PipelineConfig) -> dict[str, float]:
    hop = config.hop_length
    n_frames = max(1, len(signal) // hop)

    energy = np.array([
        np.sum(signal[i * hop:(i + 1) * hop] ** 2)
        for i in range(n_frames)
    ])

    if len(energy) < 3:
        return {"tempo_estimate": 0.0, "beat_strength": 0.0, "onset_rate": 0.0, "pulse_clarity": 0.0}

    onset = np.diff(energy)
    onset = np.maximum(onset, 0)

    if onset.max() < 1e-12:
        return {"tempo_estimate": 0.0, "beat_strength": 0.0, "onset_rate": 0.0, "pulse_clarity": 0.0}

    acf = np.correlate(onset, onset, mode="full")
    acf = acf[len(acf) // 2:]
    if len(acf) > 1:
        acf[0] = 0

    frame_rate = sr / hop
    min_lag = max(1, int(frame_rate * 60 / 240))
    max_lag = min(len(acf), int(frame_rate * 60 / 40))

    if max_lag <= min_lag:
        return {"tempo_estimate": 0.0, "beat_strength": 0.0, "onset_rate": 0.0, "pulse_clarity": 0.0}

    search = acf[min_lag:max_lag]
    best_lag = int(np.argmax(search)) + min_lag
    tempo = float(60.0 * frame_rate / best_lag) if best_lag > 0 else 0.0

    beat_strength = float(search.max() / (acf.max() + 1e-12)) if acf.max() > 0 else 0.0

    threshold = onset.mean() + onset.std()
    onset_count = int(np.sum(onset > threshold))
    duration_s = len(signal) / sr
    onset_rate = float(onset_count / duration_s) if duration_s > 0 else 0.0

    pulse_clarity = float(np.std(energy) / (np.mean(energy) + 1e-12))

    return {
        "tempo_estimate": tempo,
        "beat_strength": beat_strength,
        "onset_rate": onset_rate,
        "pulse_clarity": pulse_clarity,
    }


def extract_harmonic_features(signal: np.ndarray, sr: int, config: PipelineConfig) -> dict[str, float]:
    n = len(signal)
    if n < config.frame_length:
        return {f"chroma_{i}": 0.0 for i in range(config.chroma_bins)}

    fft = np.fft.rfft(signal[:config.frame_length])
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(config.frame_length, 1.0 / sr)

    chroma = np.zeros(config.chroma_bins)
    for i, f in enumerate(freqs):
        if f < 20 or f > 8000:
            continue
        semitone = 12 * np.log2(f / 440.0 + 1e-12)
        bin_idx = int(round(semitone)) % config.chroma_bins
        chroma[bin_idx] += mag[i]

    total = chroma.sum()
    if total > 0:
        chroma /= total

    result = {f"chroma_{i}": float(chroma[i]) for i in range(config.chroma_bins)}

    peak_bin = int(np.argmax(chroma))
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    a_bin = 9
    result["estimated_key_idx"] = float((peak_bin - a_bin) % 12)
    result["key_confidence"] = float(chroma[peak_bin])

    harmonic_energy = float(np.sum(chroma[chroma > chroma.mean()]))
    result["harmonic_concentration"] = harmonic_energy

    return result


def extract_timbral_features(signal: np.ndarray, sr: int, config: PipelineConfig) -> dict[str, float]:
    n = len(signal)
    if n < config.frame_length:
        return {"brightness": 0.0, "roughness": 0.0, "warmth": 0.0, "sharpness": 0.0}

    fft = np.fft.rfft(signal[:config.frame_length])
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(config.frame_length, 1.0 / sr)
    total = mag.sum()
    if total < 1e-12:
        return {"brightness": 0.0, "roughness": 0.0, "warmth": 0.0, "sharpness": 0.0}

    high_mask = freqs > 3000
    brightness = float(mag[high_mask].sum() / total) if high_mask.any() else 0.0

    low_mask = freqs < 500
    warmth = float(mag[low_mask].sum() / total) if low_mask.any() else 0.0

    sharp_mask = (freqs > 2000) & (freqs < 6000)
    sharpness = float(mag[sharp_mask].sum() / total) if sharp_mask.any() else 0.0

    if len(mag) > 1:
        roughness = float(np.mean(np.abs(np.diff(mag))))
    else:
        roughness = 0.0

    return {"brightness": brightness, "roughness": roughness, "warmth": warmth, "sharpness": sharpness}


def extract_emotion_features(
    spectral: dict[str, float],
    rhythm: dict[str, float],
    timbral: dict[str, float],
) -> dict[str, float]:
    tempo = rhythm.get("tempo_estimate", 120.0)
    t_norm = np.clip((tempo - 40) / 200, 0, 1)

    brightness = timbral.get("brightness", 0.5)
    energy = rhythm.get("beat_strength", 0.5)
    dance = rhythm.get("pulse_clarity", 0.5)

    mode_proxy = 1.0 if spectral.get("key_confidence", 0) > 0.15 else 0.0

    valence = float(0.45 * mode_proxy + 0.25 * (1 - abs(t_norm - 0.5) * 2)
                     + 0.20 * brightness + 0.10 * dance)
    arousal = float(0.50 * t_norm + 0.30 * energy + 0.20 * brightness)
    dominance = float(0.60 * (1 - abs(dance - 0.6)) + 0.20 * energy + 0.20 * (1 - abs(t_norm - 0.5)))

    return {"valence": valence, "arousal": arousal, "dominance": dominance}


class FeaturePipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def extract(self, signal: np.ndarray, sr: int | None = None) -> FeatureResult:
        sr = sr or self.config.sr
        result = FeatureResult()
        result.metadata["n_samples"] = len(signal)
        result.metadata["sr"] = sr
        result.metadata["duration_s"] = len(signal) / sr

        if self.config.enable_spectral:
            result.spectral = extract_spectral_features(signal, sr, self.config)

        if self.config.enable_rhythm:
            result.rhythm = extract_rhythm_features(signal, sr, self.config)

        if self.config.enable_harmonic:
            result.harmonic = extract_harmonic_features(signal, sr, self.config)

        if self.config.enable_timbral:
            result.timbral = extract_timbral_features(signal, sr, self.config)

        if self.config.enable_emotion:
            result.emotion = extract_emotion_features(result.spectral, result.rhythm, result.timbral)

        return result

    def extract_batch(self, signals: list[np.ndarray], sr: int | None = None) -> list[FeatureResult]:
        return [self.extract(s, sr) for s in signals]

    def feature_matrix(self, signals: list[np.ndarray], sr: int | None = None) -> np.ndarray:
        results = self.extract_batch(signals, sr)
        vectors = [r.to_vector() for r in results]
        if not vectors:
            return np.empty((0, 0))
        max_len = max(len(v) for v in vectors)
        padded = []
        for v in vectors:
            if len(v) < max_len:
                v = np.pad(v, (0, max_len - len(v)))
            padded.append(v)
        return np.vstack(padded)
