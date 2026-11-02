"""Arousal-valence regression from audio features.

Maps spectral, rhythmic, and harmonic descriptors onto the Russell
circumplex (valence × arousal) using lightweight ridge regression and
kernel density estimation — pure numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import gaussian_kde


@dataclass
class AVPoint:
    valence: float
    arousal: float
    dominance: float = 0.0
    time: float = 0.0


@dataclass
class AVTrajectory:
    points: List[AVPoint] = field(default_factory=list)

    @property
    def valence_array(self) -> np.ndarray:
        return np.array([p.valence for p in self.points])

    @property
    def arousal_array(self) -> np.ndarray:
        return np.array([p.arousal for p in self.points])

    def mean_valence(self) -> float:
        v = self.valence_array
        return float(np.mean(v)) if len(v) else 0.0

    def mean_arousal(self) -> float:
        a = self.arousal_array
        return float(np.mean(a)) if len(a) else 0.0

    def emotional_range(self) -> float:
        if len(self.points) < 2:
            return 0.0
        v, a = self.valence_array, self.arousal_array
        return float(np.sqrt((v.max() - v.min()) ** 2 + (a.max() - a.min()) ** 2))


def spectral_features(audio: np.ndarray, sr: int, frame_len: int = 2048, hop: int = 512) -> np.ndarray:
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    feats = np.zeros((n_frames, 4))
    for i in range(n_frames):
        frame = audio[i * hop: i * hop + frame_len]
        w = np.hanning(len(frame))
        spec = np.abs(np.fft.rfft(frame * w))
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
        total = np.sum(spec) + 1e-12
        centroid = np.sum(freqs * spec) / total
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / total)
        flatness = np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12)
        rolloff_idx = np.searchsorted(np.cumsum(spec), 0.85 * np.sum(spec))
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
        feats[i] = [centroid, spread, flatness, rolloff]
    return feats


def rhythm_features(audio: np.ndarray, sr: int) -> np.ndarray:
    frame_len = 2048
    hop = 512
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * hop: i * hop + frame_len]
        energy[i] = np.sum(frame ** 2)
    onset_env = np.maximum(0, np.diff(energy, prepend=energy[0]))
    onset_env /= np.max(onset_env) + 1e-12
    acf = np.correlate(onset_env, onset_env, mode='full')
    acf = acf[len(acf) // 2:]
    acf /= acf[0] + 1e-12
    bpm_range = (60, 200)
    lo = int(60 * sr / (hop * bpm_range[1]))
    hi = int(60 * sr / (hop * bpm_range[0]))
    lo, hi = max(lo, 1), min(hi, len(acf) - 1)
    if lo >= hi:
        return np.array([120.0, 0.5, 0.5])
    peak = lo + np.argmax(acf[lo:hi])
    tempo = 60.0 * sr / (hop * peak) if peak > 0 else 120.0
    pulse_strength = float(acf[peak])
    onset_density = float(np.mean(onset_env > 0.3))
    return np.array([tempo, pulse_strength, onset_density])


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    spec = spectral_features(audio, sr)
    spec_mean = np.mean(spec, axis=0)
    spec_std = np.std(spec, axis=0)
    rhy = rhythm_features(audio, sr)
    rms = np.sqrt(np.mean(audio ** 2))
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0
    return np.concatenate([spec_mean, spec_std, rhy, [rms, zcr]])


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    n_feat = X.shape[1]
    A = X.T @ X + alpha * np.eye(n_feat)
    b = X.T @ y
    return np.linalg.solve(A, b)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


@dataclass
class ArousalValenceModel:
    weights_v: Optional[np.ndarray] = None
    weights_a: Optional[np.ndarray] = None
    feature_mean: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None
    alpha: float = 1.0

    def fit(self, features: np.ndarray, valence: np.ndarray, arousal: np.ndarray) -> None:
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0) + 1e-8
        X = (features - self.feature_mean) / self.feature_std
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([X, ones])
        self.weights_v = ridge_fit(X, valence, self.alpha)
        self.weights_a = ridge_fit(X, arousal, self.alpha)

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        if self.weights_v is None:
            return 0.0, 0.0
        X = (features - self.feature_mean) / self.feature_std
        X = np.append(X, 1.0)
        v = float(np.clip(X @ self.weights_v, -1, 1))
        a = float(np.clip(X @ self.weights_a, -1, 1))
        return v, a


def av_trajectory(audio: np.ndarray, sr: int, model: ArousalValenceModel,
                  window_sec: float = 3.0, hop_sec: float = 1.0) -> AVTrajectory:
    window = int(sr * window_sec)
    hop = int(sr * hop_sec)
    traj = AVTrajectory()
    for i in range(0, len(audio) - window + 1, hop):
        chunk = audio[i:i + window]
        feats = extract_features(chunk, sr)
        v, a = model.predict(feats)
        traj.points.append(AVPoint(valence=v, arousal=a, time=i / sr))
    return traj


def emotion_density(points: List[AVPoint], bw_method: float = 0.3) -> Optional[gaussian_kde]:
    if len(points) < 3:
        return None
    coords = np.array([[p.valence, p.arousal] for p in points])
    return gaussian_kde(coords.T, bw_method=bw_method)


def quadrant_distribution(points: List[AVPoint]) -> dict:
    labels = {
        "happy": (1, 1), "angry": (-1, 1),
        "sad": (-1, -1), "calm": (1, -1),
    }
    counts = {k: 0 for k in labels}
    for p in points:
        if p.valence >= 0 and p.arousal >= 0:
            counts["happy"] += 1
        elif p.valence < 0 and p.arousal >= 0:
            counts["angry"] += 1
        elif p.valence < 0 and p.arousal < 0:
            counts["sad"] += 1
        else:
            counts["calm"] += 1
    total = len(points) or 1
    return {k: v / total for k, v in counts.items()}


def emotional_arc_similarity(traj_a: AVTrajectory, traj_b: AVTrajectory) -> float:
    if not traj_a.points or not traj_b.points:
        return 0.0
    n = max(len(traj_a.points), len(traj_b.points))
    va = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(traj_a.points)), traj_a.valence_array)
    aa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(traj_a.points)), traj_a.arousal_array)
    vb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(traj_b.points)), traj_b.valence_array)
    ab = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(traj_b.points)), traj_b.arousal_array)
    dist = np.sqrt(np.mean((va - vb) ** 2 + (aa - ab) ** 2))
    return float(np.exp(-dist))
