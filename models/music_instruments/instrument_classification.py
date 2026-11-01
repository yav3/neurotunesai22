"""Feature extraction and classification for musical instrument recognition."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InstrumentFeatureExtractor:
    """Extract discriminative features for instrument family identification."""

    def spectral_envelope(
        self, spectrum: np.ndarray, n_coefficients: int = 20
    ) -> np.ndarray:
        if len(spectrum) < n_coefficients + 1:
            return np.zeros(n_coefficients)
        log_spec = np.log(np.abs(spectrum) + 1e-10)
        cepstrum = np.fft.irfft(log_spec)
        liftered = np.zeros_like(cepstrum)
        n = min(n_coefficients, len(cepstrum))
        liftered[:n] = cepstrum[:n]
        envelope = np.fft.rfft(liftered).real
        return envelope[:n_coefficients]

    def attack_characteristics(
        self, signal: np.ndarray, sr: int, frame_ms: float = 10.0
    ) -> dict:
        if len(signal) < 2:
            return {"attack_time": 0.0, "attack_slope": 0.0, "onset_strength": 0.0}
        frame_size = max(1, int(sr * frame_ms / 1000.0))
        n_frames = len(signal) // frame_size
        if n_frames < 2:
            return {"attack_time": 0.0, "attack_slope": 0.0, "onset_strength": 0.0}
        rms = np.array([
            np.sqrt(np.mean(signal[i * frame_size:(i + 1) * frame_size] ** 2))
            for i in range(n_frames)
        ])
        peak_frame = np.argmax(rms)
        if peak_frame == 0 or rms[peak_frame] < 1e-10:
            return {"attack_time": 0.0, "attack_slope": 0.0, "onset_strength": 0.0}
        threshold = 0.1 * rms[peak_frame]
        onset_frame = 0
        for i in range(peak_frame):
            if rms[i] >= threshold:
                onset_frame = i
                break
        attack_frames = peak_frame - onset_frame
        attack_time = attack_frames * frame_ms / 1000.0
        attack_slope = float(rms[peak_frame] / max(attack_time, 1e-6))
        onset_strength = float(rms[peak_frame] - rms[onset_frame])
        return {
            "attack_time": attack_time,
            "attack_slope": attack_slope,
            "onset_strength": onset_strength,
        }

    def steady_state_features(
        self, signal: np.ndarray, sr: int, frame_size: int = 1024
    ) -> dict:
        if len(signal) < frame_size * 3:
            return {
                "vibrato_rate": 0.0, "vibrato_depth": 0.0,
                "spectral_flux_mean": 0.0, "spectral_flux_std": 0.0,
            }
        mid_start = len(signal) // 4
        mid_end = 3 * len(signal) // 4
        segment = signal[mid_start:mid_end]
        hop = frame_size // 2
        n_frames = max(1, (len(segment) - frame_size) // hop)
        if n_frames < 2:
            return {
                "vibrato_rate": 0.0, "vibrato_depth": 0.0,
                "spectral_flux_mean": 0.0, "spectral_flux_std": 0.0,
            }
        specs = []
        for i in range(n_frames):
            start = i * hop
            frame = segment[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_size) / frame_size))
            specs.append(np.abs(np.fft.rfft(frame * w)))
        specs = np.array(specs)
        flux = np.zeros(n_frames - 1)
        for i in range(1, n_frames):
            diff = specs[i] - specs[i - 1]
            flux[i - 1] = np.sum(np.maximum(diff, 0.0))
        centroids = np.array([
            np.sum(s * np.arange(len(s))) / max(np.sum(s), 1e-10)
            for s in specs
        ])
        if len(centroids) > 4:
            centroid_ac = np.correlate(
                centroids - np.mean(centroids),
                centroids - np.mean(centroids),
                mode="full",
            )
            centroid_ac = centroid_ac[len(centroid_ac) // 2:]
            if centroid_ac[0] > 0:
                centroid_ac /= centroid_ac[0]
            peaks = []
            for i in range(1, len(centroid_ac) - 1):
                if centroid_ac[i] > centroid_ac[i - 1] and centroid_ac[i] > centroid_ac[i + 1]:
                    peaks.append(i)
            if peaks:
                vibrato_period = peaks[0] * hop / sr
                vibrato_rate = 1.0 / max(vibrato_period, 1e-6)
            else:
                vibrato_rate = 0.0
            vibrato_depth = float(np.std(centroids))
        else:
            vibrato_rate = 0.0
            vibrato_depth = 0.0
        return {
            "vibrato_rate": vibrato_rate,
            "vibrato_depth": vibrato_depth,
            "spectral_flux_mean": float(np.mean(flux)),
            "spectral_flux_std": float(np.std(flux)),
        }

    def temporal_features(
        self, signal: np.ndarray, sr: int, frame_ms: float = 10.0
    ) -> dict:
        if len(signal) < 4:
            return {"attack_ms": 0.0, "decay_ms": 0.0, "sustain_level": 0.0, "release_ms": 0.0}
        frame_size = max(1, int(sr * frame_ms / 1000.0))
        n_frames = len(signal) // frame_size
        if n_frames < 4:
            return {"attack_ms": 0.0, "decay_ms": 0.0, "sustain_level": 0.0, "release_ms": 0.0}
        rms = np.array([
            np.sqrt(np.mean(signal[i * frame_size:(i + 1) * frame_size] ** 2))
            for i in range(n_frames)
        ])
        peak_idx = np.argmax(rms)
        peak_val = rms[peak_idx]
        if peak_val < 1e-10:
            return {"attack_ms": 0.0, "decay_ms": 0.0, "sustain_level": 0.0, "release_ms": 0.0}
        attack_ms = peak_idx * frame_ms
        sustain_start = peak_idx
        for i in range(peak_idx, n_frames):
            if rms[i] < 0.5 * peak_val:
                sustain_start = i
                break
        else:
            sustain_start = n_frames - 1
        decay_ms = (sustain_start - peak_idx) * frame_ms
        sustain_end = sustain_start
        for i in range(sustain_start, n_frames):
            if rms[i] < 0.1 * peak_val:
                sustain_end = i
                break
        else:
            sustain_end = n_frames - 1
        sustain_region = rms[sustain_start:sustain_end + 1]
        sustain_level = float(np.mean(sustain_region) / peak_val) if len(sustain_region) > 0 else 0.0
        release_ms = (n_frames - 1 - sustain_end) * frame_ms
        return {
            "attack_ms": attack_ms,
            "decay_ms": decay_ms,
            "sustain_level": sustain_level,
            "release_ms": release_ms,
        }

    def extract_all(self, signal: np.ndarray, sr: int) -> dict:
        attack = self.attack_characteristics(signal, sr)
        steady = self.steady_state_features(signal, sr)
        temporal = self.temporal_features(signal, sr)
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(min(1024, len(signal))) / min(1024, len(signal))))
        chunk = signal[:len(w)]
        if len(chunk) < len(w):
            chunk = np.pad(chunk, (0, len(w) - len(chunk)))
        spec = np.abs(np.fft.rfft(chunk * w))
        envelope = self.spectral_envelope(spec)
        return {**attack, **steady, **temporal, "spectral_envelope": envelope}


INSTRUMENT_FAMILIES = {
    "strings": {
        "attack_time_range": (0.02, 0.3),
        "vibrato_typical": True,
        "spectral_flux_low": True,
    },
    "woodwinds": {
        "attack_time_range": (0.01, 0.15),
        "vibrato_typical": True,
        "spectral_flux_low": True,
    },
    "brass": {
        "attack_time_range": (0.02, 0.1),
        "vibrato_typical": False,
        "spectral_flux_low": False,
    },
    "percussion": {
        "attack_time_range": (0.0, 0.01),
        "vibrato_typical": False,
        "spectral_flux_low": False,
    },
    "keyboard": {
        "attack_time_range": (0.0, 0.02),
        "vibrato_typical": False,
        "spectral_flux_low": True,
    },
}


@dataclass
class InstrumentSimilarityMatrix:
    """Compute pairwise timbre similarity from feature vectors."""

    feature_weights: Optional[np.ndarray] = None

    def compute(self, feature_vectors: np.ndarray) -> np.ndarray:
        n = feature_vectors.shape[0]
        if self.feature_weights is not None:
            weighted = feature_vectors * self.feature_weights[np.newaxis, :]
        else:
            weighted = feature_vectors
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = weighted / norms
        sim = normed @ normed.T
        return np.clip(sim, -1.0, 1.0)

    def cluster(
        self, feature_vectors: np.ndarray, n_clusters: int = 5, max_iter: int = 100, seed: int = 42
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = feature_vectors.shape[0]
        if n <= n_clusters:
            return np.arange(n)
        indices = rng.choice(n, n_clusters, replace=False)
        centroids = feature_vectors[indices].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.array([
                np.linalg.norm(feature_vectors - c, axis=1) for c in centroids
            ])
            new_labels = np.argmin(dists, axis=0)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_clusters):
                members = feature_vectors[labels == k]
                if len(members) > 0:
                    centroids[k] = members.mean(axis=0)
        return labels


@dataclass
class InstrumentFamilyClassifier:
    """Rule-based + feature-distance instrument family classifier."""

    prototypes: dict = field(default_factory=lambda: {
        "strings": np.array([0.15, 5.0, 0.1, 50.0, 0.5, 0.4]),
        "woodwinds": np.array([0.05, 5.5, 0.05, 30.0, 0.3, 0.5]),
        "brass": np.array([0.04, 0.0, 0.3, 40.0, 0.2, 0.3]),
        "percussion": np.array([0.002, 0.0, 0.5, 5.0, 0.0, 0.8]),
        "keyboard": np.array([0.005, 0.0, 0.1, 20.0, 0.1, 0.6]),
    })

    def _features_to_vector(self, features: dict) -> np.ndarray:
        return np.array([
            features.get("attack_time", 0.0),
            features.get("vibrato_rate", 0.0),
            features.get("spectral_flux_mean", 0.0),
            features.get("decay_ms", 0.0),
            features.get("sustain_level", 0.0),
            features.get("onset_strength", 0.0),
        ])

    def classify(self, features: dict) -> tuple[str, float]:
        vec = self._features_to_vector(features)
        best_family = "unknown"
        best_dist = float("inf")
        for family, proto in self.prototypes.items():
            d = np.linalg.norm(vec - proto)
            if d < best_dist:
                best_dist = d
                best_family = family
        max_dist = 100.0
        confidence = max(0.0, 1.0 - best_dist / max_dist)
        return best_family, confidence

    def feature_importance(self) -> dict:
        protos = np.array(list(self.prototypes.values()))
        variances = np.var(protos, axis=0)
        total = np.sum(variances)
        if total == 0:
            total = 1.0
        names = ["attack_time", "vibrato_rate", "spectral_flux_mean",
                 "decay_ms", "sustain_level", "onset_strength"]
        return {n: float(v / total) for n, v in zip(names, variances)}
