from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from models.music_features.spectral import SpectralFeatures, SpectralExtractor
from models.music_features.rhythm import RhythmFeatures, RhythmExtractor
from models.music_features.harmonic import HarmonicFeatures, HarmonicExtractor
from models.music_features.dynamics import DynamicsFeatures, DynamicsExtractor
from models.music_features.timbre import TimbreFeatures, TimbreExtractor


@dataclass
class MusicFeatureSet:
    spectral: SpectralFeatures
    rhythm: RhythmFeatures
    harmonic: HarmonicFeatures
    dynamics: DynamicsFeatures
    timbre: TimbreFeatures
    sr: int = 22050
    duration_s: float = 0.0
    n_samples: int = 0


class UnifiedMusicExtractor:
    def __init__(self, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectral_ext = SpectralExtractor(sr=sr, n_fft=n_fft, hop_length=hop_length)
        self.rhythm_ext = RhythmExtractor(sr=sr, hop_length=hop_length)
        self.harmonic_ext = HarmonicExtractor(sr=sr, n_fft=min(n_fft * 2, 4096), hop_length=hop_length)
        self.dynamics_ext = DynamicsExtractor(frame_length=n_fft, hop_length=hop_length)
        self.timbre_ext = TimbreExtractor(n_fft=n_fft, hop_length=hop_length)

    def extract(self, audio: np.ndarray, sr: int | None = None) -> MusicFeatureSet:
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        effective_sr = sr or self.sr
        duration = len(audio) / effective_sr if effective_sr > 0 else 0.0

        spectral = self.spectral_ext.extract(audio)
        rhythm = self.rhythm_ext.extract(audio)
        harmonic = self.harmonic_ext.extract(audio)
        dynamics = self.dynamics_ext.extract(audio, sr=effective_sr)
        timbre = self.timbre_ext.extract(audio, sr=effective_sr)

        return MusicFeatureSet(
            spectral=spectral,
            rhythm=rhythm,
            harmonic=harmonic,
            dynamics=dynamics,
            timbre=timbre,
            sr=effective_sr,
            duration_s=duration,
            n_samples=len(audio),
        )

    def extract_flat(self, audio: np.ndarray, sr: int | None = None) -> dict[str, float]:
        fset = self.extract(audio, sr)
        return self._flatten(fset)

    @staticmethod
    def feature_names() -> list[str]:
        names = []
        names += [
            "spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
            "spectral_flatness", "spectral_entropy", "spectral_flux",
            "zero_crossing_rate",
        ]
        names += [f"spectral_contrast_{i}" for i in range(7)]
        names += [f"mfcc_{i}" for i in range(13)]
        names += [f"chroma_{i}" for i in range(12)]
        names += [
            "bpm", "bpm_confidence", "onset_rate",
            "onset_strength_mean", "onset_strength_std",
            "tempo_stability", "beat_regularity", "swing_ratio",
            "groove_consistency",
        ]
        names += [
            "key_confidence", "mode_confidence", "harmonic_change_rate",
            "dissonance_mean", "dissonance_std", "key_clarity",
        ]
        names += [f"chroma_energy_{i}" for i in range(12)]
        names += [f"tonal_centroid_{i}" for i in range(6)]
        names += [f"pitch_class_hist_{i}" for i in range(12)]
        names += [
            "rms_mean", "rms_std", "rms_max", "dynamic_range_db",
            "loudness_lufs_approx", "crest_factor",
            "attack_time_mean", "release_time_mean", "dynamic_complexity",
        ]
        names += [
            "brightness", "warmth", "roughness", "sharpness",
            "boominess", "depth", "hardness", "spectral_centroid_var",
        ]
        names += [f"mfcc_delta_{i}" for i in range(13)]
        names += [f"mfcc_delta2_{i}" for i in range(13)]
        names += [f"timbre_contrast_{i}" for i in range(7)]
        names += [f"tonnetz_{i}" for i in range(6)]
        return names

    @staticmethod
    def to_numpy(feature_set: MusicFeatureSet) -> np.ndarray:
        flat = UnifiedMusicExtractor._flatten(feature_set)
        names = UnifiedMusicExtractor.feature_names()
        return np.array([flat.get(n, 0.0) for n in names], dtype=np.float32)

    @staticmethod
    def _flatten(fset: MusicFeatureSet) -> dict[str, float]:
        d: dict[str, float] = {}
        s = fset.spectral
        d["spectral_centroid"] = s.spectral_centroid
        d["spectral_rolloff"] = s.spectral_rolloff
        d["spectral_bandwidth"] = s.spectral_bandwidth
        d["spectral_flatness"] = s.spectral_flatness
        d["spectral_entropy"] = s.spectral_entropy
        d["spectral_flux"] = s.spectral_flux
        d["zero_crossing_rate"] = s.zero_crossing_rate
        for i, v in enumerate(s.spectral_contrast):
            d[f"spectral_contrast_{i}"] = v
        for i, v in enumerate(s.mfcc):
            d[f"mfcc_{i}"] = v
        for i, v in enumerate(s.chroma):
            d[f"chroma_{i}"] = v

        r = fset.rhythm
        d["bpm"] = r.bpm
        d["bpm_confidence"] = r.bpm_confidence
        d["onset_rate"] = r.onset_rate
        d["onset_strength_mean"] = r.onset_strength_mean
        d["onset_strength_std"] = r.onset_strength_std
        d["tempo_stability"] = r.tempo_stability
        d["beat_regularity"] = r.beat_regularity
        d["swing_ratio"] = r.swing_ratio
        d["groove_consistency"] = r.groove_consistency

        h = fset.harmonic
        d["key_confidence"] = h.key_confidence
        d["mode_confidence"] = h.mode_confidence
        d["harmonic_change_rate"] = h.harmonic_change_rate
        d["dissonance_mean"] = h.dissonance_mean
        d["dissonance_std"] = h.dissonance_std
        d["key_clarity"] = h.key_clarity
        for i, v in enumerate(h.chroma_energy):
            d[f"chroma_energy_{i}"] = v
        for i, v in enumerate(h.tonal_centroid):
            d[f"tonal_centroid_{i}"] = v
        for i, v in enumerate(h.pitch_class_histogram):
            d[f"pitch_class_hist_{i}"] = v

        dy = fset.dynamics
        d["rms_mean"] = dy.rms_mean
        d["rms_std"] = dy.rms_std
        d["rms_max"] = dy.rms_max
        d["dynamic_range_db"] = dy.dynamic_range_db
        d["loudness_lufs_approx"] = dy.loudness_lufs_approx
        d["crest_factor"] = dy.crest_factor
        d["attack_time_mean"] = dy.attack_time_mean
        d["release_time_mean"] = dy.release_time_mean
        d["dynamic_complexity"] = dy.dynamic_complexity

        t = fset.timbre
        d["brightness"] = t.brightness
        d["warmth"] = t.warmth
        d["roughness"] = t.roughness
        d["sharpness"] = t.sharpness
        d["boominess"] = t.boominess
        d["depth"] = t.depth
        d["hardness"] = t.hardness
        d["spectral_centroid_var"] = t.spectral_centroid_var
        for i, v in enumerate(t.mfcc_delta_mean):
            d[f"mfcc_delta_{i}"] = v
        for i, v in enumerate(t.mfcc_delta2_mean):
            d[f"mfcc_delta2_{i}"] = v
        for i, v in enumerate(t.spectral_contrast_mean):
            d[f"timbre_contrast_{i}"] = v
        for i, v in enumerate(t.tonnetz_mean):
            d[f"tonnetz_{i}"] = v

        return d
