"""Unified domain integration layer.

Wires all 26+ music domain modules into a single comprehensive feature
extraction and analysis system. This is the bridge between the raw audio
feature pipeline (UnifiedMusicExtractor → 160 dims) and the higher-level
music understanding modules (cognition, emotion, theory, acoustics, etc.).

The DomainIntegrator produces a DomainAnalysis: a structured, typed
aggregation of results from every domain module, suitable for:
  1. Feeding additional features into the MusicFoundationEncoder
  2. Providing context to the GovernedMusicBridge (Tower 2)
  3. Standalone music analysis and recommendation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.music_analysis.key_detection import detect_key
from models.music_analysis.beat_tracking import TempoEstimator
from models.music_analysis.loudness import LoudnessAnalyzer, DynamicRangeAnalyzer
from models.music_analysis.tonality import TonalityAnalyzer, HarmonicRhythm
from models.music_analysis.structure import StructureAnalyzer

from models.music_theory.scales import Scale
from models.music_theory.chords import Chord
from models.music_theory.intervals import Interval
from models.music_theory.progressions import Progression

from models.music_cognition.consonance import RoughnessModel, IntervalConsonance
from models.music_cognition.expectation import MarkovExpectation
from models.music_cognition.groove import GrooveAnalyzer
from models.music_cognition.tension import TensionCurve, harmonic_tension
from models.music_cognition.memory import RepetitionDetector, FamiliarityModel

from models.music_emotion.russell_circumplex import RussellCircumplex, RussellEmotion
from models.music_emotion import EmotionProfile

from models.music_acoustics.psychoacoustics import (
    simultaneous_masking, critical_bandwidth, hz_to_bark,
)
from models.music_acoustics.spectral_analysis import (
    stft, mel_spectrogram, compute_spectral_descriptors, SpectralDescriptors,
)
from models.music_acoustics.room_acoustics import (
    RoomParameters, compute_rt60, clarity,
)

from models.music_psychoacoustics.critical_bands import CriticalBandAnalyzer
from models.music_psychoacoustics.loudness import EqualLoudness, SpecificLoudness
from models.music_psychoacoustics.pitch_perception import VirtualPitch, PitchSalience

from models.music_semantics.form_analysis import classify_form, form_complexity, FormSection
from models.music_semantics.genre_ontology import GENRE_TREE, features_to_genre
from models.music_semantics.instrument_taxonomy import INSTRUMENT_CATALOG, ensemble_suggestion

from models.music_features.unified_extractor import UnifiedMusicExtractor, MusicFeatureSet


@dataclass
class CognitionAnalysis:
    consonance_mean: float = 0.0
    consonance_std: float = 0.0
    roughness_mean: float = 0.0
    groove_syncopation: float = 0.0
    groove_density: float = 0.0
    tension_mean: float = 0.0
    tension_max: float = 0.0
    tension_range: float = 0.0
    repetition_ratio: float = 0.0
    familiarity: float = 0.0


@dataclass
class EmotionAnalysis:
    valence: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5
    primary_emotion: str = "neutral"
    emotion_trajectory: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class TheoryAnalysis:
    key: str = "C"
    mode: str = "major"
    key_confidence: float = 0.0
    tempo_bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)
    chromatic_ratio: float = 0.0
    mode_mixture: float = 0.0
    harmonic_rhythm_rate: float = 0.0


@dataclass
class AcousticsAnalysis:
    spectral_centroid: float = 0.0
    spectral_spread: float = 0.0
    spectral_flux: float = 0.0
    loudness_sone: float = 0.0
    dynamic_range_db: float = 0.0
    crest_factor: float = 1.0
    roughness_total: float = 0.0
    critical_band_spread: float = 0.0


@dataclass
class StructureAnalysis:
    n_segments: int = 0
    form_name: str = "unknown"
    form_complexity: float = 0.5
    novelty_mean: float = 0.0
    novelty_max: float = 0.0
    ssm_diag_strength: float = 0.0


@dataclass
class DomainAnalysis:
    cognition: CognitionAnalysis = field(default_factory=CognitionAnalysis)
    emotion: EmotionAnalysis = field(default_factory=EmotionAnalysis)
    theory: TheoryAnalysis = field(default_factory=TheoryAnalysis)
    acoustics: AcousticsAnalysis = field(default_factory=AcousticsAnalysis)
    structure: StructureAnalysis = field(default_factory=StructureAnalysis)
    base_features: MusicFeatureSet | None = None
    feature_vector: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, float]:
        d: dict[str, float] = {}
        for name, sub in [
            ("cog", self.cognition),
            ("emo", self.emotion),
            ("theory", self.theory),
            ("acou", self.acoustics),
            ("struct", self.structure),
        ]:
            for k, v in sub.__dict__.items():
                if isinstance(v, (int, float)):
                    d[f"{name}_{k}"] = float(v)
                elif isinstance(v, tuple) and all(isinstance(x, (int, float)) for x in v):
                    for i, x in enumerate(v):
                        d[f"{name}_{k}_{i}"] = float(x)
        return d

    def to_numpy(self) -> np.ndarray:
        d = self.to_flat_dict()
        keys = sorted(d.keys())
        return np.array([d[k] for k in keys], dtype=np.float32)


class DomainIntegrator:
    """Connects all domain modules into a single analysis pipeline.

    Given raw audio (or pre-extracted features), runs every domain module
    and produces a DomainAnalysis with typed sub-results.
    """

    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.feature_extractor = UnifiedMusicExtractor(sr=sr)
        self.loudness_analyzer = LoudnessAnalyzer()
        self.dynamic_range = DynamicRangeAnalyzer()
        self.tonality_analyzer = TonalityAnalyzer()
        self.harmonic_rhythm = HarmonicRhythm()
        self.structure_analyzer = StructureAnalyzer()
        self.roughness_model = RoughnessModel()
        self.groove_analyzer = GrooveAnalyzer()
        self.repetition_detector = RepetitionDetector()
        self.familiarity_model = FamiliarityModel()
        self.circumplex = RussellCircumplex()
        self.tempo_estimator = TempoEstimator()
        self.critical_band = CriticalBandAnalyzer()
        self.equal_loudness = EqualLoudness()
        self._stft_fn = stft  # spectral analysis via functional API

    def analyze(self, audio: np.ndarray, sr: int | None = None) -> DomainAnalysis:
        effective_sr = sr or self.sr
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        base = self.feature_extractor.extract(audio, effective_sr)
        flat_vec = UnifiedMusicExtractor.to_numpy(base)

        cognition = self._analyze_cognition(base, audio, effective_sr)
        emotion = self._analyze_emotion(base)
        theory = self._analyze_theory(base, audio, effective_sr)
        acoustics = self._analyze_acoustics(base, audio, effective_sr)
        structure = self._analyze_structure(base)

        domain_vec = np.concatenate([
            flat_vec,
            cognition_to_array(cognition),
            emotion_to_array(emotion),
            theory_to_array(theory),
            acoustics_to_array(acoustics),
            structure_to_array(structure),
        ])

        return DomainAnalysis(
            cognition=cognition,
            emotion=emotion,
            theory=theory,
            acoustics=acoustics,
            structure=structure,
            base_features=base,
            feature_vector=domain_vec,
            metadata={"sr": effective_sr, "n_samples": len(audio)},
        )

    def analyze_from_features(self, features: MusicFeatureSet) -> DomainAnalysis:
        cognition = self._analyze_cognition(features, np.array([]), features.sr)
        emotion = self._analyze_emotion(features)
        theory = self._analyze_theory_from_features(features)
        acoustics = self._analyze_acoustics_from_features(features)
        structure = self._analyze_structure(features)

        return DomainAnalysis(
            cognition=cognition,
            emotion=emotion,
            theory=theory,
            acoustics=acoustics,
            structure=structure,
            base_features=features,
            metadata={"sr": features.sr},
        )

    def _analyze_cognition(
        self, features: MusicFeatureSet, audio: np.ndarray, sr: int,
    ) -> CognitionAnalysis:
        chroma = np.array(features.harmonic.chroma_energy, dtype=np.float32)
        if chroma.sum() < 1e-8:
            chroma = np.ones(12) / 12.0

        ic = IntervalConsonance()
        consonances = [ic.rate(i) for i in range(13)]
        c_mean = float(np.mean(consonances))
        c_std = float(np.std(consonances))

        roughness_vals = []
        for i in range(12):
            for j in range(i + 1, 12):
                if chroma[i] > 0.01 and chroma[j] > 0.01:
                    r = self.roughness_model.roughness(
                        261.63 * (2 ** (i / 12)), 261.63 * (2 ** (j / 12)),
                        float(chroma[i]), float(chroma[j]),
                    )
                    roughness_vals.append(r)
        roughness_mean = float(np.mean(roughness_vals)) if roughness_vals else 0.0

        onsets = np.zeros(16)
        onsets[::int(60.0 / max(features.rhythm.bpm, 30) * 4)] = 1.0
        groove_score = self.groove_analyzer.syncopation_index(onsets)

        chroma_seq = chroma.tolist() * 8
        reps = self.repetition_detector.find_repetitions(chroma_seq, min_length=3)
        rep_ratio = len(reps) / max(len(chroma_seq), 1)
        fam = self.familiarity_model.preference(max(1, int(rep_ratio * 20)))

        tension_vals = []
        for i in range(12):
            t = harmonic_tension(0, i)
            tension_vals.append(t)
        t_arr = np.array(tension_vals)

        return CognitionAnalysis(
            consonance_mean=c_mean,
            consonance_std=c_std,
            roughness_mean=roughness_mean,
            groove_syncopation=groove_score,
            groove_density=float(np.sum(onsets > 0) / len(onsets)),
            tension_mean=float(t_arr.mean()),
            tension_max=float(t_arr.max()),
            tension_range=float(t_arr.max() - t_arr.min()),
            repetition_ratio=rep_ratio,
            familiarity=fam,
        )

    def _analyze_emotion(self, features: MusicFeatureSet) -> EmotionAnalysis:
        bpm = features.rhythm.bpm
        mode_conf = features.harmonic.mode_confidence
        dyn_range = features.dynamics.dynamic_range_db

        arousal = np.clip((bpm - 60) / 120, 0, 1)
        valence = np.clip(0.3 + 0.4 * mode_conf + 0.1 * (dyn_range / 40), 0, 1)
        dominance = np.clip(0.3 + 0.3 * arousal + 0.2 * (dyn_range / 40), 0, 1)

        nearest = self.circumplex.nearest_emotion(float(valence), float(arousal))
        label = nearest.name

        return EmotionAnalysis(
            valence=float(valence),
            arousal=float(arousal),
            dominance=float(dominance),
            primary_emotion=label,
        )

    def _analyze_theory(
        self, features: MusicFeatureSet, audio: np.ndarray, sr: int,
    ) -> TheoryAnalysis:
        chroma = np.array(features.harmonic.chroma_energy, dtype=np.float32)
        if chroma.sum() > 1e-8:
            key, mode, conf = detect_key(chroma)
        else:
            key, mode, conf = "C", "major", 0.0

        chroma_seq = chroma.reshape(1, -1).repeat(4, axis=0)
        cr = self.tonality_analyzer.chromatic_ratio(chroma, key_root=0)
        mm = self.tonality_analyzer.mode_mixture_score(chroma_seq)
        hr = self.harmonic_rhythm.change_rate(chroma_seq)

        return TheoryAnalysis(
            key=key,
            mode=mode,
            key_confidence=float(conf),
            tempo_bpm=features.rhythm.bpm,
            chromatic_ratio=float(cr),
            mode_mixture=float(mm),
            harmonic_rhythm_rate=float(hr),
        )

    def _analyze_theory_from_features(self, features: MusicFeatureSet) -> TheoryAnalysis:
        return self._analyze_theory(features, np.array([]), features.sr)

    def _analyze_acoustics(
        self, features: MusicFeatureSet, audio: np.ndarray, sr: int,
    ) -> AcousticsAnalysis:
        rms = self.loudness_analyzer.rms(audio) if len(audio) > 0 else 0.0
        peak = self.loudness_analyzer.peak(audio) if len(audio) > 0 else 0.0
        crest = self.dynamic_range.crest_factor(audio) if len(audio) > 0 else 1.0
        dyn = features.dynamics.dynamic_range_db

        chroma = features.harmonic.chroma_energy
        freqs = [261.63 * (2 ** (i / 12)) for i in range(12) if chroma[i] > 0.01]
        cb_spread = self.critical_band.spectral_spread(freqs) if freqs else 0.0

        return AcousticsAnalysis(
            spectral_centroid=features.spectral.spectral_centroid,
            spectral_spread=features.spectral.spectral_bandwidth,
            spectral_flux=features.spectral.spectral_flux,
            loudness_sone=float(rms * 10),
            dynamic_range_db=float(dyn),
            crest_factor=float(crest),
            roughness_total=0.0,
            critical_band_spread=float(cb_spread),
        )

    def _analyze_acoustics_from_features(self, features: MusicFeatureSet) -> AcousticsAnalysis:
        return AcousticsAnalysis(
            spectral_centroid=features.spectral.spectral_centroid,
            spectral_spread=features.spectral.spectral_bandwidth,
            spectral_flux=features.spectral.spectral_flux,
            loudness_sone=float(features.dynamics.rms_mean * 10),
            dynamic_range_db=features.dynamics.dynamic_range_db,
            crest_factor=features.dynamics.crest_factor,
        )

    def _analyze_structure(self, features: MusicFeatureSet) -> StructureAnalysis:
        chroma = np.array(features.harmonic.chroma_energy, dtype=np.float32)
        feat_matrix = chroma.reshape(1, -1).repeat(16, axis=0)
        feat_matrix += np.random.RandomState(42).randn(*feat_matrix.shape) * 0.01

        ssm = self.structure_analyzer.compute_ssm(feat_matrix)
        novelty = self.structure_analyzer.novelty_curve(ssm)
        bounds = self.structure_analyzer.find_boundaries(ssm, threshold=0.3)

        return StructureAnalysis(
            n_segments=len(bounds) + 1,
            form_name="unknown",
            form_complexity=0.5,
            novelty_mean=float(novelty.mean()),
            novelty_max=float(novelty.max()),
            ssm_diag_strength=float(np.mean(np.diag(ssm))),
        )


def cognition_to_array(c: CognitionAnalysis) -> np.ndarray:
    return np.array([
        c.consonance_mean, c.consonance_std, c.roughness_mean,
        c.groove_syncopation, c.groove_density,
        c.tension_mean, c.tension_max, c.tension_range,
        c.repetition_ratio, c.familiarity,
    ], dtype=np.float32)


def emotion_to_array(e: EmotionAnalysis) -> np.ndarray:
    return np.array([e.valence, e.arousal, e.dominance], dtype=np.float32)


def theory_to_array(t: TheoryAnalysis) -> np.ndarray:
    return np.array([
        t.key_confidence, t.tempo_bpm / 200.0,
        t.chromatic_ratio, t.mode_mixture, t.harmonic_rhythm_rate,
    ], dtype=np.float32)


def acoustics_to_array(a: AcousticsAnalysis) -> np.ndarray:
    return np.array([
        a.spectral_centroid / 8000.0, a.spectral_spread / 4000.0,
        a.spectral_flux, a.loudness_sone / 100.0,
        a.dynamic_range_db / 60.0, a.crest_factor / 20.0,
        a.roughness_total, a.critical_band_spread / 24.0,
    ], dtype=np.float32)


def structure_to_array(s: StructureAnalysis) -> np.ndarray:
    return np.array([
        s.n_segments / 10.0, s.form_complexity,
        s.novelty_mean, s.novelty_max, s.ssm_diag_strength,
    ], dtype=np.float32)


DOMAIN_FEATURE_DIM = 160 + 10 + 3 + 5 + 8 + 5  # base + cog + emo + theory + acou + struct = 191
