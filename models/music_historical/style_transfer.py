"""Style transfer between historical periods: feature extraction, transformation, interpolation."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class PeriodStyle:
    name: str
    harmonic_vocab: np.ndarray
    rhythm_complexity: float
    voice_count_range: Tuple[int, int]
    dynamics_range: Tuple[float, float]
    ornamentation_density: float
    tempo_range_bpm: Tuple[int, int]
    modulation_frequency: float
    dissonance_tolerance: float


PERIOD_STYLES: Dict[str, PeriodStyle] = {
    "renaissance": PeriodStyle(
        name="renaissance",
        harmonic_vocab=np.array([0, 3, 4, 5, 7, 8, 9]),
        rhythm_complexity=0.3,
        voice_count_range=(3, 6),
        dynamics_range=(0.3, 0.7),
        ornamentation_density=0.1,
        tempo_range_bpm=(60, 100),
        modulation_frequency=0.05,
        dissonance_tolerance=0.2,
    ),
    "baroque": PeriodStyle(
        name="baroque",
        harmonic_vocab=np.array([0, 2, 3, 4, 5, 7, 9, 10, 11]),
        rhythm_complexity=0.6,
        voice_count_range=(2, 5),
        dynamics_range=(0.2, 0.8),
        ornamentation_density=0.5,
        tempo_range_bpm=(50, 140),
        modulation_frequency=0.15,
        dissonance_tolerance=0.4,
    ),
    "classical": PeriodStyle(
        name="classical",
        harmonic_vocab=np.array([0, 2, 4, 5, 7, 9, 11]),
        rhythm_complexity=0.4,
        voice_count_range=(1, 4),
        dynamics_range=(0.1, 0.9),
        ornamentation_density=0.2,
        tempo_range_bpm=(60, 160),
        modulation_frequency=0.1,
        dissonance_tolerance=0.3,
    ),
    "romantic": PeriodStyle(
        name="romantic",
        harmonic_vocab=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        rhythm_complexity=0.5,
        voice_count_range=(1, 8),
        dynamics_range=(0.05, 1.0),
        ornamentation_density=0.3,
        tempo_range_bpm=(40, 180),
        modulation_frequency=0.25,
        dissonance_tolerance=0.6,
    ),
    "impressionist": PeriodStyle(
        name="impressionist",
        harmonic_vocab=np.array([0, 2, 4, 5, 6, 7, 9, 10, 11]),
        rhythm_complexity=0.35,
        voice_count_range=(2, 6),
        dynamics_range=(0.1, 0.7),
        ornamentation_density=0.15,
        tempo_range_bpm=(50, 120),
        modulation_frequency=0.3,
        dissonance_tolerance=0.7,
    ),
    "modern": PeriodStyle(
        name="modern",
        harmonic_vocab=np.array(list(range(12))),
        rhythm_complexity=0.8,
        voice_count_range=(1, 12),
        dynamics_range=(0.0, 1.0),
        ornamentation_density=0.4,
        tempo_range_bpm=(30, 240),
        modulation_frequency=0.4,
        dissonance_tolerance=0.9,
    ),
}


@dataclass
class StyleFeatureVector:
    harmonic_entropy: float
    rhythm_complexity: float
    dynamic_range: float
    ornamentation: float
    dissonance: float
    modulation_freq: float
    voice_density: float
    tempo_variance: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.harmonic_entropy, self.rhythm_complexity,
            self.dynamic_range, self.ornamentation,
            self.dissonance, self.modulation_freq,
            self.voice_density, self.tempo_variance,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> StyleFeatureVector:
        return cls(*arr[:8].tolist())


def extract_period_features(style: PeriodStyle) -> StyleFeatureVector:
    h_vocab = style.harmonic_vocab
    h_entropy = len(h_vocab) / 12.0
    dyn_range = style.dynamics_range[1] - style.dynamics_range[0]
    voice_density = np.mean(style.voice_count_range) / 12.0
    tempo_var = (style.tempo_range_bpm[1] - style.tempo_range_bpm[0]) / 240.0
    return StyleFeatureVector(
        harmonic_entropy=h_entropy,
        rhythm_complexity=style.rhythm_complexity,
        dynamic_range=dyn_range,
        ornamentation=style.ornamentation_density,
        dissonance=style.dissonance_tolerance,
        modulation_freq=style.modulation_frequency,
        voice_density=voice_density,
        tempo_variance=tempo_var,
    )


def style_distance(a: StyleFeatureVector, b: StyleFeatureVector) -> float:
    return float(np.linalg.norm(a.to_array() - b.to_array()))


def interpolate_styles(source: StyleFeatureVector, target: StyleFeatureVector,
                       alpha: float) -> StyleFeatureVector:
    alpha = np.clip(alpha, 0.0, 1.0)
    blended = (1.0 - alpha) * source.to_array() + alpha * target.to_array()
    return StyleFeatureVector.from_array(blended)


def closest_period(features: StyleFeatureVector) -> str:
    best_name = ""
    best_dist = float("inf")
    for name, style in PERIOD_STYLES.items():
        pf = extract_period_features(style)
        d = style_distance(features, pf)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


@dataclass
class TransferResult:
    pitches: np.ndarray
    durations: np.ndarray
    dynamics: np.ndarray
    style_similarity: float


def constrain_to_vocab(pitches: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    result = np.empty_like(pitches)
    for i, p in enumerate(pitches):
        pc = int(p) % 12
        octave = int(p) // 12
        diffs = np.abs(vocab - pc)
        closest_pc = vocab[np.argmin(diffs)]
        result[i] = octave * 12 + closest_pc
    return result


def scale_rhythm_complexity(durations: np.ndarray, target_complexity: float) -> np.ndarray:
    if len(durations) == 0:
        return durations
    current = float(np.std(durations) / (np.mean(durations) + 1e-12))
    if current < 1e-12:
        return durations
    ratio = target_complexity / current
    mean_d = np.mean(durations)
    scaled = mean_d + (durations - mean_d) * ratio
    return np.maximum(scaled, 0.01)


def apply_dynamics_range(dynamics: np.ndarray, low: float, high: float) -> np.ndarray:
    if len(dynamics) == 0:
        return dynamics
    d_min, d_max = dynamics.min(), dynamics.max()
    if d_max - d_min < 1e-12:
        return np.full_like(dynamics, (low + high) / 2)
    normalized = (dynamics - d_min) / (d_max - d_min)
    return low + normalized * (high - low)


def add_ornamentation(pitches: np.ndarray, density: float,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    result = list(pitches)
    insertions = 0
    for i in range(len(pitches) - 1):
        if rng.random() < density:
            ornament = pitches[i] + rng.choice([-1, 1, 2, -2])
            result.insert(i + 1 + insertions, ornament)
            insertions += 1
    return np.array(result)


def transfer_style(pitches: np.ndarray, durations: np.ndarray,
                   dynamics: np.ndarray, target_period: str,
                   strength: float = 1.0) -> TransferResult:
    style = PERIOD_STYLES.get(target_period)
    if style is None:
        return TransferResult(pitches, durations, dynamics, 0.0)

    new_pitches = constrain_to_vocab(pitches, style.harmonic_vocab)
    if strength < 1.0:
        new_pitches = np.round((1.0 - strength) * pitches + strength * new_pitches).astype(int)

    new_durations = scale_rhythm_complexity(durations, style.rhythm_complexity)
    new_durations = (1.0 - strength) * durations + strength * new_durations

    new_dynamics = apply_dynamics_range(dynamics, *style.dynamics_range)
    new_dynamics = (1.0 - strength) * dynamics + strength * new_dynamics

    target_features = extract_period_features(style)
    result_style = StyleFeatureVector(
        harmonic_entropy=len(np.unique(new_pitches % 12)) / 12.0,
        rhythm_complexity=float(np.std(new_durations) / (np.mean(new_durations) + 1e-12)),
        dynamic_range=float(new_dynamics.max() - new_dynamics.min()),
        ornamentation=style.ornamentation_density * strength,
        dissonance=style.dissonance_tolerance,
        modulation_freq=style.modulation_frequency,
        voice_density=np.mean(style.voice_count_range) / 12.0,
        tempo_variance=(style.tempo_range_bpm[1] - style.tempo_range_bpm[0]) / 240.0,
    )
    sim = 1.0 - style_distance(result_style, target_features) / (np.sqrt(8) + 1e-12)

    return TransferResult(new_pitches, new_durations, new_dynamics, max(0.0, sim))


def period_timeline() -> List[Tuple[str, int, int]]:
    return [
        ("renaissance", 1400, 1600),
        ("baroque", 1600, 1750),
        ("classical", 1750, 1820),
        ("romantic", 1820, 1900),
        ("impressionist", 1875, 1925),
        ("modern", 1900, 2025),
    ]


def interpolate_by_year(year: int) -> StyleFeatureVector:
    timeline = period_timeline()
    for i in range(len(timeline) - 1):
        name1, start1, end1 = timeline[i]
        name2, start2, end2 = timeline[i + 1]
        if start1 <= year <= start2:
            alpha = (year - start1) / max(start2 - start1, 1)
            f1 = extract_period_features(PERIOD_STYLES[name1])
            f2 = extract_period_features(PERIOD_STYLES[name2])
            return interpolate_styles(f1, f2, alpha)
    if year < timeline[0][1]:
        return extract_period_features(PERIOD_STYLES[timeline[0][0]])
    return extract_period_features(PERIOD_STYLES[timeline[-1][0]])
