"""Deep music emotion — dimensional, categorical, GEMS, BRECVEMA, trajectories."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# §1  Dimensional emotion models (Russell, 1980; Thayer; VAD)
# ---------------------------------------------------------------------------

def valence_arousal(tempo: float, mode: float, loudness_db: float,
                    pitch_mean: float) -> Tuple[float, float]:
    """Gabrielsson & Lindström (2001) rule-based VA mapping.
    mode ∈ [0,1] (0=minor, 1=major); pitch_mean in Hz."""
    # arousal: tempo + loudness dominant
    a = np.tanh(0.005 * (tempo - 100)) * 0.5 + np.tanh(0.05 * (loudness_db + 20)) * 0.3
    a += np.tanh(0.001 * (pitch_mean - 300)) * 0.2
    # valence: mode dominant
    v = (mode - 0.5) * 0.6 + np.tanh(0.003 * (tempo - 100)) * 0.25
    v += np.tanh(0.05 * (loudness_db + 30)) * 0.15
    return (float(np.clip(v, -1, 1)), float(np.clip(a, -1, 1)))


def vad_from_features(tempo: float, mode: float, loudness_db: float,
                      pitch_mean: float, spectral_centroid: float = 2000.0
                      ) -> Tuple[float, float, float]:
    """3D VAD (valence, arousal, dominance)."""
    v, a = valence_arousal(tempo, mode, loudness_db, pitch_mean)
    # dominance: loudness + low pitch + spectral brightness
    d = np.tanh(0.04 * (loudness_db + 20)) * 0.4
    d += np.tanh(-0.001 * (pitch_mean - 250)) * 0.3
    d += np.tanh(0.0005 * (spectral_centroid - 2000)) * 0.3
    return (v, a, float(np.clip(d, -1, 1)))


def thayer_quadrant(valence: float, arousal: float) -> str:
    """Thayer's 4-quadrant model."""
    if valence >= 0 and arousal >= 0:
        return "happy/excited"
    if valence < 0 and arousal >= 0:
        return "angry/anxious"
    if valence < 0 and arousal < 0:
        return "sad/depressed"
    return "calm/relaxed"


def circumplex_angle(valence: float, arousal: float) -> float:
    """Angle on Russell circumplex (degrees, 0=positive valence)."""
    return float(np.degrees(np.arctan2(arousal, valence)) % 360)

# ---------------------------------------------------------------------------
# §2  Categorical emotion — Ekman + Plutchik
# ---------------------------------------------------------------------------

PLUTCHIK_WHEEL = {
    "joy":          {"angle": 90,  "opposite": "sadness"},
    "trust":        {"angle": 135, "opposite": "disgust"},
    "fear":         {"angle": 180, "opposite": "anger"},
    "surprise":     {"angle": 225, "opposite": "anticipation"},
    "sadness":      {"angle": 270, "opposite": "joy"},
    "disgust":      {"angle": 315, "opposite": "trust"},
    "anger":        {"angle": 0,   "opposite": "fear"},
    "anticipation": {"angle": 45,  "opposite": "surprise"},
}


def plutchik_intensity(base_emotion: str, intensity: float) -> str:
    """Plutchik intensity levels: mild → base → intense."""
    intensity_map = {
        "joy":      ("serenity", "joy", "ecstasy"),
        "trust":    ("acceptance", "trust", "admiration"),
        "fear":     ("apprehension", "fear", "terror"),
        "surprise": ("distraction", "surprise", "amazement"),
        "sadness":  ("pensiveness", "sadness", "grief"),
        "disgust":  ("boredom", "disgust", "loathing"),
        "anger":    ("annoyance", "anger", "rage"),
        "anticipation": ("interest", "anticipation", "vigilance"),
    }
    levels = intensity_map.get(base_emotion, (base_emotion,) * 3)
    if intensity < 0.33:
        return levels[0]
    if intensity < 0.67:
        return levels[1]
    return levels[2]


def plutchik_dyad(emotion_a: str, emotion_b: str) -> Optional[str]:
    """Primary dyads (adjacent emotions on the wheel)."""
    dyads = {
        frozenset({"joy", "trust"}): "love",
        frozenset({"trust", "fear"}): "submission",
        frozenset({"fear", "surprise"}): "awe",
        frozenset({"surprise", "sadness"}): "disapproval",
        frozenset({"sadness", "disgust"}): "remorse",
        frozenset({"disgust", "anger"}): "contempt",
        frozenset({"anger", "anticipation"}): "aggressiveness",
        frozenset({"anticipation", "joy"}): "optimism",
    }
    return dyads.get(frozenset({emotion_a, emotion_b}))


def emotion_from_va(valence: float, arousal: float) -> str:
    """Map VA coordinates to nearest Plutchik basic emotion."""
    angle = circumplex_angle(valence, arousal)
    best = "joy"
    best_dist = 360
    for name, info in PLUTCHIK_WHEEL.items():
        d = min(abs(angle - info["angle"]), 360 - abs(angle - info["angle"]))
        if d < best_dist:
            best_dist = d
            best = name
    return best

# ---------------------------------------------------------------------------
# §3  GEMS (Zentner et al., 2008)
# ---------------------------------------------------------------------------

GEMS_FACTORS = [
    "wonder", "transcendence", "tenderness", "nostalgia",
    "peacefulness", "power", "joyful_activation", "sadness", "tension",
]

GEMS_FEATURE_WEIGHTS = {
    "wonder":            {"tempo": -0.1, "mode": 0.3, "dynamics": 0.4, "harmonic_complexity": 0.5},
    "transcendence":     {"tempo": -0.2, "mode": 0.2, "dynamics": -0.1, "harmonic_complexity": 0.6},
    "tenderness":        {"tempo": -0.3, "mode": 0.4, "dynamics": -0.3, "harmonic_complexity": 0.1},
    "nostalgia":         {"tempo": -0.2, "mode": -0.3, "dynamics": -0.2, "harmonic_complexity": 0.3},
    "peacefulness":      {"tempo": -0.4, "mode": 0.3, "dynamics": -0.5, "harmonic_complexity": -0.2},
    "power":             {"tempo": 0.3, "mode": 0.1, "dynamics": 0.6, "harmonic_complexity": 0.2},
    "joyful_activation": {"tempo": 0.5, "mode": 0.5, "dynamics": 0.3, "harmonic_complexity": -0.1},
    "sadness":           {"tempo": -0.3, "mode": -0.5, "dynamics": -0.3, "harmonic_complexity": 0.2},
    "tension":           {"tempo": 0.2, "mode": -0.3, "dynamics": 0.4, "harmonic_complexity": 0.5},
}


def gems_profile(tempo: float, mode: float, dynamics: float,
                 harmonic_complexity: float) -> Dict[str, float]:
    """GEMS 9-factor profile from audio features."""
    features = {
        "tempo": np.tanh(0.005 * (tempo - 100)),
        "mode": mode * 2 - 1,
        "dynamics": np.tanh(0.03 * (dynamics + 20)),
        "harmonic_complexity": harmonic_complexity,
    }
    profile = {}
    for factor, weights in GEMS_FEATURE_WEIGHTS.items():
        score = sum(weights[f] * features[f] for f in weights)
        profile[factor] = float(np.clip((score + 1) / 2, 0, 1))
    return profile


def gems_dominant(profile: Dict[str, float]) -> str:
    return max(profile, key=profile.get)

# ---------------------------------------------------------------------------
# §4  BRECVEMA mechanisms (Juslin, 2013)
# ---------------------------------------------------------------------------

BRECVEMA = [
    "brainstem_reflex", "rhythmic_entrainment", "evaluative_conditioning",
    "contagion", "visual_imagery", "episodic_memory",
    "aesthetic_judgment",
]

BRECVEMA_WEIGHTS = {
    "brainstem_reflex":       {"loudness": 0.6, "tempo": 0.3, "onset_rate": 0.4},
    "rhythmic_entrainment":   {"tempo": 0.5, "groove": 0.4, "regularity": 0.3},
    "evaluative_conditioning": {"familiarity": 0.7, "preference": 0.5},
    "contagion":              {"mode": 0.4, "pitch_contour": 0.3, "vibrato": 0.2},
    "visual_imagery":         {"harmonic_complexity": 0.3, "timbre_variety": 0.4},
    "episodic_memory":        {"familiarity": 0.8},
    "aesthetic_judgment":      {"harmonic_complexity": 0.3, "novelty": 0.4, "structure": 0.3},
}


def brecvema_activation(features: Dict[str, float]) -> Dict[str, float]:
    """Estimate BRECVEMA mechanism activation levels."""
    activation = {}
    for mech, weights in BRECVEMA_WEIGHTS.items():
        score = sum(weights.get(f, 0) * features.get(f, 0) for f in weights)
        activation[mech] = float(np.clip(score, 0, 1))
    return activation

# ---------------------------------------------------------------------------
# §5  Musical feature → emotion rules (Gabrielsson & Lindström, 2001)
# ---------------------------------------------------------------------------

def tempo_arousal(bpm: float) -> float:
    """Tempo → arousal: sigmoid centred at 120 BPM."""
    return float(1.0 / (1.0 + np.exp(-0.03 * (bpm - 120))))


def mode_valence(is_major: bool) -> float:
    """Major/minor → valence bias."""
    return 0.7 if is_major else 0.3


def loudness_arousal(db: float) -> float:
    """Loudness → arousal."""
    return float(np.clip((db + 60) / 80, 0, 1))


def pitch_height_valence(mean_midi: float) -> float:
    """Higher pitch → more positive valence (weak effect)."""
    return float(np.clip((mean_midi - 40) / 60, 0, 1))


def articulation_energy(staccato_ratio: float) -> float:
    """Staccato → higher perceived energy."""
    return float(np.clip(staccato_ratio, 0, 1))

# ---------------------------------------------------------------------------
# §6  Emotion trajectory analysis
# ---------------------------------------------------------------------------

def emotion_arc(va_sequence: np.ndarray) -> Dict[str, float]:
    """Characterise a valence-arousal trajectory."""
    if len(va_sequence) < 2:
        return {"mean_v": 0, "mean_a": 0, "range_v": 0, "range_a": 0,
                "change_rate": 0, "arc_length": 0}
    v, a = va_sequence[:, 0], va_sequence[:, 1]
    diffs = np.sqrt(np.diff(v) ** 2 + np.diff(a) ** 2)
    return {
        "mean_v": float(v.mean()),
        "mean_a": float(a.mean()),
        "range_v": float(v.max() - v.min()),
        "range_a": float(a.max() - a.min()),
        "change_rate": float(diffs.mean()),
        "arc_length": float(diffs.sum()),
    }


def emotion_contrast(va_a: np.ndarray, va_b: np.ndarray) -> float:
    """Euclidean distance between two emotion states in VA space."""
    return float(np.linalg.norm(va_a - va_b))


def tension_release_curve(arousal_seq: np.ndarray) -> np.ndarray:
    """Cumulative tension-release: positive = tension building, negative = release."""
    if len(arousal_seq) < 2:
        return np.array([0.0])
    return np.cumsum(np.diff(arousal_seq))

# ---------------------------------------------------------------------------
# §7  Mixed emotions / bittersweetness (Hunter et al., 2010)
# ---------------------------------------------------------------------------

def bittersweet_index(valence: float, arousal: float,
                      mode: float, tempo: float) -> float:
    """Bittersweetness: co-activation of positive and negative affect.
    High when mode and tempo give conflicting valence signals."""
    mode_v = mode * 2 - 1  # -1 to 1
    tempo_v = np.tanh(0.005 * (tempo - 100))
    conflict = abs(mode_v - tempo_v)
    # low arousal amplifies bittersweetness
    arousal_mod = 1.0 - abs(arousal) * 0.5
    return float(np.clip(conflict * arousal_mod, 0, 1))


def nostalgia_index(familiarity: float, temporal_distance: float,
                    valence: float) -> float:
    """Nostalgia: familiar + distant + mildly positive."""
    return float(np.clip(
        familiarity * 0.4 +
        np.tanh(temporal_distance * 0.1) * 0.3 +
        max(0, valence) * 0.3, 0, 1
    ))


def mixed_emotion_vector(emotions: Dict[str, float]) -> np.ndarray:
    """Project categorical emotion weights into a mixed-emotion space."""
    n = len(PLUTCHIK_WHEEL)
    vec = np.zeros(n)
    for i, name in enumerate(PLUTCHIK_WHEEL):
        vec[i] = emotions.get(name, 0.0)
    return vec / (np.linalg.norm(vec) + 1e-10)

# ---------------------------------------------------------------------------
# §8  Physiological arousal model (Gomez & Danuser, 2007)
# ---------------------------------------------------------------------------

def predicted_heart_rate(tempo: float, loudness_db: float,
                         arousal: float) -> float:
    """Predicted HR change (BPM delta) from musical features."""
    base = 72.0
    delta = 0.05 * (tempo - 100) + 0.1 * (loudness_db + 20) + 5.0 * arousal
    return float(base + np.clip(delta, -15, 25))


def predicted_skin_conductance(arousal: float, surprise: float) -> float:
    """Predicted SCR amplitude (μS) from arousal and surprise."""
    return float(np.clip(0.5 * arousal + 0.3 * surprise + 0.2, 0, 2.0))


def predicted_respiration_rate(tempo: float, arousal: float) -> float:
    """Predicted breaths/min from tempo and arousal."""
    base = 16.0
    delta = 0.02 * (tempo - 100) + 3.0 * arousal
    return float(base + np.clip(delta, -5, 10))

# ---------------------------------------------------------------------------
# §9  Cross-cultural emotion (Fritz et al., 2009)
# ---------------------------------------------------------------------------

UNIVERSAL_RECOGNITION = {
    "happy": 0.85, "sad": 0.80, "scary": 0.75,
}


def cultural_modulation(base_score: float, familiarity: float,
                        enculturation: float) -> float:
    """Modulate universal recognition by cultural familiarity."""
    return float(np.clip(
        base_score * (0.5 + 0.3 * familiarity + 0.2 * enculturation), 0, 1
    ))


def cross_cultural_recognition(emotion: str, familiarity: float = 0.5,
                                enculturation: float = 0.5) -> float:
    """Predicted recognition accuracy accounting for cultural factors."""
    base = UNIVERSAL_RECOGNITION.get(emotion, 0.5)
    return cultural_modulation(base, familiarity, enculturation)

# ---------------------------------------------------------------------------
# §10  Pipeline
# ---------------------------------------------------------------------------

@dataclass
class DeepEmotionAnalysis:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    quadrant: str = ""
    circumplex_angle_deg: float = 0.0
    categorical_emotion: str = ""
    plutchik_intensity_label: str = ""
    gems: Dict[str, float] = field(default_factory=dict)
    gems_dominant: str = ""
    brecvema: Dict[str, float] = field(default_factory=dict)
    bittersweet: float = 0.0
    nostalgia: float = 0.0
    predicted_hr: float = 0.0
    predicted_scr: float = 0.0
    predicted_rr: float = 0.0
    arc: Dict[str, float] = field(default_factory=dict)


def analyze_emotion_deep(tempo: float = 120.0, mode: float = 1.0,
                         loudness_db: float = -20.0,
                         pitch_mean: float = 440.0,
                         spectral_centroid: float = 2000.0,
                         harmonic_complexity: float = 0.5,
                         familiarity: float = 0.5,
                         va_trajectory: Optional[np.ndarray] = None
                         ) -> DeepEmotionAnalysis:
    """Full deep emotion analysis pipeline."""
    v, a, d = vad_from_features(tempo, mode, loudness_db, pitch_mean,
                                 spectral_centroid)
    quad = thayer_quadrant(v, a)
    angle = circumplex_angle(v, a)
    cat = emotion_from_va(v, a)
    intensity = (abs(v) + abs(a)) / 2
    intens_label = plutchik_intensity(cat, intensity)

    gems_prof = gems_profile(tempo, mode, loudness_db, harmonic_complexity)
    gems_dom = gems_dominant(gems_prof)

    brecv_features = {
        "loudness": loudness_arousal(loudness_db),
        "tempo": tempo_arousal(tempo),
        "mode": mode,
        "harmonic_complexity": harmonic_complexity,
        "familiarity": familiarity,
        "onset_rate": tempo / 60.0,
        "groove": 0.5,
        "regularity": 0.5,
        "preference": familiarity,
        "pitch_contour": 0.5,
        "vibrato": 0.3,
        "timbre_variety": 0.5,
        "novelty": 1.0 - familiarity,
        "structure": 0.5,
    }
    brecv = brecvema_activation(brecv_features)

    bitter = bittersweet_index(v, a, mode, tempo)
    nost = nostalgia_index(familiarity, 5.0, v)

    hr = predicted_heart_rate(tempo, loudness_db, a)
    scr = predicted_skin_conductance(a, 0.3)
    rr = predicted_respiration_rate(tempo, a)

    arc = {}
    if va_trajectory is not None and len(va_trajectory) >= 2:
        arc = emotion_arc(va_trajectory)

    return DeepEmotionAnalysis(
        valence=v, arousal=a, dominance=d,
        quadrant=quad, circumplex_angle_deg=angle,
        categorical_emotion=cat,
        plutchik_intensity_label=intens_label,
        gems=gems_prof, gems_dominant=gems_dom,
        brecvema=brecv,
        bittersweet=bitter, nostalgia=nost,
        predicted_hr=hr, predicted_scr=scr, predicted_rr=rr,
        arc=arc,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = analyze_emotion_deep(tempo=140, mode=1.0, loudness_db=-15,
                                   pitch_mean=500, harmonic_complexity=0.6)
    assert -1 <= result.valence <= 1
    assert -1 <= result.arousal <= 1
    assert result.quadrant != ""
    assert result.categorical_emotion in PLUTCHIK_WHEEL
    assert len(result.gems) == 9
    assert len(result.brecvema) == 7
    print(f"VA=({result.valence:.2f}, {result.arousal:.2f}), D={result.dominance:.2f}")
    print(f"Quadrant: {result.quadrant}, Emotion: {result.categorical_emotion}")
    print(f"GEMS dominant: {result.gems_dominant}")
    print(f"Bittersweet: {result.bittersweet:.2f}, Nostalgia: {result.nostalgia:.2f}")
    print(f"HR: {result.predicted_hr:.0f}, SCR: {result.predicted_scr:.2f}, "
          f"RR: {result.predicted_rr:.1f}")
    print("deep_emotion OK")
