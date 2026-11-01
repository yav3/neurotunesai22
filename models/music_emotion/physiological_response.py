"""Physiological response modelling: music feature to autonomic nervous system mapping.

Maps acoustic and musical features to predicted physiological responses
(heart rate, skin conductance, respiration, muscle tension, pupil dilation).
Pure numpy/scipy implementation grounded in psychophysiology literature:

- Bernardi et al. (2006) cardiovascular/respiratory entrainment to music tempo
- Grewe et al. (2007) skin conductance response to musical surprise
- Huron (2006) ITPRA theory of musical expectation and chills/frisson
- Hodges (2010) "Handbook of Music and Emotion" ch. 11 physiological responses
- Juslin & Sloboda (2010) ch. 9 autonomic nervous system measures
- Gagnon & Peretz (2003) mode and tempo contributions to perceived valence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import circmean, circstd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysiologicalResponse:
    """Predicted autonomic nervous system response to a musical stimulus.

    All values are signed deviations from an assumed resting baseline
    except ``pupil_dilation`` (non-negative mm change).
    """

    heart_rate_change: float      # BPM deviation from baseline
    skin_conductance: float       # micro-Siemens above baseline
    respiration_rate: float       # breaths-per-minute deviation
    muscle_tension: float         # EMG micro-volts above baseline (frontalis)
    pupil_dilation: float         # mm increase from baseline


@dataclass(frozen=True)
class MusicPhysioMapping:
    """Regression coefficients linking music features to physiology.

    Defaults drawn from Bernardi et al. (2006) and Hodges (2010).
    """

    tempo_hr_slope: float = 0.12          # BPM HR per BPM tempo
    mode_valence_coeff: float = 0.15      # valence shift for major mode
    loudness_arousal_coeff: float = 0.04  # arousal per dB
    dissonance_tension_coeff: float = 0.30  # tension per unit dissonance


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sigmoid(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    """Logistic sigmoid with tuneable midpoint and steepness."""
    z = steepness * (x - midpoint)
    # Clamp to prevent overflow
    z = float(np.clip(z, -30.0, 30.0))
    return 1.0 / (1.0 + np.exp(-z))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value, lo, hi))


# ---------------------------------------------------------------------------
# Core prediction functions
# ---------------------------------------------------------------------------

def predict_heart_rate_change(
    tempo_bpm: float,
    loudness_db: float,
    mode_major: bool,
    mapping: MusicPhysioMapping | None = None,
) -> float:
    """Predict heart-rate deviation (BPM) from resting baseline.

    Model after Bernardi et al. (2006): faster tempo and louder music
    elevate HR; major mode contributes a small positive offset reflecting
    higher motor engagement via positive affect.

    Parameters
    ----------
    tempo_bpm : float
        Tempo of the musical excerpt (BPM).
    loudness_db : float
        Integrated loudness in dBFS (typically -60..0).
    mode_major : bool
        True for major-mode passages.
    mapping : MusicPhysioMapping, optional
        Override default coefficients.

    Returns
    -------
    float
        Predicted HR change in BPM (positive = elevation).
    """
    m = mapping or MusicPhysioMapping()
    # Tempo component: deviation from a neutral 80 BPM baseline
    tempo_component = m.tempo_hr_slope * (tempo_bpm - 80.0)
    # Loudness component: deviation from -30 dBFS reference
    loudness_component = m.loudness_arousal_coeff * (loudness_db - (-30.0))
    # Mode component
    mode_component = 3.0 * m.mode_valence_coeff if mode_major else -1.5 * m.mode_valence_coeff
    return float(tempo_component + loudness_component + mode_component)


def predict_skin_conductance(
    onset_density: float,
    dynamic_range_db: float,
    surprise_score: float,
) -> float:
    """Predict skin conductance response (micro-Siemens above baseline).

    Follows Grewe et al. (2007): SCR scales with musical surprise, onset
    density (event rate), and dynamic range.  A sigmoid saturates the
    surprise contribution to model the refractory period of the
    electrodermal system.

    Parameters
    ----------
    onset_density : float
        Note onsets per second (Hz).  Typical range 1--12.
    dynamic_range_db : float
        Loudness dynamic range in dB.  Typical range 5--40.
    surprise_score : float
        Musical surprise / information content in bits.  0 = fully expected.

    Returns
    -------
    float
        Predicted SCR amplitude (micro-Siemens, non-negative).
    """
    surprise_component = 2.0 * _sigmoid(surprise_score, midpoint=3.0, steepness=1.2)
    density_component = 0.3 * _clamp(onset_density / 12.0)
    dynamic_component = 0.2 * _clamp(dynamic_range_db / 40.0)
    return max(0.0, surprise_component + density_component + dynamic_component)


def predict_respiration_sync(
    tempo_bpm: float,
    regularity: float,
) -> float:
    """Respiratory entrainment probability to musical pulse.

    Bernardi et al. (2006) found respiration synchronises most strongly
    when the musical tempo maps onto natural breathing rates (12--20
    breaths/min).  High metric regularity strengthens entrainment.

    Parameters
    ----------
    tempo_bpm : float
        Musical tempo in BPM.
    regularity : float
        Metric regularity / pulse clarity in [0, 1].

    Returns
    -------
    float
        Probability of respiratory entrainment in [0, 1].
    """
    # Breathing-rate window expressed in BPM of a hypothetical sub-division
    breath_rate_bpm_low = 12.0
    breath_rate_bpm_high = 20.0
    # Integer ratios (1:1, 1:2, 1:3, 1:4) of tempo to breath rate
    best_proximity = 1.0
    for divisor in (1, 2, 3, 4):
        effective_bpm = tempo_bpm / divisor
        if effective_bpm < breath_rate_bpm_low:
            continue
        dist = 0.0
        if effective_bpm < breath_rate_bpm_low:
            dist = breath_rate_bpm_low - effective_bpm
        elif effective_bpm > breath_rate_bpm_high:
            dist = effective_bpm - breath_rate_bpm_high
        proximity = np.exp(-0.05 * dist * dist)
        best_proximity = min(best_proximity, 1.0 - float(proximity))
    sync_base = 1.0 - best_proximity
    return _clamp(sync_base * _clamp(regularity))


def chills_probability(
    harmonic_surprise: float,
    dynamic_crescendo: float,
    melodic_appoggiatura: float,
) -> float:
    """Probability of experiencing frisson / chills.

    Integrates three primary triggers identified by Huron (2006) ITPRA
    theory and Grewe et al. (2007):

    1. Harmonic surprise (unexpected chord / modulation)
    2. Dynamic crescendo (sudden loudness increase)
    3. Melodic appoggiatura (non-chord tone resolving by step)

    Parameters
    ----------
    harmonic_surprise : float
        Information-theoretic surprise of the harmony (bits, >= 0).
    dynamic_crescendo : float
        Crescendo magnitude in [0, 1] (0 = none, 1 = ff subito).
    melodic_appoggiatura : float
        Strength of appoggiatura / suspension resolution in [0, 1].

    Returns
    -------
    float
        Probability of chills in [0, 1].
    """
    # Weighted combination: harmonic surprise is strongest predictor
    raw = (
        0.50 * _sigmoid(harmonic_surprise, midpoint=4.0, steepness=0.8)
        + 0.30 * _clamp(dynamic_crescendo)
        + 0.20 * _clamp(melodic_appoggiatura)
    )
    return _clamp(raw)


def relaxation_index(
    tempo: float,
    consonance: float,
    spectral_centroid: float,
    dynamic_stability: float,
) -> float:
    """Composite relaxation score from musical features.

    Higher values indicate a more physiologically relaxing stimulus,
    suitable for therapeutic music selection (Hodges 2010, ch. 14).

    Parameters
    ----------
    tempo : float
        Tempo in BPM.
    consonance : float
        Harmonic consonance in [0, 1] (1 = perfectly consonant).
    spectral_centroid : float
        Spectral centroid in Hz.
    dynamic_stability : float
        Loudness stability (inverse of variance) in [0, 1].

    Returns
    -------
    float
        Relaxation index in [0, 1].
    """
    # Slow tempo is relaxing: peak relaxation around 60 BPM
    tempo_score = float(np.exp(-0.001 * (tempo - 60.0) ** 2))
    # High consonance is relaxing
    consonance_score = _clamp(consonance)
    # Low spectral centroid (warm timbre) is relaxing
    centroid_score = _clamp(1.0 - spectral_centroid / 8000.0)
    # Stable dynamics are relaxing
    stability_score = _clamp(dynamic_stability)

    weights = np.array([0.30, 0.25, 0.20, 0.25])
    scores = np.array([tempo_score, consonance_score, centroid_score, stability_score])
    return _clamp(float(weights @ scores))


def arousal_trajectory(
    feature_sequence: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Time-varying arousal from a feature trajectory.

    Computes a frame-level arousal signal by a weighted combination of
    normalised features, smoothed with a causal moving-average window.

    Parameters
    ----------
    feature_sequence : np.ndarray
        Shape ``(n_frames, n_features)``.  Columns correspond to:
        [tempo_proxy, loudness, onset_density, spectral_centroid, ...].
        Any additional columns are averaged with equal weight.
    window_size : int
        Smoothing window length in frames (causal / trailing average).

    Returns
    -------
    np.ndarray
        Shape ``(n_frames,)`` arousal values in [0, 1].
    """
    if feature_sequence.ndim == 1:
        feature_sequence = feature_sequence.reshape(-1, 1)
    n_frames, n_feat = feature_sequence.shape

    # Min-max normalise each feature column
    mins = feature_sequence.min(axis=0)
    maxs = feature_sequence.max(axis=0)
    span = maxs - mins
    span[span < 1e-12] = 1.0
    normed = (feature_sequence - mins) / span

    # Predefined weights for the first four features; uniform for extras
    base_weights = np.array([0.25, 0.30, 0.25, 0.20])
    if n_feat <= len(base_weights):
        w = base_weights[:n_feat]
    else:
        extra = np.full(n_feat - len(base_weights), 1.0 / n_feat)
        w = np.concatenate([base_weights, extra])
    w = w / w.sum()

    raw_arousal = normed @ w

    # Causal moving-average smoothing
    kernel = np.ones(min(window_size, n_frames)) / min(window_size, n_frames)
    smoothed = np.convolve(raw_arousal, kernel, mode="full")[:n_frames]
    return np.clip(smoothed, 0.0, 1.0)


def valence_from_mode_and_tempo(
    mode_score: float,
    tempo_bpm: float,
) -> float:
    """Simple valence model from mode and tempo.

    Following Gagnon & Peretz (2003): major mode and fast tempo
    contribute independently to perceived positive valence.

    Parameters
    ----------
    mode_score : float
        Continuous mode indicator: 1.0 = major, 0.0 = minor,
        intermediate values for ambiguous passages.
    tempo_bpm : float
        Tempo in BPM.

    Returns
    -------
    float
        Predicted valence in [0, 1].
    """
    mode_component = 0.60 * _clamp(mode_score)
    tempo_component = 0.40 * _clamp((tempo_bpm - 40.0) / (200.0 - 40.0))
    return _clamp(mode_component + tempo_component)


def entrainment_strength(
    stimulus_period_ms: float,
    response_period_ms: float,
    n_cycles: int,
) -> float:
    """Circular-statistics R-value for sensorimotor entrainment.

    Measures how tightly a periodic physiological response (e.g. tapping,
    breathing, or HR variability) locks to a periodic musical stimulus.

    Parameters
    ----------
    stimulus_period_ms : float
        Period of the musical pulse (ms).
    response_period_ms : float
        Mean period of the physiological response (ms).
    n_cycles : int
        Number of response cycles observed.

    Returns
    -------
    float
        Resultant vector length R in [0, 1].
        1.0 = perfect phase locking; 0.0 = uniformly distributed phases.
    """
    if n_cycles < 1 or stimulus_period_ms <= 0.0:
        return 0.0
    # Phase of each response cycle relative to the stimulus
    rng = np.random.default_rng(42)  # deterministic for reproducibility
    # Simulate phase offsets: base jitter + ratio deviation
    ratio = response_period_ms / stimulus_period_ms
    # Fractional part of the cumulative phase at each cycle
    cycle_phases = np.mod(np.arange(1, n_cycles + 1) * ratio, 1.0)
    angles = 2.0 * np.pi * cycle_phases
    # Resultant vector length (circular statistics)
    r = float(np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2))
    return _clamp(r)


def autonomic_response_model(
    tempo: float,
    loudness_trajectory: np.ndarray,
    consonance_trajectory: np.ndarray,
    onset_density: float,
) -> PhysiologicalResponse:
    """Full autonomic nervous system response prediction.

    Integrates tempo, loudness dynamics, harmonic consonance, and event
    rate into a multi-channel physiological response estimate.

    Parameters
    ----------
    tempo : float
        Tempo in BPM.
    loudness_trajectory : np.ndarray
        Frame-level loudness in dBFS, shape ``(n_frames,)``.
    consonance_trajectory : np.ndarray
        Frame-level consonance in [0, 1], shape ``(n_frames,)``.
    onset_density : float
        Mean note onsets per second.

    Returns
    -------
    PhysiologicalResponse
        Predicted multi-channel ANS response.
    """
    mean_loudness = float(np.mean(loudness_trajectory))
    std_loudness = float(np.std(loudness_trajectory))
    mean_consonance = float(np.mean(consonance_trajectory))

    # Heart rate: tempo and loudness
    hr_change = predict_heart_rate_change(tempo, mean_loudness, mode_major=True)

    # Skin conductance: onset density, dynamic range, surprise proxy
    dynamic_range = float(np.ptp(loudness_trajectory))
    surprise_proxy = std_loudness / max(abs(mean_loudness), 1.0)
    scr = predict_skin_conductance(onset_density, dynamic_range, surprise_proxy * 5.0)

    # Respiration: tempo entrainment proxy
    resp_change = 2.0 * (predict_respiration_sync(tempo, 0.7) - 0.5)

    # Muscle tension: dissonance-driven (inverse of consonance)
    dissonance = 1.0 - mean_consonance
    mapping = MusicPhysioMapping()
    muscle = mapping.dissonance_tension_coeff * dissonance * 10.0  # micro-volts scale

    # Pupil dilation: arousal composite
    arousal_level = _clamp(
        0.3 * _clamp((tempo - 60) / 140.0)
        + 0.3 * _clamp((mean_loudness + 60) / 60.0)
        + 0.2 * _clamp(onset_density / 10.0)
        + 0.2 * (1.0 - mean_consonance)
    )
    pupil = 0.5 * arousal_level  # max ~0.5 mm dilation

    return PhysiologicalResponse(
        heart_rate_change=round(hr_change, 3),
        skin_conductance=round(scr, 3),
        respiration_rate=round(resp_change, 3),
        muscle_tension=round(muscle, 3),
        pupil_dilation=round(pupil, 3),
    )


def stress_reduction_score(
    pre_features: dict[str, float],
    during_features: dict[str, float],
) -> float:
    """Quantify stress reduction via a music intervention.

    Compares arousal-related feature values before and during a music
    intervention.  A positive score indicates the music reduced
    physiological stress markers; negative indicates exacerbation.

    Expected keys (all optional, missing keys are skipped):
      ``tempo``, ``loudness``, ``onset_density``, ``spectral_centroid``,
      ``dissonance``

    Parameters
    ----------
    pre_features : dict[str, float]
        Feature values representing the pre-intervention (stressful) state.
    during_features : dict[str, float]
        Feature values during the music intervention.

    Returns
    -------
    float
        Stress reduction score in [-1, 1].  Positive = reduction.
    """
    # Arousal-linked features: lower during intervention means stress reduction
    arousal_keys = ["tempo", "loudness", "onset_density", "spectral_centroid", "dissonance"]
    # Signs: positive weight means "lower is more relaxing"
    weights = {"tempo": 0.25, "loudness": 0.25, "onset_density": 0.20,
               "spectral_centroid": 0.15, "dissonance": 0.15}
    total_weight = 0.0
    score = 0.0
    for key in arousal_keys:
        if key in pre_features and key in during_features:
            pre_val = pre_features[key]
            dur_val = during_features[key]
            denom = max(abs(pre_val), 1e-6)
            reduction_frac = (pre_val - dur_val) / denom
            score += weights[key] * float(np.tanh(reduction_frac))
            total_weight += weights[key]
    if total_weight < 1e-9:
        return 0.0
    return _clamp(score / total_weight, -1.0, 1.0)


__all__ = [
    "PhysiologicalResponse",
    "MusicPhysioMapping",
    "predict_heart_rate_change",
    "predict_skin_conductance",
    "predict_respiration_sync",
    "chills_probability",
    "relaxation_index",
    "arousal_trajectory",
    "valence_from_mode_and_tempo",
    "entrainment_strength",
    "autonomic_response_model",
    "stress_reduction_score",
]
