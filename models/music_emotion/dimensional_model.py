"""Deep dimensional emotion models for music understanding.

Formalizes the three principal continuous-space emotion models used in
music information retrieval and affective computing:

1. **Russell's Circumplex** (Russell, 1980): 2-D valence × arousal circle.
   Emotion words lie on a ring; position encodes affect, angular distance
   encodes dissimilarity. Extended with Posner et al. (2005) density model.

2. **Thayer's Energy–Stress** (Thayer, 1989): 45° rotation of the VA plane
   yielding energy (calm–energetic) and stress (relaxed–tense) axes.

3. **VAD Space** (Mehrabian, 1996): 3-D valence–arousal–dominance cube.
   Dominance captures perceived control/power — critical for distinguishing
   anger (high D) from fear (low D) at similar V–A coordinates.

Mathematical components:

- Circumplex geometry: angular embedding, Procrustes alignment, geodesic
  distance on the affective circle.
- Emotion-to-feature regression: multivariate ridge regression mapping
  acoustic features → VAD, with optional Bayesian posterior (conjugate
  normal-inverse-Wishart).
- Feature-to-emotion inverse: given a target VAD point, solve for the
  feature vector that minimises ‖f(x) − y‖² + λ‖x‖² via pseudo-inverse.
- Trajectory analysis: emotion path in VAD space, arc-length, curvature,
  torsion, Frenet–Serret frame, emotional "velocity" and "acceleration".
- GEMS integration: 9-factor Geneva model projected onto VA/VAD axes via
  empirical loadings (Zentner et al., 2008).
- Plutchik wheel: 8 primary emotions placed on the circumplex with
  intensity rings (Plutchik, 1980, 2001).

Pure numpy. No external ML framework dependencies.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ============================================================
# Constants & emotion lexicons
# ============================================================

PLUTCHIK_PRIMARY = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

PLUTCHIK_SECONDARY = {
    ("joy", "trust"): "love",
    ("trust", "fear"): "submission",
    ("fear", "surprise"): "awe",
    ("surprise", "sadness"): "disapproval",
    ("sadness", "disgust"): "remorse",
    ("disgust", "anger"): "contempt",
    ("anger", "anticipation"): "aggressiveness",
    ("anticipation", "joy"): "optimism",
}

PLUTCHIK_TERTIARY = {
    ("joy", "fear"): "guilt",
    ("trust", "surprise"): "curiosity",
    ("fear", "sadness"): "despair",
    ("surprise", "disgust"): "unbelief",
    ("sadness", "anger"): "envy",
    ("disgust", "anticipation"): "cynicism",
    ("anger", "joy"): "pride",
    ("anticipation", "trust"): "fatalism",
}

PLUTCHIK_INTENSITIES = {
    "joy":          ("serenity",    "joy",          "ecstasy"),
    "trust":        ("acceptance",  "trust",        "admiration"),
    "fear":         ("apprehension","fear",          "terror"),
    "surprise":     ("distraction", "surprise",      "amazement"),
    "sadness":      ("pensiveness", "sadness",       "grief"),
    "disgust":      ("boredom",     "disgust",       "loathing"),
    "anger":        ("annoyance",   "anger",         "rage"),
    "anticipation": ("interest",    "anticipation",  "vigilance"),
}

PLUTCHIK_VAD = {
    "joy":          np.array([ 0.90,  0.70,  0.70]),
    "trust":        np.array([ 0.60,  0.20,  0.40]),
    "fear":         np.array([-0.70,  0.80, -0.80]),
    "surprise":     np.array([ 0.20,  0.90,  0.10]),
    "sadness":      np.array([-0.80, -0.50, -0.40]),
    "disgust":      np.array([-0.60,  0.30,  0.30]),
    "anger":        np.array([-0.50,  0.80,  0.60]),
    "anticipation": np.array([ 0.40,  0.60,  0.50]),
}

RUSSELL_AFFECT_WORDS = {
    "happy":        ( 0.85,  0.45),
    "excited":      ( 0.60,  0.85),
    "alert":        ( 0.25,  0.90),
    "tense":        (-0.30,  0.80),
    "nervous":      (-0.55,  0.70),
    "stressed":     (-0.65,  0.60),
    "sad":          (-0.80, -0.40),
    "depressed":    (-0.85, -0.55),
    "bored":        (-0.40, -0.65),
    "fatigued":     (-0.30, -0.75),
    "calm":         ( 0.30, -0.60),
    "relaxed":      ( 0.55, -0.50),
    "serene":       ( 0.65, -0.35),
    "content":      ( 0.75, -0.10),
    "elated":       ( 0.80,  0.65),
    "miserable":    (-0.90, -0.20),
}

GEMS_FACTORS = [
    "wonder", "transcendence", "tenderness", "nostalgia", "peacefulness",
    "power", "joyful_activation", "tension", "sadness",
]

GEMS_VA_LOADINGS = np.array([
    [ 0.55,  0.45],   # wonder
    [ 0.60,  0.30],   # transcendence
    [ 0.65, -0.25],   # tenderness
    [-0.15, -0.40],   # nostalgia
    [ 0.45, -0.65],   # peacefulness
    [ 0.10,  0.80],   # power
    [ 0.80,  0.70],   # joyful activation
    [-0.55,  0.70],   # tension
    [-0.75, -0.35],   # sadness
])

THAYER_ROTATION = np.pi / 4  # 45° rotation from VA → energy-stress


# ============================================================
# Data classes
# ============================================================

@dataclass
class VADPoint:
    valence: float
    arousal: float
    dominance: float

    def as_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VADPoint":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def norm(self) -> float:
        return float(np.linalg.norm(self.as_array()))


@dataclass
class CircumplexPoint:
    valence: float
    arousal: float

    @property
    def angle(self) -> float:
        return float(np.arctan2(self.arousal, self.valence))

    @property
    def radius(self) -> float:
        return float(np.sqrt(self.valence ** 2 + self.arousal ** 2))

    @classmethod
    def from_polar(cls, angle: float, radius: float) -> "CircumplexPoint":
        return cls(radius * np.cos(angle), radius * np.sin(angle))


@dataclass
class ThayerPoint:
    energy: float
    stress: float

    @classmethod
    def from_va(cls, valence: float, arousal: float) -> "ThayerPoint":
        c, s = np.cos(THAYER_ROTATION), np.sin(THAYER_ROTATION)
        energy = c * arousal + s * valence
        stress = -s * arousal + c * valence  # note sign: stress ≈ -V rotated
        return cls(float(energy), float(-stress))  # flip so tense = +stress

    def to_va(self) -> Tuple[float, float]:
        c, s = np.cos(THAYER_ROTATION), np.sin(THAYER_ROTATION)
        neg_stress = -self.stress
        arousal = c * self.energy - s * neg_stress
        valence = s * self.energy + c * neg_stress
        return float(valence), float(arousal)


@dataclass
class EmotionLabel:
    name: str
    intensity: float  # 0 = mild, 0.5 = moderate, 1 = intense
    confidence: float  # 0–1


@dataclass
class TrajectoryFrame:
    """Frenet–Serret frame at one point of an emotion trajectory."""
    position: np.ndarray   # (3,) VAD
    tangent: np.ndarray    # (3,) unit tangent T
    normal: np.ndarray     # (3,) principal normal N
    binormal: np.ndarray   # (3,) binormal B = T × N
    curvature: float       # κ
    torsion: float         # τ
    speed: float           # |dr/dt|


@dataclass
class RidgeModel:
    W: np.ndarray        # (d_out, d_in) weight matrix
    b: np.ndarray        # (d_out,) bias
    lambda_: float       # regularisation strength
    residual_var: float  # mean squared residual on training data


# ============================================================
# Russell Circumplex geometry
# ============================================================

def circumplex_angle(valence: float, arousal: float) -> float:
    """Angle on the Russell circumplex (radians, 0 = positive valence)."""
    return float(np.arctan2(arousal, valence))


def circumplex_angular_distance(a1: float, a2: float) -> float:
    """Geodesic angular distance on the unit circle."""
    d = abs(a1 - a2)
    return float(min(d, 2 * np.pi - d))


def circumplex_chord_distance(p1: CircumplexPoint, p2: CircumplexPoint) -> float:
    """Euclidean distance in VA space."""
    return float(np.sqrt((p1.valence - p2.valence) ** 2 +
                         (p1.arousal - p2.arousal) ** 2))


def project_to_circumplex(valence: float, arousal: float,
                          radius: float = 1.0) -> CircumplexPoint:
    """Project a VA point onto the circumplex ring of given radius."""
    r = np.sqrt(valence ** 2 + arousal ** 2)
    if r < 1e-10:
        return CircumplexPoint(radius, 0.0)
    scale = radius / r
    return CircumplexPoint(float(valence * scale), float(arousal * scale))


def circumplex_density(valence: float, arousal: float,
                       bandwidth: float = 0.3) -> float:
    """KDE density estimate on the circumplex from Russell's affect words.

    Higher density = more affect words cluster near this point (Posner et al., 2005).
    """
    total = 0.0
    for v, a in RUSSELL_AFFECT_WORDS.values():
        d2 = (valence - v) ** 2 + (arousal - a) ** 2
        total += np.exp(-d2 / (2 * bandwidth ** 2))
    return float(total / len(RUSSELL_AFFECT_WORDS))


def procrustes_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """Orthogonal Procrustes alignment of two point clouds in VA space.

    Finds rotation R minimising ‖source·R − target‖_F (Schönemann, 1966).

    Args:
        source: (N, 2) points in source coordinates
        target: (N, 2) corresponding points in target coordinates

    Returns:
        (R, residual) where R is (2,2) orthogonal and residual = ‖S·R − T‖_F
    """
    M = source.T @ target  # (2, 2)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    residual = float(np.linalg.norm(source @ R - target, 'fro'))
    return R, residual


# ============================================================
# Thayer energy–stress transform
# ============================================================

def va_to_thayer(valence: float, arousal: float) -> ThayerPoint:
    """Convert valence–arousal to Thayer's energy–stress space."""
    return ThayerPoint.from_va(valence, arousal)


def thayer_to_va(energy: float, stress: float) -> Tuple[float, float]:
    """Inverse of va_to_thayer."""
    return ThayerPoint(energy, stress).to_va()


def thayer_quadrant(energy: float, stress: float) -> str:
    """Name the Thayer quadrant."""
    if energy >= 0 and stress >= 0:
        return "high_energy_high_stress"    # tense, nervous
    elif energy >= 0 and stress < 0:
        return "high_energy_low_stress"     # happy, elated
    elif energy < 0 and stress >= 0:
        return "low_energy_high_stress"     # sad, depressed
    else:
        return "low_energy_low_stress"      # calm, relaxed


# ============================================================
# VAD space operations
# ============================================================

def vad_distance(a: np.ndarray, b: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
    """Weighted Euclidean distance in VAD space.

    Default weights: V=1.0, A=0.85, D=0.70 (empirical salience, Warriner et al., 2013).
    """
    if weights is None:
        weights = np.array([1.0, 0.85, 0.70])
    diff = np.asarray(a)[:3] - np.asarray(b)[:3]
    return float(np.sqrt(np.sum(weights * diff ** 2)))


def vad_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two VAD vectors."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def vad_centroid(points: np.ndarray) -> np.ndarray:
    """Centroid of a set of VAD points (N, 3)."""
    return np.mean(points, axis=0)


def vad_dispersion(points: np.ndarray) -> float:
    """Mean distance of points from their centroid."""
    c = vad_centroid(points)
    return float(np.mean(np.linalg.norm(points - c, axis=1)))


def vad_covariance(points: np.ndarray) -> np.ndarray:
    """Covariance matrix (3, 3) of VAD point cloud."""
    return np.cov(points.T) if points.shape[0] > 1 else np.zeros((3, 3))


def vad_mahalanobis(point: np.ndarray, mean: np.ndarray,
                    cov: np.ndarray) -> float:
    """Mahalanobis distance of a point from a distribution in VAD space."""
    diff = np.asarray(point) - np.asarray(mean)
    try:
        cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(3))
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(diff))
    return float(np.sqrt(diff @ cov_inv @ diff))


def nearest_emotion(vad: np.ndarray,
                    lexicon: Optional[Dict[str, np.ndarray]] = None) -> Tuple[str, float]:
    """Find the nearest Plutchik emotion to a VAD point."""
    if lexicon is None:
        lexicon = PLUTCHIK_VAD
    best_name, best_dist = "", np.inf
    vad = np.asarray(vad)
    for name, center in lexicon.items():
        d = float(np.linalg.norm(vad - center))
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, best_dist


def vad_to_plutchik(vad: np.ndarray) -> EmotionLabel:
    """Map a VAD point to a Plutchik emotion with intensity."""
    name, dist = nearest_emotion(vad)
    intensity_raw = float(np.linalg.norm(vad))
    intensity = float(np.clip(intensity_raw / 1.5, 0, 1))
    confidence = float(np.exp(-dist))
    return EmotionLabel(name, intensity, confidence)


def plutchik_dyad(e1: str, e2: str) -> Optional[str]:
    """Look up secondary or tertiary Plutchik dyad."""
    key = (e1, e2)
    rev = (e2, e1)
    if key in PLUTCHIK_SECONDARY:
        return PLUTCHIK_SECONDARY[key]
    if rev in PLUTCHIK_SECONDARY:
        return PLUTCHIK_SECONDARY[rev]
    if key in PLUTCHIK_TERTIARY:
        return PLUTCHIK_TERTIARY[key]
    if rev in PLUTCHIK_TERTIARY:
        return PLUTCHIK_TERTIARY[rev]
    return None


def plutchik_intensity_label(emotion: str, intensity: float) -> str:
    """Map an emotion + intensity ∈ [0,1] to its Plutchik intensity word."""
    if emotion not in PLUTCHIK_INTENSITIES:
        return emotion
    mild, moderate, intense = PLUTCHIK_INTENSITIES[emotion]
    if intensity < 0.33:
        return mild
    elif intensity < 0.67:
        return moderate
    else:
        return intense


# ============================================================
# GEMS projection
# ============================================================

def gems_to_va(gems_scores: np.ndarray) -> CircumplexPoint:
    """Project a 9-D GEMS profile into VA space via empirical loadings.

    gems_scores: (9,) in the order of GEMS_FACTORS.
    """
    scores = np.asarray(gems_scores)
    va = scores @ GEMS_VA_LOADINGS  # (2,)
    return CircumplexPoint(float(va[0]), float(va[1]))


def gems_to_vad(gems_scores: np.ndarray) -> VADPoint:
    """Approximate VAD from GEMS.

    V and A from the loadings; D estimated as (power − tension + joyful_activation) / 3.
    """
    va = gems_scores @ GEMS_VA_LOADINGS
    power = gems_scores[5]
    tension = gems_scores[7]
    joy_act = gems_scores[6]
    dominance = (power - tension + joy_act) / 3.0
    return VADPoint(float(va[0]), float(va[1]), float(np.clip(dominance, -1, 1)))


def va_to_gems_approx(valence: float, arousal: float) -> np.ndarray:
    """Approximate GEMS from VA via pseudo-inverse of the loading matrix.

    Under-determined (2→9), returns the minimum-norm GEMS vector.
    """
    va = np.array([valence, arousal])
    pinv = np.linalg.pinv(GEMS_VA_LOADINGS)  # (2, 9)
    return pinv @ va


# ============================================================
# Audio features → VAD regression
# ============================================================

FEATURE_NAMES = [
    "tempo", "mode", "spectral_centroid", "spectral_bandwidth",
    "spectral_rolloff", "spectral_flatness", "zero_crossing_rate",
    "rms_energy", "mfcc_1", "mfcc_2", "mfcc_3",
    "chroma_mean", "onset_rate", "dynamic_range", "harmonic_ratio",
]

# Prior feature-to-VAD mapping (hand-tuned from literature):
PRIOR_WEIGHTS_V = np.array([
     0.15,  0.30,  0.10,  0.05,  0.05, -0.05, -0.05,
     0.10,  0.15, -0.10,  0.05,  0.10,  0.05,  0.05,  0.10,
])
PRIOR_WEIGHTS_A = np.array([
     0.30,  0.00,  0.15,  0.10,  0.10,  0.05,  0.10,
     0.25,  0.05,  0.05, -0.05, -0.05,  0.20,  0.15,  0.05,
])
PRIOR_WEIGHTS_D = np.array([
     0.20,  0.10,  0.10,  0.05,  0.05, -0.05,  0.00,
     0.25,  0.10,  0.00,  0.00,  0.05,  0.10,  0.20,  0.05,
])

FEATURE_MEANS = np.array([
    120.0, 0.5, 2000.0, 1500.0, 4000.0, 0.1, 0.08,
    0.1, 0.0, 0.0, 0.0, 0.5, 2.0, 30.0, 0.7,
])
FEATURE_STDS = np.array([
    40.0, 0.5, 1000.0, 800.0, 2000.0, 0.1, 0.05,
    0.08, 15.0, 10.0, 8.0, 0.2, 1.5, 15.0, 0.2,
])


def standardise_features(features: np.ndarray) -> np.ndarray:
    """Z-score normalise with literature-derived means/stds."""
    return (np.asarray(features) - FEATURE_MEANS) / (FEATURE_STDS + 1e-10)


def features_to_vad_prior(features: np.ndarray) -> VADPoint:
    """Map raw features → VAD using the hand-tuned prior weights."""
    z = standardise_features(features)
    v = float(np.clip(np.tanh(z @ PRIOR_WEIGHTS_V), -1, 1))
    a = float(np.clip(np.tanh(z @ PRIOR_WEIGHTS_A), -1, 1))
    d = float(np.clip(np.tanh(z @ PRIOR_WEIGHTS_D), -1, 1))
    return VADPoint(v, a, d)


def fit_ridge_vad(X: np.ndarray, Y: np.ndarray,
                  lambda_: float = 1.0) -> RidgeModel:
    """Multivariate ridge regression: X (N, d_in) → Y (N, 3) [V, A, D].

    Closed-form solution: W = (X^T X + λI)^{-1} X^T Y.
    """
    N, d_in = X.shape
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    A = Xc.T @ Xc + lambda_ * np.eye(d_in)
    W = np.linalg.solve(A, Xc.T @ Yc)  # (d_in, 3)

    b = Y_mean - X_mean @ W
    residuals = Yc - Xc @ W
    residual_var = float(np.mean(residuals ** 2))

    return RidgeModel(W=W.T, b=b, lambda_=lambda_, residual_var=residual_var)


def predict_vad_ridge(model: RidgeModel, features: np.ndarray) -> VADPoint:
    """Predict VAD from features using a fitted ridge model."""
    x = np.asarray(features)
    pred = model.W @ x + model.b
    pred = np.clip(pred, -1, 1)
    return VADPoint(float(pred[0]), float(pred[1]), float(pred[2]))


def predict_vad_posterior(model: RidgeModel, features: np.ndarray,
                         n_samples: int = 100,
                         seed: int = 42) -> Tuple[VADPoint, np.ndarray]:
    """Bayesian posterior prediction: mean + covariance of predicted VAD.

    Approximates the posterior by sampling from N(Wx+b, σ²(X^TX+λI)^{-1}).
    Returns (mean_prediction, cov_3x3).
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(features)
    mean = model.W @ x + model.b

    d_in = model.W.shape[1]
    noise_var = model.residual_var
    cov_w = noise_var * np.eye(d_in) / (model.lambda_ + 1e-10)

    samples = np.zeros((n_samples, 3))
    for i in range(n_samples):
        w_noise = rng.multivariate_normal(np.zeros(d_in), cov_w)
        samples[i] = mean + model.W @ w_noise + rng.randn(3) * np.sqrt(noise_var)

    cov = np.cov(samples.T)
    mean_pred = np.clip(np.mean(samples, axis=0), -1, 1)
    return VADPoint.from_array(mean_pred), cov


def invert_vad_to_features(model: RidgeModel, target_vad: np.ndarray,
                           lambda_inv: float = 0.1) -> np.ndarray:
    """Given target VAD, find the feature vector that produces it.

    Solves min_x ‖W·x + b − y‖² + λ_inv·‖x‖² via regularised pseudo-inverse.
    """
    y = np.asarray(target_vad) - model.b
    W = model.W  # (3, d_in)
    A = W.T @ W + lambda_inv * np.eye(W.shape[1])
    return np.linalg.solve(A, W.T @ y)


# ============================================================
# Emotion trajectory analysis in VAD space
# ============================================================

def emotion_velocity(trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """First derivative of VAD trajectory: emotional "velocity".

    trajectory: (T, 3) VAD time series.
    Returns (T-1, 3).
    """
    return np.diff(trajectory, axis=0) / dt


def emotion_acceleration(trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Second derivative: emotional "acceleration" (T-2, 3)."""
    vel = emotion_velocity(trajectory, dt)
    return np.diff(vel, axis=0) / dt


def trajectory_arc_length(trajectory: np.ndarray) -> float:
    """Total arc-length of the emotion path in VAD space."""
    diffs = np.diff(trajectory, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def trajectory_curvature(trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Curvature κ(t) of the 3-D emotion path.

    κ = ‖v × a‖ / ‖v‖³ where v = dr/dt, a = d²r/dt².
    Returns (T-2,).
    """
    v = emotion_velocity(trajectory, dt)
    a = np.diff(v, axis=0) / dt
    T = min(len(v) - 1, len(a))
    kappa = np.zeros(T)
    for i in range(T):
        cross = np.cross(v[i], a[i])
        speed = np.linalg.norm(v[i])
        kappa[i] = np.linalg.norm(cross) / (speed ** 3 + 1e-10)
    return kappa


def trajectory_torsion(trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Torsion τ(t) of the 3-D emotion path.

    τ = (v × a) · a' / ‖v × a‖² where a' = d³r/dt³.
    Returns (T-3,).
    """
    v = emotion_velocity(trajectory, dt)
    a = np.diff(v, axis=0) / dt
    a_prime = np.diff(a, axis=0) / dt

    T = min(len(v) - 2, len(a) - 1, len(a_prime))
    tau = np.zeros(T)
    for i in range(T):
        cross = np.cross(v[i], a[i])
        cross_norm_sq = np.dot(cross, cross)
        if cross_norm_sq < 1e-10:
            tau[i] = 0.0
        else:
            tau[i] = np.dot(cross, a_prime[i]) / cross_norm_sq
    return tau


def frenet_serret_frame(trajectory: np.ndarray,
                        dt: float = 1.0) -> List[TrajectoryFrame]:
    """Compute the Frenet–Serret frame at each interior point of the trajectory."""
    v = emotion_velocity(trajectory, dt)
    a = np.diff(v, axis=0) / dt

    T = min(len(v) - 1, len(a))
    frames = []
    for i in range(T):
        speed = np.linalg.norm(v[i])
        if speed < 1e-10:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = v[i] / speed

        cross = np.cross(v[i], a[i])
        cross_norm = np.linalg.norm(cross)
        kappa = cross_norm / (speed ** 3 + 1e-10)

        if cross_norm < 1e-10:
            normal = np.array([0.0, 1.0, 0.0])
        else:
            binormal_vec = cross / cross_norm
            normal = np.cross(binormal_vec, tangent)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-10:
                normal = normal / normal_norm

        binormal = np.cross(tangent, normal)

        frames.append(TrajectoryFrame(
            position=trajectory[i + 1],
            tangent=tangent,
            normal=normal,
            binormal=binormal,
            curvature=float(kappa),
            torsion=0.0,  # filled below if enough points
            speed=float(speed),
        ))

    if len(trajectory) >= 4:
        tau = trajectory_torsion(trajectory, dt)
        for i in range(min(len(tau), len(frames))):
            frames[i].torsion = float(tau[i])

    return frames


def emotional_surprise(trajectory: np.ndarray, window: int = 5) -> np.ndarray:
    """Surprise = deviation from local expectation (sliding-window mean).

    High surprise ↔ abrupt emotional shift (e.g., a musical surprise chord).
    Returns (T,) surprise values.
    """
    T = len(trajectory)
    surprise = np.zeros(T)
    for i in range(T):
        start = max(0, i - window)
        local_mean = np.mean(trajectory[start:i + 1], axis=0)
        if i + 1 < T:
            surprise[i] = float(np.linalg.norm(trajectory[i + 1] - local_mean))
    return surprise


def emotional_momentum(trajectory: np.ndarray, decay: float = 0.9) -> np.ndarray:
    """Exponential-moving-average momentum of the VAD trajectory.

    Captures the "emotional inertia" — how much the current direction persists.
    Returns (T, 3).
    """
    T = len(trajectory)
    momentum = np.zeros_like(trajectory)
    if T < 2:
        return momentum
    momentum[0] = np.zeros(3)
    for i in range(1, T):
        delta = trajectory[i] - trajectory[i - 1]
        momentum[i] = decay * momentum[i - 1] + (1 - decay) * delta
    return momentum


# ============================================================
# Music-specific feature → emotion mapping
# ============================================================

def tempo_valence_arousal(tempo: float) -> Tuple[float, float]:
    """Heuristic mapping from tempo (BPM) to (valence_bias, arousal_bias).

    Based on: Gabrielsson & Lindström (2001), Webster & Weir (2005).
    """
    a_bias = float(np.tanh((tempo - 110) / 50))
    v_bias = float(0.3 * np.tanh((tempo - 120) / 40))
    return v_bias, a_bias


def mode_valence(is_major: bool) -> float:
    """Major/minor → valence shift (Hevner, 1935; Dalla Bella et al., 2001)."""
    return 0.3 if is_major else -0.3


def spectral_brightness_arousal(centroid: float) -> float:
    """Spectral centroid → arousal (Schubert, 2004)."""
    return float(np.tanh((centroid - 1800) / 1000))


def loudness_dominance(loudness_db: float) -> float:
    """Perceived loudness → dominance (Ilie & Thompson, 2006)."""
    return float(np.tanh((loudness_db + 10) / 15))


def harmonic_complexity_valence(dissonance: float) -> float:
    """Harmonic dissonance → negative valence shift (Helmholtz, 1863; Sethares, 1998)."""
    return float(-0.5 * np.tanh(dissonance / 0.3))


def articulation_arousal(staccato_ratio: float) -> float:
    """Staccato proportion → arousal increase (Juslin & Laukka, 2003)."""
    return float(0.4 * staccato_ratio)


def register_valence(pitch_mean: float) -> float:
    """Mean pitch register → valence (higher ≈ brighter).

    Normalised MIDI pitch 60 = middle C = neutral.
    """
    return float(0.2 * np.tanh((pitch_mean - 60) / 12))


def feature_dict_to_vad(features: Dict) -> VADPoint:
    """Aggregate multiple feature-based heuristics into a single VAD point.

    Combines tempo, mode, spectral, loudness, and harmonic cues
    with weights from Juslin & Laukka (2003) meta-analysis.
    """
    v, a, d = 0.0, 0.0, 0.0

    if "tempo" in features:
        vb, ab = tempo_valence_arousal(features["tempo"])
        v += 0.20 * vb
        a += 0.30 * ab

    if "mode" in features:
        v += 0.25 * mode_valence(features["mode"] > 0.5 if isinstance(features["mode"], float) else features["mode"])

    if "spectral_centroid" in features:
        a += 0.20 * spectral_brightness_arousal(features["spectral_centroid"])
        v += 0.10 * spectral_brightness_arousal(features["spectral_centroid"])

    if "loudness" in features:
        d += 0.30 * loudness_dominance(features["loudness"])
        a += 0.20 * loudness_dominance(features["loudness"])

    if "rms_energy" in features:
        a += 0.15 * float(np.tanh(features["rms_energy"] / 0.1))

    if "dissonance" in features:
        v += 0.15 * harmonic_complexity_valence(features["dissonance"])

    if "pitch_mean" in features:
        v += 0.10 * register_valence(features["pitch_mean"])

    if "staccato_ratio" in features:
        a += 0.10 * articulation_arousal(features["staccato_ratio"])

    if "dynamic_range" in features:
        d += 0.20 * float(np.tanh(features["dynamic_range"] / 30))

    return VADPoint(
        float(np.clip(v, -1, 1)),
        float(np.clip(a, -1, 1)),
        float(np.clip(d, -1, 1)),
    )


# ============================================================
# Emotion interpolation and blending
# ============================================================

def slerp_va(p1: CircumplexPoint, p2: CircumplexPoint, t: float) -> CircumplexPoint:
    """Spherical linear interpolation between two points on the circumplex."""
    a1, a2 = p1.angle, p2.angle
    r1, r2 = p1.radius, p2.radius
    omega = circumplex_angular_distance(a1, a2)

    if omega < 1e-8:
        return CircumplexPoint(
            float((1 - t) * p1.valence + t * p2.valence),
            float((1 - t) * p1.arousal + t * p2.arousal),
        )

    diff = a2 - a1
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi

    angle = a1 + t * diff
    radius = (1 - t) * r1 + t * r2
    return CircumplexPoint.from_polar(angle, radius)


def lerp_vad(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two VAD points."""
    return (1 - t) * np.asarray(a) + t * np.asarray(b)


def blend_emotions(emotions: List[Tuple[str, float]]) -> VADPoint:
    """Blend multiple named Plutchik emotions with weights → single VAD point."""
    vad_sum = np.zeros(3)
    weight_sum = 0.0
    for name, weight in emotions:
        if name in PLUTCHIK_VAD:
            vad_sum += weight * PLUTCHIK_VAD[name]
            weight_sum += weight
    if weight_sum < 1e-10:
        return VADPoint(0.0, 0.0, 0.0)
    result = vad_sum / weight_sum
    return VADPoint.from_array(np.clip(result, -1, 1))


# ============================================================
# Emotion segmentation
# ============================================================

def segment_by_emotion(trajectory: np.ndarray,
                       threshold: float = 0.3) -> List[Tuple[int, int, str]]:
    """Segment an emotion trajectory into contiguous emotional regions.

    A new segment starts when the VAD shift exceeds `threshold`.
    Returns list of (start_idx, end_idx, dominant_emotion).
    """
    segments = []
    start = 0
    for i in range(1, len(trajectory)):
        shift = float(np.linalg.norm(trajectory[i] - trajectory[i - 1]))
        if shift > threshold:
            mid_vad = np.mean(trajectory[start:i], axis=0)
            name, _ = nearest_emotion(mid_vad)
            segments.append((start, i, name))
            start = i
    if start < len(trajectory):
        mid_vad = np.mean(trajectory[start:], axis=0)
        name, _ = nearest_emotion(mid_vad)
        segments.append((start, len(trajectory), name))
    return segments


def emotional_contrast(segments: List[Tuple[int, int, str]]) -> float:
    """Average inter-segment emotional distance (higher = more contrasting piece)."""
    if len(segments) < 2:
        return 0.0
    distances = []
    for i in range(len(segments) - 1):
        e1 = PLUTCHIK_VAD.get(segments[i][2], np.zeros(3))
        e2 = PLUTCHIK_VAD.get(segments[i + 1][2], np.zeros(3))
        distances.append(float(np.linalg.norm(e1 - e2)))
    return float(np.mean(distances))


# ============================================================
# Self-test
# ============================================================

def _self_test():
    """Verify core computations."""
    p = CircumplexPoint(0.5, 0.5)
    assert abs(p.radius - np.sqrt(0.5)) < 1e-6

    tp = ThayerPoint.from_va(0.8, 0.6)
    v, a = tp.to_va()
    assert abs(v - 0.8) < 1e-6 and abs(a - 0.6) < 1e-6

    d = vad_distance(np.array([0, 0, 0]), np.array([1, 1, 1]))
    assert d > 0

    vad = feature_dict_to_vad({"tempo": 140, "mode": True, "loudness": -5})
    assert -1 <= vad.valence <= 1 and -1 <= vad.arousal <= 1

    traj = np.random.randn(20, 3) * 0.3
    traj = np.cumsum(traj, axis=0) * 0.1
    kappa = trajectory_curvature(traj)
    assert len(kappa) == len(traj) - 2

    frames = frenet_serret_frame(traj)
    assert len(frames) > 0
    for f in frames:
        assert abs(np.linalg.norm(f.tangent) - 1.0) < 1e-5

    gems = np.array([0.3, 0.2, 0.5, 0.1, 0.4, 0.6, 0.7, 0.2, 0.1])
    va = gems_to_va(gems)
    assert isinstance(va, CircumplexPoint)

    X = np.random.randn(50, 15)
    Y = np.random.randn(50, 3) * 0.5
    model = fit_ridge_vad(X, Y)
    pred = predict_vad_ridge(model, X[0])
    assert -1 <= pred.valence <= 1

    inv = invert_vad_to_features(model, np.array([0.5, 0.3, 0.1]))
    assert len(inv) == 15

    segs = segment_by_emotion(traj)
    assert len(segs) > 0

    print(f"Self-test passed: circumplex, Thayer, VAD ops, ridge, trajectory, GEMS, segmentation.")


if __name__ == "__main__":
    _self_test()
