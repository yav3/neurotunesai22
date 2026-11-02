"""Statistical classification of music into historical periods using feature vectors.

Uses Mahalanobis distance and Bayesian posterior estimation against period prototypes
defined by harmonic, rhythmic, timbral, and structural feature centroids."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PeriodPrototype:
    """Statistical prototype for a historical music period."""
    name: str
    centroid: np.ndarray
    covariance: np.ndarray
    prior: float = 1.0

    @property
    def n_features(self) -> int:
        return len(self.centroid)


FEATURE_NAMES = [
    "harmonic_complexity",
    "dissonance_ratio",
    "rhythmic_regularity",
    "metric_hierarchy_depth",
    "dynamic_range_db",
    "timbral_brightness",
    "polyphonic_density",
    "form_repetition_rate",
    "modulation_frequency",
    "ornamentation_density",
]


def _default_prototypes() -> Dict[str, PeriodPrototype]:
    """Built-in period prototypes with 10-D feature centroids."""
    dim = len(FEATURE_NAMES)
    base_cov = np.eye(dim) * 0.04

    return {
        "medieval": PeriodPrototype(
            name="medieval",
            centroid=np.array([0.15, 0.10, 0.80, 0.30, 0.25, 0.35, 0.20, 0.60, 0.05, 0.15]),
            covariance=base_cov * 1.2,
            prior=0.08,
        ),
        "renaissance": PeriodPrototype(
            name="renaissance",
            centroid=np.array([0.30, 0.15, 0.70, 0.45, 0.30, 0.40, 0.55, 0.50, 0.10, 0.25]),
            covariance=base_cov,
            prior=0.10,
        ),
        "baroque": PeriodPrototype(
            name="baroque",
            centroid=np.array([0.50, 0.25, 0.75, 0.60, 0.45, 0.50, 0.65, 0.55, 0.30, 0.70]),
            covariance=base_cov,
            prior=0.15,
        ),
        "classical": PeriodPrototype(
            name="classical",
            centroid=np.array([0.45, 0.20, 0.85, 0.70, 0.55, 0.55, 0.50, 0.65, 0.25, 0.30]),
            covariance=base_cov * 0.8,
            prior=0.18,
        ),
        "romantic": PeriodPrototype(
            name="romantic",
            centroid=np.array([0.70, 0.40, 0.60, 0.55, 0.80, 0.60, 0.70, 0.45, 0.55, 0.45]),
            covariance=base_cov * 1.1,
            prior=0.20,
        ),
        "impressionist": PeriodPrototype(
            name="impressionist",
            centroid=np.array([0.65, 0.50, 0.45, 0.40, 0.60, 0.75, 0.55, 0.35, 0.45, 0.35]),
            covariance=base_cov * 1.3,
            prior=0.08,
        ),
        "modern": PeriodPrototype(
            name="modern",
            centroid=np.array([0.80, 0.65, 0.40, 0.50, 0.70, 0.65, 0.60, 0.30, 0.60, 0.20]),
            covariance=base_cov * 1.5,
            prior=0.12,
        ),
        "contemporary": PeriodPrototype(
            name="contemporary",
            centroid=np.array([0.85, 0.70, 0.35, 0.45, 0.85, 0.70, 0.75, 0.25, 0.70, 0.15]),
            covariance=base_cov * 1.8,
            prior=0.09,
        ),
    }


DEFAULT_PROTOTYPES = _default_prototypes()


def mahalanobis_distance(x: np.ndarray, proto: PeriodPrototype) -> float:
    """Mahalanobis distance from x to a period prototype."""
    diff = x - proto.centroid
    try:
        inv_cov = np.linalg.inv(proto.covariance)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(proto.covariance)
    return float(np.sqrt(diff @ inv_cov @ diff))


def log_likelihood(x: np.ndarray, proto: PeriodPrototype) -> float:
    """Log-likelihood of x under a multivariate Gaussian for a period prototype."""
    d = len(x)
    diff = x - proto.centroid
    try:
        sign, logdet = np.linalg.slogdet(proto.covariance)
        inv_cov = np.linalg.inv(proto.covariance)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(proto.covariance)
        logdet = np.log(np.linalg.det(proto.covariance) + 1e-300)
        sign = 1
    maha_sq = float(diff @ inv_cov @ diff)
    return -0.5 * (d * np.log(2 * np.pi) + sign * logdet + maha_sq)


def classify_period(
    features: np.ndarray,
    prototypes: Dict[str, PeriodPrototype] | None = None,
) -> Tuple[str, Dict[str, float]]:
    """Classify a feature vector into a historical period via MAP estimation.

    Returns (predicted_period, posterior_dict).
    """
    protos = prototypes or DEFAULT_PROTOTYPES
    log_posteriors: Dict[str, float] = {}
    for name, proto in protos.items():
        ll = log_likelihood(features, proto)
        log_posteriors[name] = ll + np.log(proto.prior + 1e-12)

    max_lp = max(log_posteriors.values())
    posteriors = {k: np.exp(v - max_lp) for k, v in log_posteriors.items()}
    total = sum(posteriors.values())
    posteriors = {k: v / total for k, v in posteriors.items()}

    best = max(posteriors, key=posteriors.get)  # type: ignore[arg-type]
    return best, posteriors


def period_distances(
    features: np.ndarray,
    prototypes: Dict[str, PeriodPrototype] | None = None,
) -> Dict[str, float]:
    """Mahalanobis distance from a feature vector to each period prototype."""
    protos = prototypes or DEFAULT_PROTOTYPES
    return {name: mahalanobis_distance(features, proto) for name, proto in protos.items()}


def nearest_periods(
    features: np.ndarray,
    k: int = 3,
    prototypes: Dict[str, PeriodPrototype] | None = None,
) -> List[Tuple[str, float]]:
    """Return the k nearest period prototypes by Mahalanobis distance."""
    dists = period_distances(features, prototypes)
    sorted_periods = sorted(dists.items(), key=lambda x: x[1])
    return sorted_periods[:k]


def transition_probability(
    from_period: str, to_period: str, *, year: int = 1800
) -> float:
    """Heuristic transition probability between periods based on historical proximity.

    Uses a simple exponential decay over the distance between period centroids
    in the time dimension.
    """
    period_midpoints = {
        "medieval": 1100, "renaissance": 1500, "baroque": 1680,
        "classical": 1780, "romantic": 1850, "impressionist": 1900,
        "modern": 1950, "contemporary": 2000,
    }
    if from_period not in period_midpoints or to_period not in period_midpoints:
        return 0.0
    t1 = period_midpoints[from_period]
    t2 = period_midpoints[to_period]
    if t2 < t1:
        return 0.01
    dt = abs(t2 - t1)
    return float(np.exp(-dt / 150.0))


def feature_importance(
    features: np.ndarray,
    predicted: str,
    prototypes: Dict[str, PeriodPrototype] | None = None,
) -> np.ndarray:
    """Per-feature contribution to the classification decision.

    Returns |Σ⁻¹(x - μ)| element-wise — larger values indicate more influential features.
    """
    protos = prototypes or DEFAULT_PROTOTYPES
    proto = protos[predicted]
    diff = features - proto.centroid
    try:
        inv_cov = np.linalg.inv(proto.covariance)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(proto.covariance)
    return np.abs(inv_cov @ diff)


def classify_batch(
    feature_matrix: np.ndarray,
    prototypes: Dict[str, PeriodPrototype] | None = None,
) -> List[Tuple[str, Dict[str, float]]]:
    """Classify multiple feature vectors."""
    return [classify_period(row, prototypes) for row in feature_matrix]


def confusion_matrix(
    true_labels: List[str],
    pred_labels: List[str],
    period_names: List[str] | None = None,
) -> np.ndarray:
    """Compute confusion matrix for period classification results."""
    names = period_names or sorted(set(true_labels) | set(pred_labels))
    idx = {n: i for i, n in enumerate(names)}
    mat = np.zeros((len(names), len(names)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1
    return mat
