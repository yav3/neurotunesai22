"""Music data validation and quality assessment.

Schema enforcement, range checks, consistency validation, and
data quality scoring for music feature datasets.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ValidationResult:
    """Outcome of a validation check."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_audio_features(features: dict[str, float]) -> ValidationResult:
    """Validate that audio features fall within physically plausible ranges."""
    result = ValidationResult()
    checks = {
        "tempo_bpm": (20.0, 300.0),
        "spectral_centroid_hz": (0.0, 22050.0),
        "rms_energy": (0.0, 1.0),
        "zero_crossing_rate": (0.0, 1.0),
        "spectral_rolloff_hz": (0.0, 22050.0),
        "mfcc_0": (-100.0, 100.0),
    }
    for key, (lo, hi) in checks.items():
        if key in features:
            v = features[key]
            if not np.isfinite(v):
                result.valid = False
                result.errors.append(f"{key} is not finite: {v}")
            elif v < lo or v > hi:
                result.valid = False
                result.errors.append(f"{key}={v} outside [{lo}, {hi}]")
    return result


def validate_pitch_sequence(midi_notes: np.ndarray) -> ValidationResult:
    """Validate MIDI note sequence for plausible musical content."""
    result = ValidationResult()
    notes = np.asarray(midi_notes, dtype=float)

    if len(notes) == 0:
        result.warnings.append("Empty pitch sequence")
        return result

    if np.any(~np.isfinite(notes)):
        result.valid = False
        result.errors.append("Non-finite values in pitch sequence")
        return result

    valid_notes = notes[(notes >= 0) & (notes <= 127)]
    if len(valid_notes) < len(notes):
        n_invalid = len(notes) - len(valid_notes)
        result.warnings.append(f"{n_invalid} notes outside MIDI range [0, 127]")

    if len(valid_notes) >= 2:
        intervals = np.abs(np.diff(valid_notes))
        if np.max(intervals) > 24:
            result.warnings.append(f"Interval > 2 octaves: {np.max(intervals)} semitones")

    return result


def validate_onset_times(onsets: np.ndarray, sr: int = 44100) -> ValidationResult:
    """Validate onset times for monotonicity and plausible density."""
    result = ValidationResult()
    t = np.asarray(onsets, dtype=float)

    if len(t) == 0:
        return result

    if not np.all(np.diff(t) >= 0):
        result.valid = False
        result.errors.append("Onset times not monotonically non-decreasing")

    if len(t) >= 2:
        ioi = np.diff(t)
        min_ioi = ioi[ioi > 0].min() if np.any(ioi > 0) else 0
        if min_ioi > 0 and min_ioi < 0.02:
            result.warnings.append(f"Very short IOI: {min_ioi:.4f}s (< 20ms)")

    return result


# ── Dataset quality ──────────────────────────────────────────────────

def completeness_score(data: dict[str, Any], required_keys: list[str]) -> float:
    """Fraction of required keys present and non-None."""
    if not required_keys:
        return 1.0
    present = sum(1 for k in required_keys if k in data and data[k] is not None)
    return present / len(required_keys)


def feature_matrix_quality(X: np.ndarray) -> dict[str, float]:
    """Assess quality of a feature matrix (samples × features).

    Returns:
        nan_fraction: fraction of NaN entries
        inf_fraction: fraction of Inf entries
        zero_fraction: fraction of zero entries
        variance_ratio: ratio of min to max feature variance (1.0 = all equal)
        condition_number: condition number of the covariance matrix (log10)
    """
    X = np.asarray(X, dtype=float)
    n_elements = max(X.size, 1)

    nan_frac = float(np.isnan(X).sum() / n_elements)
    inf_frac = float(np.isinf(X).sum() / n_elements)
    zero_frac = float((X == 0).sum() / n_elements)

    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    variances = np.var(X_clean, axis=0)
    var_max = variances.max() if len(variances) > 0 else 1.0
    var_min = variances.min() if len(variances) > 0 else 1.0
    var_ratio = float(var_min / max(var_max, 1e-12))

    if X_clean.shape[0] > X_clean.shape[1] and X_clean.shape[1] > 0:
        cov = np.cov(X_clean.T)
        if cov.ndim == 0:
            cond = 1.0
        else:
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = eigvals[eigvals > 1e-12]
            cond = float(np.log10(eigvals.max() / eigvals.min())) if len(eigvals) > 1 else 0.0
    else:
        cond = 0.0

    return {
        "nan_fraction": nan_frac,
        "inf_fraction": inf_frac,
        "zero_fraction": zero_frac,
        "variance_ratio": var_ratio,
        "condition_number": cond,
    }


def detect_duplicates(ids: list[str]) -> list[str]:
    """Find duplicate IDs in a dataset."""
    seen: set[str] = set()
    dupes: list[str] = []
    for x in ids:
        if x in seen:
            dupes.append(x)
        seen.add(x)
    return dupes


def detect_outliers_iqr(values: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Boolean mask of outliers using the IQR method."""
    v = np.asarray(values, dtype=float)
    if len(v) < 4:
        return np.zeros(len(v), dtype=bool)
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return (v < q1 - factor * iqr) | (v > q3 + factor * iqr)
