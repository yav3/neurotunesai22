"""Musical call-and-response pattern modeling."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class TransformationType(Enum):
    EXACT_REPEAT = "exact_repeat"
    TRANSPOSITION = "transposition"
    INVERSION = "inversion"
    AUGMENTATION = "augmentation"
    DIMINUTION = "diminution"
    VARIATION = "variation"
    CONTRAST = "contrast"


@dataclass
class CallResponsePair:
    call_pitches: list[float]
    call_durations: list[float]
    response_pitches: list[float]
    response_durations: list[float]
    latency_ms: float = 0.0
    transformation: TransformationType = TransformationType.VARIATION


def _contour(pitches: list[float]) -> list[str]:
    if len(pitches) < 2:
        return []
    return [
        "U" if pitches[i + 1] > pitches[i] + 0.5
        else "D" if pitches[i + 1] < pitches[i] - 0.5
        else "R"
        for i in range(len(pitches) - 1)
    ]


def _interval_vector(pitches: list[float]) -> np.ndarray:
    if len(pitches) < 2:
        return np.array([])
    return np.diff(pitches)


def classify_response(pair: CallResponsePair) -> TransformationType:
    cp = np.array(pair.call_pitches)
    rp = np.array(pair.response_pitches)
    cd = np.array(pair.call_durations)
    rd = np.array(pair.response_durations)

    if len(cp) == 0 or len(rp) == 0:
        return TransformationType.CONTRAST

    n = min(len(cp), len(rp))
    pitch_diff = rp[:n] - cp[:n]

    if np.allclose(pitch_diff, 0, atol=0.5) and len(cd) == len(rd) and np.allclose(cd, rd, rtol=0.1):
        return TransformationType.EXACT_REPEAT

    if np.std(pitch_diff[:n]) < 0.5 and abs(np.mean(pitch_diff[:n])) > 0.5:
        return TransformationType.TRANSPOSITION

    iv_call = _interval_vector(pair.call_pitches)
    iv_resp = _interval_vector(pair.response_pitches)
    if len(iv_call) > 0 and len(iv_resp) > 0:
        m = min(len(iv_call), len(iv_resp))
        if np.allclose(iv_call[:m], -iv_resp[:m], atol=1.0):
            return TransformationType.INVERSION

    if len(cd) > 0 and len(rd) > 0:
        ratio = np.mean(rd[:n]) / max(np.mean(cd[:n]), 1e-10)
        if ratio > 1.5:
            return TransformationType.AUGMENTATION
        if ratio < 0.67:
            return TransformationType.DIMINUTION

    contour_c = _contour(pair.call_pitches)
    contour_r = _contour(pair.response_pitches)
    if contour_c and contour_r:
        m = min(len(contour_c), len(contour_r))
        match_rate = sum(1 for i in range(m) if contour_c[i] == contour_r[i]) / m
        if match_rate > 0.5:
            return TransformationType.VARIATION

    return TransformationType.CONTRAST


def generate_response_constraints(
    call_pitches: list[float], transform: TransformationType,
) -> dict:
    if not call_pitches:
        return {"pitch_range": (60, 72), "contour": []}
    lo, hi = min(call_pitches), max(call_pitches)
    contour = _contour(call_pitches)

    if transform == TransformationType.EXACT_REPEAT:
        return {"pitch_range": (lo, hi), "contour": contour, "transposition": 0}
    if transform == TransformationType.TRANSPOSITION:
        return {"pitch_range": (lo + 5, hi + 5), "contour": contour, "transposition": 5}
    if transform == TransformationType.INVERSION:
        inv_contour = [{"U": "D", "D": "U", "R": "R"}[c] for c in contour]
        return {"pitch_range": (lo, hi), "contour": inv_contour}
    if transform == TransformationType.AUGMENTATION:
        return {"pitch_range": (lo, hi), "contour": contour, "duration_scale": 2.0}
    if transform == TransformationType.DIMINUTION:
        return {"pitch_range": (lo, hi), "contour": contour, "duration_scale": 0.5}
    if transform == TransformationType.CONTRAST:
        return {"pitch_range": (lo - 12, hi + 12), "contour": []}
    return {"pitch_range": (lo - 4, hi + 4), "contour": contour}


def dialogue_score(pairs: list[CallResponsePair]) -> float:
    if not pairs:
        return 0.0
    transforms = [classify_response(p) for p in pairs]
    unique_types = len(set(transforms))
    variety = min(1.0, unique_types / max(len(TransformationType), 1))
    coherence_scores = []
    for p in pairs:
        contour_c = _contour(p.call_pitches)
        contour_r = _contour(p.response_pitches)
        if contour_c and contour_r:
            m = min(len(contour_c), len(contour_r))
            coherence_scores.append(sum(1 for i in range(m) if contour_c[i] == contour_r[i]) / m)
    coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.5
    development = 0.0
    for i in range(1, len(transforms)):
        if transforms[i] != transforms[i - 1]:
            development += 1
    development = development / max(len(transforms) - 1, 1)
    return 0.4 * variety + 0.3 * coherence + 0.3 * development
