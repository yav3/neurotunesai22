"""Emotional Impact Factor — exact solver grounded in the patent specification.

Two computations:
  eif_from_parameters: n>12 weighted parameters → (valence, arousal, magnitude)
  eif_spreadsheet: the NeuroTunes product scorer from audio features
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PARAMETER_GROUPS = ("anthropological", "ethnographic", "psycho_social")
MIN_PARAMETERS = 13

AUDIENCE_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "general": {"anthropological": (0.3, 0.3), "ethnographic": (0.3, 0.3), "psycho_social": (0.4, 0.4)},
    "youth": {"anthropological": (0.2, 0.4), "ethnographic": (0.3, 0.3), "psycho_social": (0.5, 0.3)},
    "clinical": {"anthropological": (0.3, 0.2), "ethnographic": (0.2, 0.2), "psycho_social": (0.5, 0.6)},
}

BPM_CEILING = (46.0, 186.0)


@dataclass(frozen=True)
class EIFParameter:
    name: str
    group: str
    value: float
    valence_loading: float = 1.0
    arousal_loading: float = 0.0


@dataclass(frozen=True)
class EIF:
    valence: float
    arousal: float
    magnitude: float
    audience: str
    n_parameters: int
    group_contributions: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class SpreadsheetEIF:
    eif_arousal: int
    eif_valence: int
    eif_patient: int
    eif_total: int
    therapeutic_mode: str
    mood_tier: str
    in_bpm_ceiling: bool
    dominance: float


def eif_from_parameters(parameters: list[EIFParameter], audience: str = "general") -> EIF:
    if len(parameters) < MIN_PARAMETERS:
        raise ValueError(f"EIF requires n > 12 parameters; got {len(parameters)}")
    if audience not in AUDIENCE_PROFILES:
        raise ValueError(f"unknown audience {audience!r}")
    prof = AUDIENCE_PROFILES[audience]
    contrib: dict[str, list] = {g: [0.0, 0.0, 0] for g in PARAMETER_GROUPS}
    for p in parameters:
        if p.group not in PARAMETER_GROUPS:
            raise ValueError(f"parameter group must be one of {PARAMETER_GROUPS}")
        if not -1.0 <= p.value <= 1.0:
            raise ValueError("parameter values must be in [-1, 1]")
        contrib[p.group][0] += p.value * p.valence_loading
        contrib[p.group][1] += p.value * p.arousal_loading
        contrib[p.group][2] += 1
    v = a = 0.0
    gc: dict[str, tuple[float, float]] = {}
    for g, (sv, sa, n) in contrib.items():
        if n == 0:
            continue
        wv, wa = prof[g]
        gv, ga = wv * sv / n, wa * sa / n
        v += gv
        a += ga
        gc[g] = (gv, ga)
    v = float(np.clip(v, -1, 1))
    a = float(np.clip(a, -1, 1))
    return EIF(valence=v, arousal=a, magnitude=float(np.hypot(v, a)),
               audience=audience, n_parameters=len(parameters), group_contributions=gc)


def _patient_weight(aro: int, val: int) -> int:
    if val >= 8 and aro <= 5:
        return 20
    if val >= 7 and aro <= 3:
        return 20
    if val >= 7 and aro <= 6:
        return 10
    if val >= 5 and aro <= 4:
        return 10
    ratio = val / aro if aro > 0 else float(val)
    if ratio >= 1.6:
        return 10
    if val >= 5 and aro <= 7:
        return 5
    if val >= 8:
        return 5
    if val >= 3 and aro <= 6:
        return 2
    if aro >= 7 and val < 5:
        return 1
    return 0


def _total(patient: int, aro: int, val: int) -> int:
    if patient == 0:
        return 0
    if aro == 0:
        return patient * val
    ratio = val / aro
    if ratio >= 1.6 * 1.6:
        mult = round(ratio)
    elif ratio >= 1.6:
        mult = int(ratio) + 1
    else:
        mult = int(ratio)
    return patient * max(0, mult)


def mood_tier(v: float, a: float, d: float) -> str:
    if v >= 1.6 * 1.6 * a:
        return "deep_regeneration"
    if v >= 1.6 * d:
        return "sleep_gating"
    if v >= 1.6 * a:
        return "high_valence"
    return "standard"


def therapeutic_mode(eif_arousal: int, eif_valence: int) -> str:
    if eif_arousal <= 2 and eif_valence >= 6:
        return "sleep"
    if eif_arousal <= 4 and eif_valence >= 5:
        return "recovery"
    if eif_arousal >= 8 and eif_valence >= 7:
        return "energy"
    if 5 <= eif_arousal <= 7 and eif_valence >= 6:
        return "focus"
    if eif_valence >= 6 and eif_arousal < 6:
        return "calm"
    if eif_arousal < 4 and eif_valence < 5:
        return "depression_support"
    return "focus"


def eif_spreadsheet(energy: float, valence: float, tempo: float,
                    loudness_db: float, acousticness: float) -> SpreadsheetEIF:
    aro = round(min(1.0, max(0.0, energy)) * 10)
    val = round(min(1.0, max(0.0, valence)) * 10)
    loud = min(1.0, max(0.0, (loudness_db + 60) / 60))
    dom = (1 - acousticness) * 0.6 + loud * 0.4
    patient = _patient_weight(aro, val)
    return SpreadsheetEIF(
        eif_arousal=aro, eif_valence=val, eif_patient=patient,
        eif_total=_total(patient, aro, val),
        therapeutic_mode=therapeutic_mode(aro, val),
        mood_tier=mood_tier(valence, energy, dom),
        in_bpm_ceiling=BPM_CEILING[0] <= tempo <= BPM_CEILING[1],
        dominance=float(dom),
    )


def veracity(parameters: list[EIFParameter], audience: str = "general",
             threshold: float = 0.6, reference_valence: float = 0.0,
             reference_arousal: float = 0.0) -> dict:
    e = eif_from_parameters(parameters, audience)
    dist = float(np.hypot(e.valence - reference_valence, e.arousal - reference_arousal))
    falseness = min(1.0, dist / (2 * np.sqrt(2)) * 2.0)
    flagged = falseness > threshold
    return {"eif": e, "falseness": falseness, "flagged": flagged, "threshold": threshold}


def recommend(user_eif: EIF, items: list[tuple[str, EIF]], limit: int = 10) -> list[dict]:
    ranked = sorted(items, key=lambda it: float(np.hypot(
        it[1].valence - user_eif.valence, it[1].arousal - user_eif.arousal)))
    return [{"id": i, "distance": float(np.hypot(
        e.valence - user_eif.valence, e.arousal - user_eif.arousal))} for i, e in ranked[:limit]]
