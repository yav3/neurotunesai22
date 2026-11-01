from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class GEMSFactor(Enum):
    WONDER = "wonder"
    TRANSCENDENCE = "transcendence"
    TENDERNESS = "tenderness"
    NOSTALGIA = "nostalgia"
    PEACEFULNESS = "peacefulness"
    POWER = "power"
    JOYFUL_ACTIVATION = "joyful_activation"
    TENSION = "tension"
    SADNESS = "sadness"


@dataclass
class GEMSProfile:
    scores: dict[GEMSFactor, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for f in GEMSFactor:
            self.scores.setdefault(f, 0.0)


# Each factor's centroid in (valence, arousal, dominance) [0-1] space.
_FACTOR_CENTROIDS_VAD: dict[GEMSFactor, tuple[float, float, float]] = {
    GEMSFactor.WONDER:             (0.75, 0.65, 0.50),
    GEMSFactor.TRANSCENDENCE:      (0.70, 0.45, 0.40),
    GEMSFactor.TENDERNESS:         (0.80, 0.30, 0.45),
    GEMSFactor.NOSTALGIA:          (0.45, 0.30, 0.35),
    GEMSFactor.PEACEFULNESS:       (0.70, 0.15, 0.50),
    GEMSFactor.POWER:              (0.50, 0.85, 0.80),
    GEMSFactor.JOYFUL_ACTIVATION:  (0.90, 0.80, 0.70),
    GEMSFactor.TENSION:            (0.20, 0.80, 0.55),
    GEMSFactor.SADNESS:            (0.15, 0.20, 0.25),
}

_FACTOR_VALENCE_SIGN: dict[GEMSFactor, float] = {
    GEMSFactor.WONDER:             0.7,
    GEMSFactor.TRANSCENDENCE:      0.6,
    GEMSFactor.TENDERNESS:         0.8,
    GEMSFactor.NOSTALGIA:          0.1,
    GEMSFactor.PEACEFULNESS:       0.6,
    GEMSFactor.POWER:              0.3,
    GEMSFactor.JOYFUL_ACTIVATION:  0.9,
    GEMSFactor.TENSION:           -0.4,
    GEMSFactor.SADNESS:           -0.6,
}


class MusicToGEMS:
    FACTOR_CENTROIDS_VAD = _FACTOR_CENTROIDS_VAD

    @staticmethod
    def vad_to_gems(v: float, a: float, d: float) -> GEMSProfile:
        raw: dict[GEMSFactor, float] = {}
        for factor, (cv, ca, cd) in _FACTOR_CENTROIDS_VAD.items():
            dist = math.sqrt((v - cv) ** 2 + (a - ca) ** 2 + (d - cd) ** 2)
            raw[factor] = 1.0 / (1.0 + dist * 5.0)
        mx = max(raw.values()) or 1.0
        return GEMSProfile(scores={f: s / mx for f, s in raw.items()})

    @staticmethod
    def gems_dominant(profile: GEMSProfile) -> GEMSFactor:
        return max(profile.scores, key=lambda f: profile.scores[f])

    @staticmethod
    def gems_valence(profile: GEMSProfile) -> float:
        num = 0.0
        den = 0.0
        for factor, score in profile.scores.items():
            num += score * _FACTOR_VALENCE_SIGN[factor]
            den += score
        if den <= 0:
            return 0.0
        return max(-1.0, min(1.0, num / den))
