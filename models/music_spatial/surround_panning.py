"""Surround-sound panning: VBAP, DBAP, and speaker-layout utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class PanningMethod(str, Enum):
    VBAP = "vbap"
    DBAP = "dbap"
    EQUAL_POWER = "equal_power"


@dataclass
class SpeakerLayout:
    azimuths_deg: List[float]
    elevations_deg: List[float] = field(default_factory=list)

    @property
    def n_speakers(self) -> int:
        return len(self.azimuths_deg)

    def cartesian(self) -> List[Tuple[float, float, float]]:
        pts: List[Tuple[float, float, float]] = []
        for i, az in enumerate(self.azimuths_deg):
            el = self.elevations_deg[i] if i < len(self.elevations_deg) else 0.0
            ra = math.radians(az)
            re = math.radians(el)
            pts.append((math.cos(re) * math.cos(ra), math.cos(re) * math.sin(ra), math.sin(re)))
        return pts


STEREO = SpeakerLayout(azimuths_deg=[-30.0, 30.0])
QUAD = SpeakerLayout(azimuths_deg=[-45.0, 45.0, -135.0, 135.0])
SURROUND_5_1 = SpeakerLayout(azimuths_deg=[0.0, -30.0, 30.0, -110.0, 110.0])
SURROUND_7_1 = SpeakerLayout(azimuths_deg=[0.0, -30.0, 30.0, -90.0, 90.0, -135.0, 135.0])


def _normalize(v: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def _dot(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def vbap_gains(layout: SpeakerLayout, azimuth_deg: float, elevation_deg: float = 0.0) -> List[float]:
    ra = math.radians(azimuth_deg)
    re = math.radians(elevation_deg)
    src = (math.cos(re) * math.cos(ra), math.cos(re) * math.sin(ra), math.sin(re))
    spk = layout.cartesian()
    n = layout.n_speakers
    gains = [0.0] * n
    best_pair = (-1, -1)
    best_sum = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            gi = _dot(src, spk[i])
            gj = _dot(src, spk[j])
            if gi >= 0 and gj >= 0 and gi + gj > best_sum:
                best_sum = gi + gj
                best_pair = (i, j)
    if best_pair[0] >= 0:
        gi = max(_dot(src, spk[best_pair[0]]), 0.0)
        gj = max(_dot(src, spk[best_pair[1]]), 0.0)
        norm = math.sqrt(gi * gi + gj * gj) or 1.0
        gains[best_pair[0]] = gi / norm
        gains[best_pair[1]] = gj / norm
    else:
        nearest = min(range(n), key=lambda k: sum((a - b) ** 2 for a, b in zip(src, spk[k])))
        gains[nearest] = 1.0
    return gains


def dbap_gains(layout: SpeakerLayout, azimuth_deg: float, elevation_deg: float = 0.0,
               rolloff: float = 6.0) -> List[float]:
    ra = math.radians(azimuth_deg)
    re = math.radians(elevation_deg)
    src = (math.cos(re) * math.cos(ra), math.cos(re) * math.sin(ra), math.sin(re))
    spk = layout.cartesian()
    a = rolloff / 20.0
    weights: List[float] = []
    for s in spk:
        d = math.sqrt(sum((si - pi) ** 2 for si, pi in zip(src, s))) or 1e-6
        weights.append(1.0 / (d ** a))
    total = sum(w * w for w in weights)
    norm = math.sqrt(total) or 1.0
    return [w / norm for w in weights]


def equal_power_pan(position: float) -> Tuple[float, float]:
    theta = max(0.0, min(1.0, position)) * math.pi / 2.0
    return (math.cos(theta), math.sin(theta))


def speaker_distance_matrix(layout: SpeakerLayout) -> List[List[float]]:
    spk = layout.cartesian()
    n = layout.n_speakers
    mat: List[List[float]] = []
    for i in range(n):
        row: List[float] = []
        for j in range(n):
            row.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(spk[i], spk[j]))))
        mat.append(row)
    return mat


def pan_trajectory(layout: SpeakerLayout, azimuths: List[float],
                   method: PanningMethod = PanningMethod.VBAP) -> List[List[float]]:
    fn = vbap_gains if method == PanningMethod.VBAP else dbap_gains
    return [fn(layout, az) for az in azimuths]
