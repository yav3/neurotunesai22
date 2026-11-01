"""Cadence analysis: classify and model harmonic cadential patterns.

Identify authentic, half, plagal, deceptive, and Phrygian cadences from
chord progressions. Score cadential strength, detect elided and evaded
cadences, and compute cadential rhythm (IAC vs PAC distinction).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CadenceType(str, Enum):
    PAC = "perfect_authentic"
    IAC = "imperfect_authentic"
    HC = "half"
    PC = "plagal"
    DC = "deceptive"
    PHRYGIAN = "phrygian"
    EVADED = "evaded"


@dataclass
class CadenceEvent:
    type: CadenceType
    position_beat: float
    penult_chord: str
    final_chord: str
    strength: float


def roman_to_scale_degree(roman: str) -> int:
    mapping = {"I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11,
               "i": 0, "ii": 2, "iii": 4, "iv": 5, "v": 7, "vi": 9, "vii": 11}
    base = roman.replace("b", "").replace("#", "").replace("7", "").replace("°", "")
    return mapping.get(base, 0)


def classify_cadence(penult: str, final: str,
                     melody_on_tonic: bool = True) -> CadenceType:
    p_deg = roman_to_scale_degree(penult)
    f_deg = roman_to_scale_degree(final)
    if p_deg == 7 and f_deg == 0:
        return CadenceType.PAC if melody_on_tonic else CadenceType.IAC
    if f_deg == 7:
        return CadenceType.HC
    if p_deg == 5 and f_deg == 0:
        return CadenceType.PC
    if p_deg == 7 and f_deg == 9:
        return CadenceType.DC
    if p_deg == 7 and f_deg != 0:
        return CadenceType.EVADED
    if p_deg == 11 and f_deg == 0:
        return CadenceType.PHRYGIAN
    return CadenceType.HC


def cadential_strength(cadence: CadenceType,
                       duration_beats: float = 1.0,
                       is_root_position: bool = True) -> float:
    base = {
        CadenceType.PAC: 1.0,
        CadenceType.IAC: 0.7,
        CadenceType.HC: 0.5,
        CadenceType.PC: 0.4,
        CadenceType.DC: 0.6,
        CadenceType.PHRYGIAN: 0.5,
        CadenceType.EVADED: 0.3,
    }.get(cadence, 0.3)
    if not is_root_position and cadence in (CadenceType.PAC, CadenceType.IAC):
        base *= 0.7
    return base * min(duration_beats / 2.0, 1.0)


def detect_cadences(chords: List[str], beat_positions: List[float],
                    melody_degrees: Optional[List[int]] = None) -> List[CadenceEvent]:
    events = []
    for i in range(1, len(chords)):
        on_tonic = melody_degrees[i] == 0 if melody_degrees and i < len(melody_degrees) else True
        ctype = classify_cadence(chords[i - 1], chords[i], on_tonic)
        strength = cadential_strength(ctype)
        events.append(CadenceEvent(
            type=ctype, position_beat=beat_positions[i],
            penult_chord=chords[i - 1], final_chord=chords[i],
            strength=strength,
        ))
    return events


def cadential_rhythm(cadences: List[CadenceEvent]) -> Dict[str, float]:
    if not cadences:
        return {"mean_interval": 0.0, "regularity": 0.0}
    positions = [c.position_beat for c in cadences]
    intervals = np.diff(positions)
    if len(intervals) == 0:
        return {"mean_interval": 0.0, "regularity": 0.0}
    mean = float(np.mean(intervals))
    regularity = 1.0 / (1.0 + float(np.std(intervals)))
    return {"mean_interval": mean, "regularity": regularity}


def cadence_distribution(cadences: List[CadenceEvent]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for c in cadences:
        key = c.type.value
        dist[key] = dist.get(key, 0) + 1
    return dist
