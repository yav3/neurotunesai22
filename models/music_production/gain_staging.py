"""Gain staging: headroom management, level metering, clipping detection, calibration."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class MeterType(str, Enum):
    PEAK = "peak"
    RMS = "rms"
    VU = "vu"
    LUFS = "lufs"


@dataclass
class StageMeter:
    peak_db: float
    rms_db: float
    headroom_db: float
    clipping: bool
    crest_factor_db: float


@dataclass
class GainStageReport:
    input_meter: StageMeter
    output_meter: StageMeter
    recommended_gain_db: float
    target_headroom_db: float


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    if linear <= 0:
        return -120.0
    return 20.0 * math.log10(linear)


def peak_db(samples: List[float]) -> float:
    if not samples:
        return -120.0
    return linear_to_db(max(abs(s) for s in samples))


def rms_db(samples: List[float]) -> float:
    if not samples:
        return -120.0
    mean_sq = sum(s * s for s in samples) / len(samples)
    return linear_to_db(math.sqrt(mean_sq))


def meter(samples: List[float], ceiling_db: float = 0.0) -> StageMeter:
    pk = peak_db(samples)
    rm = rms_db(samples)
    return StageMeter(
        peak_db=pk,
        rms_db=rm,
        headroom_db=ceiling_db - pk,
        clipping=pk >= ceiling_db,
        crest_factor_db=pk - rm,
    )


def apply_gain(samples: List[float], gain_db: float) -> List[float]:
    g = db_to_linear(gain_db)
    return [s * g for s in samples]


def clip_count(samples: List[float], threshold: float = 1.0) -> int:
    return sum(1 for s in samples if abs(s) >= threshold)


def clip_ratio(samples: List[float], threshold: float = 1.0) -> float:
    if not samples:
        return 0.0
    return clip_count(samples, threshold) / len(samples)


def true_peak_estimate(samples: List[float], oversample: int = 4) -> float:
    if not samples:
        return -120.0
    n = len(samples)
    peak = 0.0
    for i in range(n - 1):
        for k in range(oversample):
            t = k / oversample
            interp = samples[i] * (1 - t) + samples[i + 1] * t
            peak = max(peak, abs(interp))
    peak = max(peak, abs(samples[-1]))
    return linear_to_db(peak)


def recommended_gain(samples: List[float], target_headroom_db: float = 6.0,
                     ceiling_db: float = 0.0) -> float:
    pk = peak_db(samples)
    target_peak = ceiling_db - target_headroom_db
    return target_peak - pk


def gain_stage_report(input_samples: List[float], output_samples: List[float],
                      target_headroom_db: float = 6.0, ceiling_db: float = 0.0) -> GainStageReport:
    inp = meter(input_samples, ceiling_db)
    out = meter(output_samples, ceiling_db)
    rec = recommended_gain(output_samples, target_headroom_db, ceiling_db)
    return GainStageReport(input_meter=inp, output_meter=out,
                            recommended_gain_db=rec, target_headroom_db=target_headroom_db)


def calibrate_chain(stage_gains_db: List[float], stage_headrooms_db: List[float],
                    target_headroom_db: float = 6.0) -> List[float]:
    adjustments: List[float] = []
    for gain, headroom in zip(stage_gains_db, stage_headrooms_db):
        deficit = target_headroom_db - headroom
        adjustments.append(-deficit)
    return adjustments


def k_system_reference(k_level: int = 14) -> float:
    return -k_level
