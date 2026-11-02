"""Temporal perception: integration windows, Weber's law, masking, gap detection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IntegrationWindow:
    name: str
    duration_ms: float
    description: str


INTEGRATION_WINDOWS = {
    "fine_structure": IntegrationWindow("fine_structure", 2.5, "Phase-locking, binaural cues"),
    "pitch": IntegrationWindow("pitch", 30.0, "Pitch extraction, timbre"),
    "loudness": IntegrationWindow("loudness", 200.0, "Loudness integration"),
    "rhythm": IntegrationWindow("rhythm", 2000.0, "Beat, meter, grouping"),
}


def weber_tempo_jnd(base_ioi_ms: float, weber_fraction: float = 0.03) -> float:
    return base_ioi_ms * weber_fraction


def tempo_discrimination(ioi_a_ms: float, ioi_b_ms: float,
                         weber_fraction: float = 0.03) -> bool:
    jnd = weber_tempo_jnd(min(ioi_a_ms, ioi_b_ms), weber_fraction)
    return abs(ioi_a_ms - ioi_b_ms) > jnd


def subjective_duration(physical_ms: float, filled: bool = True,
                        n_events: int = 0) -> float:
    base = physical_ms
    if filled:
        base *= 1.1
    if n_events > 1:
        base *= 1.0 + 0.02 * min(n_events, 10)
    return base


def forward_masking_threshold(delay_ms: float, masker_level_db: float = 60.0,
                              masker_duration_ms: float = 200.0) -> float:
    if delay_ms <= 0:
        return masker_level_db
    decay = masker_level_db * np.exp(-delay_ms / 50.0)
    dur_factor = min(masker_duration_ms / 200.0, 1.0)
    return float(decay * dur_factor)


def backward_masking_threshold(advance_ms: float,
                               masker_level_db: float = 60.0) -> float:
    if advance_ms <= 0:
        return masker_level_db
    if advance_ms > 100:
        return 0.0
    decay = masker_level_db * (1.0 - advance_ms / 100.0)
    return float(max(decay, 0.0))


def gap_detection_threshold(center_freq_hz: float = 1000.0,
                            level_db: float = 60.0) -> float:
    base_ms = 3.0
    if center_freq_hz < 500:
        base_ms += 2.0 * (500 - center_freq_hz) / 500
    level_factor = max(0.5, 1.0 - (level_db - 40) / 80)
    return base_ms * level_factor


def temporal_order_judgment(soa_ms: float) -> float:
    threshold = 20.0
    if abs(soa_ms) < threshold:
        return 0.5
    return float(1.0 / (1.0 + np.exp(-0.15 * (abs(soa_ms) - threshold))))


def temporal_modulation_transfer(mod_freq_hz: float,
                                 carrier_freq_hz: float = 4000.0) -> float:
    cutoff = 64.0
    return float(1.0 / np.sqrt(1.0 + (mod_freq_hz / cutoff) ** 2))


def rhythm_perception_window(tempo_bpm: float) -> tuple[float, float]:
    ioi_ms = 60000.0 / max(tempo_bpm, 1)
    lower = max(ioi_ms * 0.5, 100)
    upper = min(ioi_ms * 4.0, 5000)
    return (lower, upper)


def preferred_tempo_resonance(tempo_bpm: float,
                              preferred_bpm: float = 120.0) -> float:
    log_ratio = np.log2(tempo_bpm / preferred_bpm)
    return float(np.exp(-2.0 * log_ratio ** 2))


def ioi_variability(onset_times: np.ndarray) -> float:
    if len(onset_times) < 3:
        return 0.0
    iois = np.diff(np.sort(onset_times))
    return float(np.std(iois) / max(np.mean(iois), 1e-10))


def perceptual_attack_time(envelope: np.ndarray, sr: float = 44100.0,
                           threshold: float = 0.1) -> float:
    if len(envelope) == 0:
        return 0.0
    peak = np.max(np.abs(envelope))
    if peak < 1e-10:
        return 0.0
    norm = np.abs(envelope) / peak
    onset_idx = np.argmax(norm >= threshold)
    peak_idx = np.argmax(norm)
    if peak_idx <= onset_idx:
        return 0.0
    return float((peak_idx - onset_idx) / sr)
