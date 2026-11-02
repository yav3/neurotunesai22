"""Rhythmic entrainment for self-regulation — Kuramoto-style phase coupling."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class EntropicCoupling:
    natural_freq: float
    coupling_k: float = 1.0
    phase: float = 0.0

    def step(self, driver_phase: float, dt: float = 0.01) -> float:
        dphi = self.natural_freq + self.coupling_k * np.sin(driver_phase - self.phase)
        self.phase += dphi * dt
        return float(self.phase)

    def run(self, driver_phases: list[float], dt: float = 0.01) -> list[float]:
        out = []
        for dp in driver_phases:
            out.append(self.step(dp, dt))
        return out


def phase_coupling_strength(
    driver_freq: float, response_freq: float, tolerance: float = 0.1
) -> float:
    if driver_freq <= 0 or response_freq <= 0:
        return 0.0
    ratio = response_freq / driver_freq
    nearest_int = round(ratio)
    if nearest_int == 0:
        nearest_int = 1
    deviation = abs(ratio - nearest_int)
    return float(max(0.0, 1.0 - deviation / tolerance))


def entrainment_time(freq_diff: float, coupling_k: float) -> float:
    if coupling_k <= 0:
        return float("inf")
    if abs(freq_diff) < 1e-12:
        return 0.0
    return float(abs(freq_diff) / coupling_k)


def pulse_synchronization(
    tap_times: list[float], beat_times: list[float]
) -> tuple[float, list[float]]:
    if not tap_times or not beat_times:
        return 0.0, []
    beats = np.array(beat_times)
    offsets = []
    for tap in tap_times:
        idx = np.argmin(np.abs(beats - tap))
        offsets.append(float(tap - beats[idx]))
    beat_period = np.median(np.diff(beats)) if len(beats) > 1 else 1.0
    sync_ratio = float(np.mean(np.abs(offsets) < beat_period * 0.25))
    return sync_ratio, offsets


def kuramoto_order_parameter(phases: list[float]) -> float:
    if not phases:
        return 0.0
    z = np.mean(np.exp(1j * np.array(phases)))
    return float(np.abs(z))
