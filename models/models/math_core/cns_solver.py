"""CNS pharmacodynamics — rTMS/GPCR modulation, synchronization distance, affect purity."""

from __future__ import annotations

import numpy as np

PHI = (1.0 + 5.0 ** 0.5) / 2.0


def gpcr_rtms_modulation(g0: float, field_strength: float,
                         f_rtms: float, t_stim: float) -> dict:
    if g0 < 0 or field_strength < 0 or f_rtms < 0 or t_stim < 0:
        raise ValueError("all inputs must be non-negative")
    rate = field_strength * f_rtms
    g_adj = float(g0 * np.exp(-(rate * t_stim)))
    return {
        "g_adj": g_adj,
        "decay_rate": float(rate),
        "fraction_remaining": float(g_adj / g0) if g0 > 0 else 0.0,
    }


def cns_synchronization_distance(velocity: float, frequency: float,
                                 delta_phi: float, delta_t: float = 0.0,
                                 coherence: float | None = None) -> dict:
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    phase_term = float(velocity / (2.0 * np.pi * frequency) * delta_phi)
    delay_term = float(velocity * delta_t)
    terms: dict[str, float] = {"phase_shift": phase_term, "time_delay": delay_term}
    hybrid = phase_term + delay_term
    if coherence is not None:
        if coherence <= 0:
            raise ValueError("coherence must be positive")
        coherence_term = float(1.0 / coherence)
        terms["coherence"] = coherence_term
        hybrid += coherence_term
    terms["hybrid"] = float(hybrid)
    return terms


def affect_purity(valence: float, arousal: float, dominance: float) -> dict:
    mags = np.sort(np.abs([valence, arousal, dominance]))[::-1]
    eps = 1e-9
    r1 = float(mags[0] / (mags[1] + eps))
    r2 = float(mags[1] / (mags[2] + eps))
    deviation = abs(r1 - PHI) + abs(r2 - PHI)
    purity = float(np.exp(-deviation / PHI))
    return {
        "purity": purity,
        "ratios": [r1, r2],
        "phi": PHI,
        "magnitude": float(np.linalg.norm([valence, arousal, dominance])),
    }
