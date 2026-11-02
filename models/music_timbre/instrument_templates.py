"""Instrument templates: spectral template matching, instrument identification, family classification."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class InstrumentFamily(str, Enum):
    STRINGS = "strings"
    WOODWINDS = "woodwinds"
    BRASS = "brass"
    PERCUSSION = "percussion"
    KEYBOARDS = "keyboards"
    VOICE = "voice"
    ELECTRONIC = "electronic"
    PLUCKED = "plucked"


@dataclass
class SpectralTemplate:
    name: str
    family: InstrumentFamily
    harmonic_weights: np.ndarray
    formant_freqs: List[float] = field(default_factory=list)
    attack_ms: float = 10.0
    spectral_centroid_ratio: float = 1.0
    spectral_slope: float = -6.0


@dataclass
class IdentificationResult:
    instrument: str
    family: InstrumentFamily
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)


INSTRUMENT_TEMPLATES: Dict[str, SpectralTemplate] = {
    "violin": SpectralTemplate(
        name="violin", family=InstrumentFamily.STRINGS,
        harmonic_weights=np.array([1.0, 0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]),
        formant_freqs=[500.0, 1500.0, 3500.0],
        attack_ms=30.0, spectral_centroid_ratio=3.5, spectral_slope=-3.0,
    ),
    "cello": SpectralTemplate(
        name="cello", family=InstrumentFamily.STRINGS,
        harmonic_weights=np.array([1.0, 0.9, 0.6, 0.45, 0.35, 0.25, 0.15, 0.1]),
        formant_freqs=[250.0, 700.0, 2500.0],
        attack_ms=40.0, spectral_centroid_ratio=2.8, spectral_slope=-4.0,
    ),
    "flute": SpectralTemplate(
        name="flute", family=InstrumentFamily.WOODWINDS,
        harmonic_weights=np.array([1.0, 0.3, 0.1, 0.05, 0.02]),
        formant_freqs=[800.0],
        attack_ms=15.0, spectral_centroid_ratio=1.8, spectral_slope=-8.0,
    ),
    "clarinet": SpectralTemplate(
        name="clarinet", family=InstrumentFamily.WOODWINDS,
        harmonic_weights=np.array([1.0, 0.1, 0.7, 0.1, 0.5, 0.08, 0.3]),
        formant_freqs=[1500.0, 3500.0],
        attack_ms=20.0, spectral_centroid_ratio=2.5, spectral_slope=-5.0,
    ),
    "trumpet": SpectralTemplate(
        name="trumpet", family=InstrumentFamily.BRASS,
        harmonic_weights=np.array([1.0, 0.9, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2]),
        formant_freqs=[1200.0, 2500.0],
        attack_ms=8.0, spectral_centroid_ratio=4.0, spectral_slope=-2.0,
    ),
    "piano": SpectralTemplate(
        name="piano", family=InstrumentFamily.KEYBOARDS,
        harmonic_weights=np.array([1.0, 0.5, 0.3, 0.15, 0.08, 0.04]),
        formant_freqs=[],
        attack_ms=5.0, spectral_centroid_ratio=2.2, spectral_slope=-6.0,
    ),
    "acoustic_guitar": SpectralTemplate(
        name="acoustic_guitar", family=InstrumentFamily.PLUCKED,
        harmonic_weights=np.array([1.0, 0.6, 0.4, 0.3, 0.2, 0.12, 0.08]),
        formant_freqs=[400.0, 1000.0, 3000.0],
        attack_ms=3.0, spectral_centroid_ratio=2.8, spectral_slope=-5.0,
    ),
    "snare_drum": SpectralTemplate(
        name="snare_drum", family=InstrumentFamily.PERCUSSION,
        harmonic_weights=np.array([1.0, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3]),
        formant_freqs=[200.0, 5000.0],
        attack_ms=1.0, spectral_centroid_ratio=5.0, spectral_slope=-1.5,
    ),
}


def harmonic_spectrum_from_template(
    template: SpectralTemplate, fundamental_hz: float, sr: int = 44100,
    n_fft: int = 2048,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    spectrum = np.zeros(len(freqs))
    for k, weight in enumerate(template.harmonic_weights):
        f = fundamental_hz * (k + 1)
        if f > sr / 2:
            break
        idx = np.argmin(np.abs(freqs - f))
        spread = max(1, int(n_fft * 5.0 / sr))
        lo = max(0, idx - spread)
        hi = min(len(spectrum), idx + spread + 1)
        for i in range(lo, hi):
            d = abs(i - idx)
            spectrum[i] += weight * math.exp(-0.5 * (d / max(spread * 0.3, 0.1)) ** 2)
    for ff in template.formant_freqs:
        bw = ff * 0.15
        mask = np.exp(-0.5 * ((freqs - ff) / max(bw, 1.0)) ** 2)
        spectrum *= (1.0 + 0.5 * mask)
    peak = np.max(spectrum)
    if peak > 0:
        spectrum /= peak
    return spectrum


def template_match_score(
    observed_spectrum: np.ndarray, template_spectrum: np.ndarray,
) -> float:
    na = np.linalg.norm(observed_spectrum)
    nb = np.linalg.norm(template_spectrum)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(observed_spectrum, template_spectrum) / (na * nb))


def identify_instrument(
    spectrum: np.ndarray, fundamental_hz: float,
    sr: int = 44100, n_fft: int = 2048,
    candidates: Optional[List[str]] = None,
) -> IdentificationResult:
    templates = INSTRUMENT_TEMPLATES
    if candidates:
        templates = {k: v for k, v in templates.items() if k in candidates}
    if not templates:
        return IdentificationResult(
            instrument="unknown", family=InstrumentFamily.ELECTRONIC,
            confidence=0.0,
        )
    scores: Dict[str, float] = {}
    for name, tmpl in templates.items():
        ref = harmonic_spectrum_from_template(tmpl, fundamental_hz, sr, n_fft)
        if len(ref) != len(spectrum):
            min_len = min(len(ref), len(spectrum))
            ref = ref[:min_len]
            obs = spectrum[:min_len]
        else:
            obs = spectrum
        scores[name] = template_match_score(obs, ref)
    best = max(scores, key=scores.get)
    tmpl = templates[best]
    return IdentificationResult(
        instrument=best, family=tmpl.family,
        confidence=scores[best], scores=scores,
    )


def family_probabilities(scores: Dict[str, float]) -> Dict[str, float]:
    family_max: Dict[str, float] = {}
    for name, score in scores.items():
        tmpl = INSTRUMENT_TEMPLATES.get(name)
        if tmpl:
            fam = tmpl.family.value
            family_max[fam] = max(family_max.get(fam, 0.0), score)
    total = sum(family_max.values())
    if total < 1e-12:
        return family_max
    return {k: v / total for k, v in family_max.items()}


def attack_time_estimate(signal: np.ndarray, sr: int = 44100, threshold: float = 0.1) -> float:
    env = np.abs(signal)
    n = len(env)
    if n == 0:
        return 0.0
    peak = np.max(env)
    if peak < 1e-12:
        return 0.0
    env_norm = env / peak
    onset = 0
    for i in range(n):
        if env_norm[i] >= threshold:
            onset = i
            break
    peak_idx = int(np.argmax(env_norm))
    return (peak_idx - onset) / sr * 1000.0


def match_by_attack(
    attack_ms: float, tolerance_ms: float = 10.0,
) -> List[Tuple[str, float]]:
    results = []
    for name, tmpl in INSTRUMENT_TEMPLATES.items():
        diff = abs(attack_ms - tmpl.attack_ms)
        score = max(0.0, 1.0 - diff / max(tolerance_ms, 1.0))
        if score > 0:
            results.append((name, score))
    results.sort(key=lambda x: -x[1])
    return results
