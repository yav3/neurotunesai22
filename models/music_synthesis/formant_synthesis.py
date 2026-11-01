"""Formant synthesis: vocal tract modeling, vowel interpolation,
singing voice synthesis via cascaded resonators.

Pure numpy — no external audio frameworks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FormantFreqs:
    f1: float
    f2: float
    f3: float
    f4: float = 3300.0
    f5: float = 3750.0
    bw1: float = 60.0
    bw2: float = 80.0
    bw3: float = 100.0
    bw4: float = 120.0
    bw5: float = 140.0


VOWEL_FORMANTS: Dict[str, FormantFreqs] = {
    "a": FormantFreqs(f1=730, f2=1090, f3=2440),
    "e": FormantFreqs(f1=530, f2=1840, f3=2480),
    "i": FormantFreqs(f1=270, f2=2290, f3=3010),
    "o": FormantFreqs(f1=570, f2=840, f3=2410),
    "u": FormantFreqs(f1=300, f2=870, f3=2240),
    "ae": FormantFreqs(f1=660, f2=1720, f3=2410),
    "schwa": FormantFreqs(f1=500, f2=1500, f3=2500),
}


def _biquad_resonator(freq: float, bandwidth: float,
                      sr: float) -> Tuple[np.ndarray, np.ndarray]:
    r = math.exp(-math.pi * bandwidth / sr)
    theta = 2.0 * math.pi * freq / sr
    b = np.array([1.0 - r, 0.0, -(1.0 - r)])
    a = np.array([1.0, -2.0 * r * math.cos(theta), r * r])
    return b, a


def _apply_filter(signal: np.ndarray, b: np.ndarray,
                  a: np.ndarray) -> np.ndarray:
    n = len(signal)
    y = np.zeros(n)
    nb = len(b)
    na = len(a)
    for i in range(n):
        val = 0.0
        for j in range(nb):
            if i - j >= 0:
                val += b[j] * signal[i - j]
        for j in range(1, na):
            if i - j >= 0:
                val -= a[j] * y[i - j]
        y[i] = val
    return y


def glottal_pulse(f0: float, sr: float, n_samples: int,
                  open_quotient: float = 0.6) -> np.ndarray:
    period = sr / f0
    t = np.arange(n_samples)
    phase = (t / period) % 1.0

    pulse = np.zeros(n_samples)
    oq = np.clip(open_quotient, 0.1, 0.95)
    opening = phase < oq
    pulse[opening] = 0.5 * (1.0 - np.cos(math.pi * phase[opening] / oq))
    closing = (phase >= oq) & (phase < 1.0)
    close_phase = (phase[closing] - oq) / (1.0 - oq)
    pulse[closing] = 0.5 * (1.0 + np.cos(math.pi * close_phase))

    diff = np.diff(pulse, prepend=pulse[0])
    return diff


def aspiration_noise(n_samples: int, level: float = 0.01,
                     seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return level * rng.standard_normal(n_samples)


def formant_filter(signal: np.ndarray, formants: FormantFreqs,
                   sr: float) -> np.ndarray:
    freqs = [formants.f1, formants.f2, formants.f3, formants.f4, formants.f5]
    bws = [formants.bw1, formants.bw2, formants.bw3, formants.bw4, formants.bw5]

    output = np.zeros_like(signal)
    for f, bw in zip(freqs, bws):
        if f >= sr / 2:
            continue
        b, a = _biquad_resonator(f, bw, sr)
        output += _apply_filter(signal, b, a)
    return output


def interpolate_formants(fa: FormantFreqs, fb: FormantFreqs,
                         t: float) -> FormantFreqs:
    t = np.clip(t, 0.0, 1.0)
    return FormantFreqs(
        f1=fa.f1 + t * (fb.f1 - fa.f1),
        f2=fa.f2 + t * (fb.f2 - fa.f2),
        f3=fa.f3 + t * (fb.f3 - fa.f3),
        f4=fa.f4 + t * (fb.f4 - fa.f4),
        f5=fa.f5 + t * (fb.f5 - fa.f5),
        bw1=fa.bw1 + t * (fb.bw1 - fa.bw1),
        bw2=fa.bw2 + t * (fb.bw2 - fa.bw2),
        bw3=fa.bw3 + t * (fb.bw3 - fa.bw3),
        bw4=fa.bw4 + t * (fb.bw4 - fa.bw4),
        bw5=fa.bw5 + t * (fb.bw5 - fa.bw5),
    )


def vowel_trajectory(vowel_sequence: List[str], durations: List[float],
                     sr: float, f0: float = 150.0,
                     open_quotient: float = 0.6) -> np.ndarray:
    total_samples = int(sum(durations) * sr)
    source = glottal_pulse(f0, sr, total_samples, open_quotient)
    source += aspiration_noise(total_samples, 0.005)

    output = np.zeros(total_samples)
    sample_idx = 0

    for seg_i in range(len(vowel_sequence)):
        v = vowel_sequence[seg_i]
        dur_samples = int(durations[seg_i] * sr)
        if dur_samples <= 0:
            continue

        fa = VOWEL_FORMANTS.get(v, VOWEL_FORMANTS["schwa"])
        if seg_i + 1 < len(vowel_sequence):
            fb = VOWEL_FORMANTS.get(vowel_sequence[seg_i + 1],
                                    VOWEL_FORMANTS["schwa"])
        else:
            fb = fa

        for k in range(dur_samples):
            if sample_idx + k >= total_samples:
                break
        end_idx = min(sample_idx + dur_samples, total_samples)
        seg_source = source[sample_idx:end_idx]

        transition_start = int(0.7 * dur_samples)
        for k in range(end_idx - sample_idx):
            if k < transition_start:
                t_interp = 0.0
            else:
                t_interp = (k - transition_start) / max(1, dur_samples - transition_start)
            f_interp = interpolate_formants(fa, fb, t_interp)
            pass

        filtered = formant_filter(seg_source, fa, sr)
        output[sample_idx:end_idx] = filtered
        sample_idx = end_idx

    peak = np.max(np.abs(output)) + 1e-12
    return output / peak


def vibrato(f0_base: float, rate: float, extent_cents: float,
            sr: float, n_samples: int) -> np.ndarray:
    t = np.arange(n_samples) / sr
    cents_mod = extent_cents * np.sin(2.0 * math.pi * rate * t)
    f0_mod = f0_base * (2.0 ** (cents_mod / 1200.0))
    phase = np.cumsum(2.0 * math.pi * f0_mod / sr)
    return np.sin(phase)


def singing_synthesis(text_phonemes: List[str], pitches: List[float],
                      durations: List[float], sr: float = 44100.0,
                      vibrato_rate: float = 5.5,
                      vibrato_extent: float = 30.0) -> np.ndarray:
    total_samples = int(sum(durations) * sr)
    output = np.zeros(total_samples)
    idx = 0

    for phoneme, pitch, dur in zip(text_phonemes, pitches, durations):
        n = int(dur * sr)
        if n <= 0 or idx >= total_samples:
            continue
        n = min(n, total_samples - idx)

        source = vibrato(pitch, vibrato_rate, vibrato_extent, sr, n)
        source += aspiration_noise(n, 0.003)

        formants = VOWEL_FORMANTS.get(phoneme, VOWEL_FORMANTS["schwa"])
        filtered = formant_filter(source, formants, sr)

        env = np.ones(n)
        attack = min(int(0.02 * sr), n // 4)
        release = min(int(0.03 * sr), n // 4)
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)
        if release > 0:
            env[-release:] = np.linspace(1, 0, release)
        output[idx:idx + n] = filtered * env
        idx += n

    peak = np.max(np.abs(output)) + 1e-12
    return output / peak


def formant_analysis(spectrum: np.ndarray, sr: float,
                     n_formants: int = 5) -> List[Tuple[float, float]]:
    n = len(spectrum)
    freqs_hz = np.linspace(0, sr / 2, n)
    log_spec = np.log(np.abs(spectrum) + 1e-10)

    cepstrum = np.fft.irfft(log_spec)
    lifter = min(30, len(cepstrum) // 2)
    cepstrum_smooth = cepstrum.copy()
    cepstrum_smooth[lifter:-lifter] = 0
    envelope = np.fft.rfft(cepstrum_smooth).real[:n]

    formants = []
    for i in range(1, len(envelope) - 1):
        if envelope[i] > envelope[i - 1] and envelope[i] > envelope[i + 1]:
            freq = freqs_hz[i] if i < len(freqs_hz) else 0.0
            if 80 < freq < sr / 2:
                prominence = envelope[i] - 0.5 * (envelope[i - 1] + envelope[i + 1])
                formants.append((freq, prominence))
    formants.sort(key=lambda x: -x[1])
    return formants[:n_formants]


def vocal_tract_length_estimate(formants: List[float]) -> float:
    c = 34300.0
    if len(formants) < 2:
        return 17.0
    estimates = []
    for i, f in enumerate(formants):
        if f > 0:
            vtl = c / (4.0 * f) * (2 * (i + 1) - 1)
            estimates.append(vtl)
    return float(np.median(estimates)) if estimates else 17.0
