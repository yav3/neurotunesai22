"""Formant synthesis and estimation for vocal/instrument modelling."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal as sig


VOWEL_FORMANTS: dict[str, tuple[float, ...]] = {
    "a": (730, 1090, 2440, 80, 90, 120),
    "e": (530, 1840, 2480, 60, 90, 120),
    "i": (270, 2290, 3010, 50, 90, 120),
    "o": (570, 840, 2410, 70, 80, 120),
    "u": (300, 870, 2240, 50, 70, 110),
}


class FormantSynthesizer:
    """Formant-based synthesis and analysis."""

    @staticmethod
    def vowel_formants(vowel: str) -> tuple[float, ...]:
        return VOWEL_FORMANTS.get(vowel.lower(), VOWEL_FORMANTS["a"])

    @staticmethod
    def formant_filter(
        signal_in: np.ndarray,
        sr: int,
        formant_freqs: list[float],
        bandwidths: list[float],
        gains: list[float] | None = None,
    ) -> np.ndarray:
        if len(signal_in) == 0:
            return signal_in.copy()
        if gains is None:
            gains = [1.0] * len(formant_freqs)
        out = np.zeros_like(signal_in, dtype=float)
        for f, bw, g in zip(formant_freqs, bandwidths, gains):
            low = max((f - bw / 2) / (sr / 2), 0.001)
            high = min((f + bw / 2) / (sr / 2), 0.999)
            if low >= high:
                continue
            sos = sig.butter(2, [low, high], btype="bandpass", output="sos")
            out += g * sig.sosfilt(sos, signal_in)
        return out

    @staticmethod
    def singing_formant(
        f0: float, formant_cluster_center: float = 3000.0
    ) -> tuple[float, float, float]:
        bw = 400.0
        gain = 1.0 + 0.5 * min(f0 / 500.0, 1.0)
        return formant_cluster_center, bw, gain

    @staticmethod
    def formant_transition(
        start_vowel: str, end_vowel: str, n_frames: int = 20
    ) -> np.ndarray:
        s = np.array(VOWEL_FORMANTS.get(start_vowel.lower(), VOWEL_FORMANTS["a"])[:3])
        e = np.array(VOWEL_FORMANTS.get(end_vowel.lower(), VOWEL_FORMANTS["a"])[:3])
        t = np.linspace(0, 1, n_frames).reshape(-1, 1)
        return s * (1 - t) + e * t

    @staticmethod
    def estimate_formants(
        spectrum: np.ndarray, freqs: np.ndarray, n_formants: int = 4
    ) -> list[float]:
        if len(spectrum) < 3:
            return [0.0] * n_formants
        log_spec = np.log(np.abs(spectrum) + 1e-30)
        order = min(2 * n_formants + 2, len(log_spec) - 1)
        coeffs = np.polyfit(np.arange(len(log_spec)), log_spec, order)
        roots = np.roots(coeffs)
        real_freqs = []
        for r in roots:
            if np.imag(r) != 0:
                angle = np.abs(np.angle(r))
                f = angle / np.pi * (freqs[-1] if len(freqs) > 0 else 8000)
                if 90 < f < 5000:
                    real_freqs.append(f)
        real_freqs = sorted(set(int(f) for f in real_freqs))
        while len(real_freqs) < n_formants:
            real_freqs.append(0.0)
        return [float(f) for f in real_freqs[:n_formants]]


class ResonanceModel:
    """Physical resonance models for strings, tubes, membranes."""

    @staticmethod
    def string_modes(
        length: float,
        tension: float,
        density: float,
        n_modes: int = 16,
    ) -> np.ndarray:
        if length <= 0 or tension <= 0 or density <= 0:
            return np.zeros(n_modes)
        f1 = (1 / (2 * length)) * np.sqrt(tension / density)
        return f1 * np.arange(1, n_modes + 1)

    @staticmethod
    def inharmonicity_coefficient(
        string_diameter: float, length: float, youngs_modulus: float
    ) -> float:
        if length <= 0 or string_diameter <= 0:
            return 0.0
        r = string_diameter / 2
        return (np.pi ** 3 * youngs_modulus * r ** 4) / (4 * length ** 2)

    @staticmethod
    def tube_resonance(
        length: float,
        end_correction: float = 0.6,
        n_modes: int = 8,
        open_open: bool = True,
    ) -> np.ndarray:
        if length <= 0:
            return np.zeros(n_modes)
        c = 343.0
        eff_length = length + end_correction * (0.02 if open_open else 0.01)
        if open_open:
            return (c / (2 * eff_length)) * np.arange(1, n_modes + 1)
        return (c / (4 * eff_length)) * (2 * np.arange(n_modes) + 1)

    @staticmethod
    def membrane_modes(
        radius: float,
        tension: float,
        density: float,
        n_modes: int = 10,
    ) -> np.ndarray:
        if radius <= 0 or tension <= 0 or density <= 0:
            return np.zeros(n_modes)
        from scipy.special import jn_zeros

        c = np.sqrt(tension / density)
        zeros_list = []
        for m in range(4):
            z = jn_zeros(m, 3)
            zeros_list.extend(z.tolist())
        zeros_list = sorted(zeros_list)[:n_modes]
        while len(zeros_list) < n_modes:
            zeros_list.append(zeros_list[-1] + 1.0 if zeros_list else 1.0)
        return np.array([z * c / (2 * np.pi * radius) for z in zeros_list])

    @staticmethod
    def coupled_oscillators(
        frequencies: np.ndarray,
        coupling_matrix: np.ndarray,
        t_span: float,
        dt: float = 0.001,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(frequencies)
        if n == 0:
            return np.array([0.0]), np.zeros((1, 0))
        omega = 2 * np.pi * np.array(frequencies)
        steps = int(t_span / dt)
        t = np.linspace(0, t_span, steps)
        x = np.zeros((steps, n))
        v = np.zeros((steps, n))
        x[0] = 1.0
        K = np.array(coupling_matrix)
        for i in range(steps - 1):
            acc = -(omega ** 2) * x[i]
            for j in range(n):
                for k in range(n):
                    if j != k:
                        acc[j] += K[j, k] * (x[i, k] - x[i, j])
            v_half = v[i] + 0.5 * dt * acc
            x[i + 1] = x[i] + dt * v_half
            acc_new = -(omega ** 2) * x[i + 1]
            for j in range(n):
                for k in range(n):
                    if j != k:
                        acc_new[j] += K[j, k] * (x[i + 1, k] - x[i + 1, j])
            v[i + 1] = v_half + 0.5 * dt * acc_new
        return t, x

    @staticmethod
    def decay_rates(
        frequencies: np.ndarray, q_factors: np.ndarray
    ) -> np.ndarray:
        freqs = np.asarray(frequencies, dtype=float)
        qs = np.asarray(q_factors, dtype=float)
        safe_f = np.where(freqs > 0, freqs, 1.0)
        tau = qs / (np.pi * safe_f)
        tau = np.where(freqs > 0, tau, 0.0)
        return tau
