"""Physical resonance models for string, membrane, and bar instruments."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StringResonance:
    """Damped harmonic oscillator model for vibrating strings.

    f_n = (n / (2L)) * sqrt(T / mu)  where mu = linear density.
    """

    def modes(
        self,
        length: float,
        tension: float,
        density: float,
        n_modes: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        mu = density
        f1 = (1.0 / (2.0 * length)) * np.sqrt(tension / mu)
        ns = np.arange(1, n_modes + 1)
        freqs = f1 * ns
        amps = 1.0 / ns
        return freqs, amps

    def decay_envelope(
        self, f0: float, damping: float, duration: float, sr: int
    ) -> np.ndarray:
        t = np.arange(int(sr * duration)) / sr
        return np.exp(-damping * t) * np.cos(2.0 * np.pi * f0 * t)

    def pluck_spectrum(
        self, position: float, n_modes: int = 10
    ) -> np.ndarray:
        ns = np.arange(1, n_modes + 1)
        return np.abs(np.sin(ns * np.pi * position))

    def synthesize(
        self,
        length: float,
        tension: float,
        density: float,
        pluck_pos: float,
        damping: float,
        duration: float,
        sr: int,
        n_modes: int = 10,
    ) -> np.ndarray:
        freqs, _ = self.modes(length, tension, density, n_modes)
        weights = self.pluck_spectrum(pluck_pos, n_modes)
        t = np.arange(int(sr * duration)) / sr
        signal = np.zeros(len(t))
        for i, (f, w) in enumerate(zip(freqs, weights)):
            mode_damping = damping * (1.0 + 0.5 * i)
            signal += w * np.exp(-mode_damping * t) * np.sin(2 * np.pi * f * t)
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal /= peak
        return signal


@dataclass
class MembraneResonance:
    """Circular membrane vibration (drums, timpani).

    Modes follow Bessel function zeros: f_{mn} = (z_{mn} / (2 pi R)) sqrt(T / sigma).
    """

    _bessel_zeros: list = field(
        default_factory=lambda: [
            2.4048, 3.8317, 5.1356, 5.5201, 6.3802,
            7.0156, 7.5883, 8.4172, 8.6537, 8.7715,
        ]
    )

    def modes(
        self,
        radius: float,
        tension: float,
        density: float,
        n_modes: int = 10,
    ) -> np.ndarray:
        sigma = density
        n = min(n_modes, len(self._bessel_zeros))
        zeros = np.array(self._bessel_zeros[:n])
        freqs = (zeros / (2.0 * np.pi * radius)) * np.sqrt(tension / sigma)
        return freqs

    def mode_shapes(
        self,
        radius: float,
        n_radial: int = 0,
        n_angular: int = 0,
        grid_size: int = 64,
    ) -> np.ndarray:
        x = np.linspace(-radius, radius, grid_size)
        y = np.linspace(-radius, radius, grid_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        from scipy.special import jn_zeros, jn

        if n_radial >= 0 and n_angular >= 0:
            z = jn_zeros(n_angular, n_radial + 1)[n_radial]
            shape = jn(n_angular, z * R / radius) * np.cos(n_angular * Theta)
            shape[R > radius] = 0.0
            return shape
        return np.zeros((grid_size, grid_size))

    def inharmonicity_ratio(
        self, radius: float, tension: float, density: float, n_modes: int = 6
    ) -> np.ndarray:
        freqs = self.modes(radius, tension, density, n_modes)
        if len(freqs) < 2 or freqs[0] == 0:
            return np.ones(len(freqs))
        return freqs / freqs[0]


@dataclass
class BarResonance:
    """Euler-Bernoulli beam model for mallet percussion.

    f_n = (beta_n^2 / (2 pi L^2)) sqrt(E I / (rho A))
    """

    _free_free_betas: list = field(
        default_factory=lambda: [4.730, 7.853, 10.996, 14.137, 17.279, 20.420, 23.562, 26.704]
    )
    _clamped_free_betas: list = field(
        default_factory=lambda: [1.875, 4.694, 7.855, 10.996, 14.137, 17.279, 20.420, 23.562]
    )

    def modes(
        self,
        length: float,
        young_mod: float,
        density: float,
        area_moment: float,
        n_modes: int = 8,
        boundary: str = "free-free",
    ) -> np.ndarray:
        if boundary == "clamped-free":
            betas = self._clamped_free_betas
        else:
            betas = self._free_free_betas
        n = min(n_modes, len(betas))
        b = np.array(betas[:n])
        coeff = 1.0 / (2.0 * np.pi * length**2)
        freqs = coeff * b**2 * np.sqrt(young_mod * area_moment / (density))
        return freqs

    def frequency_ratio(
        self, length: float, young_mod: float, density: float, area_moment: float,
        n_modes: int = 6, boundary: str = "free-free",
    ) -> np.ndarray:
        freqs = self.modes(length, young_mod, density, area_moment, n_modes, boundary)
        if len(freqs) == 0 or freqs[0] == 0:
            return np.ones(max(1, len(freqs)))
        return freqs / freqs[0]


@dataclass
class ResonatorBank:
    """Combine multiple resonators into a composite sound source."""

    resonators: list = field(default_factory=list)

    def add(self, frequencies: np.ndarray, amplitudes: np.ndarray, damping: float) -> None:
        self.resonators.append((frequencies.copy(), amplitudes.copy(), damping))

    def synthesize(
        self, excitation: np.ndarray, sr: int, duration: float
    ) -> np.ndarray:
        n_samples = int(sr * duration)
        output = np.zeros(n_samples)
        t = np.arange(n_samples) / sr
        for freqs, amps, damping in self.resonators:
            for f, a in zip(freqs, amps):
                mode = a * np.exp(-damping * t) * np.sin(2 * np.pi * f * t)
                output[:len(mode)] += mode[:n_samples]
        if len(excitation) > 0:
            from numpy.fft import rfft, irfft
            n_fft = max(len(excitation), n_samples)
            exc_fft = rfft(excitation, n=n_fft)
            out_fft = rfft(output, n=n_fft)
            convolved = irfft(exc_fft * out_fft, n=n_fft)[:n_samples]
            peak = np.max(np.abs(convolved))
            if peak > 0:
                convolved /= peak
            return convolved
        peak = np.max(np.abs(output))
        if peak > 0:
            output /= peak
        return output


@dataclass
class CoupledStringModel:
    """Two strings coupled via a bridge — energy transfer model."""

    def simulate(
        self,
        f1: float,
        f2: float,
        coupling: float,
        duration: float,
        sr: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(sr * duration)
        dt = 1.0 / sr
        x1 = np.zeros(n)
        x2 = np.zeros(n)
        v1, v2 = 1.0, 0.0
        p1, p2 = 0.0, 0.0
        omega1 = 2.0 * np.pi * f1
        omega2 = 2.0 * np.pi * f2
        damping = 0.5
        for i in range(n):
            a1 = -omega1**2 * p1 - coupling * (p1 - p2) - damping * v1
            a2 = -omega2**2 * p2 - coupling * (p2 - p1) - damping * v2
            v1 += a1 * dt
            v2 += a2 * dt
            p1 += v1 * dt
            p2 += v2 * dt
            x1[i] = p1
            x2[i] = p2
        for sig in (x1, x2):
            peak = np.max(np.abs(sig))
            if peak > 0:
                sig /= peak
        return x1, x2

    def energy_transfer_curve(
        self, f1: float, f2: float, coupling: float, duration: float, sr: int,
        window_ms: float = 50.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        x1, x2 = self.simulate(f1, f2, coupling, duration, sr)
        win = int(sr * window_ms / 1000.0)
        if win < 1:
            win = 1
        n_frames = len(x1) // win
        e1 = np.zeros(n_frames)
        e2 = np.zeros(n_frames)
        for i in range(n_frames):
            s = i * win
            e = s + win
            e1[i] = np.mean(x1[s:e] ** 2)
            e2[i] = np.mean(x2[s:e] ** 2)
        total = e1 + e2
        total[total == 0] = 1.0
        return e1 / total, e2 / total
